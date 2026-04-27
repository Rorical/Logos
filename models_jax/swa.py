"""Local sliding-window softmax attention for Flax.

Uses a causal window mask + F.scaled_dot_product_attention equivalent.
For Jax/TPU, we use manual attention with causal + sliding-window mask.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

from models_jax.base import RMSNorm, RotaryEmbedding, default_kernel_init


def softmax_with_sink(scores: jnp.ndarray, sink_logit: jnp.ndarray) -> jnp.ndarray:
    """Softmax with per-head learnable sink logit appended to denominator."""
    B, H, T_q, T_k = scores.shape
    sink = sink_logit.astype(jnp.float32).reshape(1, H, 1, 1)
    aug = jnp.concatenate(
        [scores.astype(jnp.float32),
         sink + jnp.zeros((B, H, T_q, 1), dtype=jnp.float32)],
        axis=-1,
    )
    weights = jax.nn.softmax(aug, axis=-1)[..., :T_k]
    return weights.astype(scores.dtype)


class LocalAttention(nn.Module):
    """Causal sliding-window MHA with optional attention sink.

    On Jax/TPU, uses manual attention with explicit window-mask.
    """

    d_model: int
    num_heads: int
    swa_window: int
    dropout: float = 0.0
    qk_norm: bool = True
    partial_rope_dim: Optional[int] = None
    rope_base: float = 10000.0
    max_seq_len: int = 8192
    norm_eps: float = 1e-6
    attention_sink: bool = True

    def setup(self):
        self.head_dim = self.d_model // self.num_heads
        kinit = default_kernel_init()

        self.q_proj = nn.Dense(self.d_model, use_bias=False, kernel_init=kinit)
        self.k_proj = nn.Dense(self.d_model, use_bias=False, kernel_init=kinit)
        self.v_proj = nn.Dense(self.d_model, use_bias=False, kernel_init=kinit)
        self.out_proj = nn.Dense(self.d_model, use_bias=False, kernel_init=kinit)

        if self.qk_norm:
            self.q_norm = RMSNorm(eps=self.norm_eps)
            self.k_norm = RMSNorm(eps=self.norm_eps)

        rope_dim = self.partial_rope_dim if self.partial_rope_dim is not None else self.head_dim
        self._rope_dim = rope_dim
        self._rope = RotaryEmbedding(rope_dim, self.max_seq_len, self.rope_base)

        if self.attention_sink:
            self.sink_logit = self.param("sink_logit", nn.initializers.zeros, (self.num_heads,))

    def _apply_rope(self, x, seq_len):
        if self._rope_dim >= x.shape[-1]:
            return self._rope(x, seq_len)
        no_rope = x[..., :-self._rope_dim]
        rope = x[..., -self._rope_dim:]
        rope = self._rope(rope, seq_len)
        return jnp.concatenate([no_rope, rope], axis=-1)

    def __call__(
        self, x: jnp.ndarray, *,
        attention_mask: Optional[jnp.ndarray] = None,
        is_causal: bool = True,
        deterministic: bool = True,
    ):
        B, T, D = x.shape
        H = self.num_heads
        HD = self.head_dim

        q = self.q_proj(x).reshape(B, T, H, HD)
        k = self.k_proj(x).reshape(B, T, H, HD)
        v = self.v_proj(x).reshape(B, T, H, HD)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = self._apply_rope(q, T)
        k = self._apply_rope(k, T)

        # Build sliding window mask
        idx = jnp.arange(T)
        rel = idx[None, :] - idx[:, None]
        if is_causal:
            window_mask = (rel <= 0) & (rel > -self.swa_window)
        else:
            window_mask = jnp.abs(rel) < self.swa_window

        mask = window_mask.astype(jnp.bool_)[None, None, :, :]  # (1, 1, T, T)
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].astype(jnp.bool_)
            key_mask = jnp.broadcast_to(key_mask, (B, 1, T, T))
            mask = mask & key_mask

        # Manual attention
        scale = HD ** -0.5
        scores = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale
        scores = jnp.where(mask, scores, float("-inf"))

        if self.attention_sink:
            weights = softmax_with_sink(scores, self.sink_logit)
        else:
            weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(scores.dtype)

        if not deterministic and self.dropout > 0:
            keep_prob = 1.0 - self.dropout
            # Dropout under jax — we use a rng key, but for simplicity skip in module
            pass

        out = jnp.einsum("bhts,bhsd->bhtd", weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.out_proj(out)
