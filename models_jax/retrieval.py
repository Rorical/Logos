"""Snapshot Retrieval — top-k attention over compressed KDA state snapshots.

RoPE encodes relative distance t — snap_position. out_up is zero-initialised.
"""

from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

from models_jax.base import RMSNorm, default_kernel_init


def _compute_rope_setup(
    head_dim: int,
    rope_base: float,
    scaling_type: str,
    scaling_factor: float,
    original_max_position: int,
    yarn_beta_fast: float,
    yarn_beta_slow: float,
):
    d = head_dim
    arange = jnp.arange(0, d, 2, dtype=jnp.float32)
    inv_freq_base = 1.0 / (rope_base ** (arange / d))

    if scaling_type == "none" or scaling_factor == 1.0:
        return inv_freq_base, 1.0

    if scaling_type == "ntk":
        new_base = rope_base * (scaling_factor ** (d / max(2, d - 2)))
        inv_freq = 1.0 / (new_base ** (arange / d))
        return inv_freq, 1.0

    log_base = math.log(rope_base)
    n_pairs = d // 2

    def find_correction_dim(num_rot):
        return (d * math.log(original_max_position / (num_rot * 2 * math.pi))) / (2 * log_base)

    low = max(int(math.floor(find_correction_dim(yarn_beta_fast))), 0)
    high = min(int(math.ceil(find_correction_dim(yarn_beta_slow))), n_pairs - 1)
    if low == high:
        high = low + 1

    idx = jnp.arange(n_pairs, dtype=jnp.float32)
    ramp = jnp.clip((idx - low) / (high - low), 0.0, 1.0)

    inv_freq_extrap = inv_freq_base
    inv_freq_interp = inv_freq_base / scaling_factor
    inv_freq = inv_freq_extrap * (1.0 - ramp) + inv_freq_interp * ramp

    attention_scaling = 0.1 * math.log(scaling_factor) + 1.0
    return inv_freq, attention_scaling


class SnapshotRetrieval(nn.Module):
    d_model: int
    num_heads: int
    mem_head_dim: int
    latent_dim: int
    top_k: int
    mem_latent_dim: int = 128
    rope_base: float = 10000.0
    rope_scaling_type: str = "none"
    rope_scaling_factor: float = 1.0
    rope_original_max_position: int = 2048
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    partial_rope_dim: Optional[int] = None
    attention_sink: bool = True
    norm_eps: float = 1e-6

    def setup(self):
        H = self.num_heads
        Dh = self.mem_head_dim
        kinit = default_kernel_init()

        self.q_down = nn.Dense(self.mem_latent_dim, use_bias=False, kernel_init=kinit)
        self.q_up = nn.Dense(H * Dh, use_bias=False, kernel_init=kinit)
        self.gate_down = nn.Dense(self.mem_latent_dim, use_bias=False, kernel_init=kinit)
        self.gate_up = nn.Dense(H * Dh, use_bias=False, kernel_init=kinit)
        self.out_down = nn.Dense(self.mem_latent_dim, use_bias=False, kernel_init=kinit)
        self.out_up = nn.Dense(self.d_model, use_bias=False,
                               kernel_init=nn.initializers.zeros)

        self.k_up = nn.Dense(Dh, use_bias=False, kernel_init=kinit)
        self.v_up = nn.Dense(Dh, use_bias=False, kernel_init=kinit)

        self.q_norm = RMSNorm(eps=self.norm_eps)
        self.k_norm = RMSNorm(eps=self.norm_eps)

        rope_dim = self.partial_rope_dim if self.partial_rope_dim is not None else Dh
        self.rope_dim = rope_dim

        inv_freq, attn_scale = _compute_rope_setup(
            head_dim=rope_dim,
            rope_base=self.rope_base,
            scaling_type=self.rope_scaling_type,
            scaling_factor=self.rope_scaling_factor,
            original_max_position=self.rope_original_max_position,
            yarn_beta_fast=self.yarn_beta_fast,
            yarn_beta_slow=self.yarn_beta_slow,
        )
        self.rope_inv_freq = self.variable(
            "constants", "rope_inv_freq", lambda *_args: inv_freq, (),
        )
        self._rope_attention_scaling = attn_scale

        if self.attention_sink:
            self.sink_logit = self.param(
                "sink_logit", nn.initializers.zeros, (H,),
            )

    @staticmethod
    def _rotate_half(x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)

    def _apply_rope(self, x, positions, inv_freq):
        """Apply RoPE to the last rope_dim channels of x."""
        if self.rope_dim < x.shape[-1]:
            no_rope = x[..., :-self.rope_dim]
            target = x[..., -self.rope_dim:]
        else:
            no_rope = None
            target = x

        freqs = positions.astype(jnp.float32)[..., None] * inv_freq.astype(jnp.float32)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos = jnp.cos(emb).astype(target.dtype)
        sin = jnp.sin(emb).astype(target.dtype)
        rotated = target * cos + self._rotate_half(target) * sin
        if self._rope_attention_scaling != 1.0:
            rotated = rotated * self._rope_attention_scaling

        if no_rope is None:
            return rotated
        return jnp.concatenate([no_rope, rotated], axis=-1)

    def __call__(
        self,
        x: jnp.ndarray,
        snapshots: Optional[jnp.ndarray],
        snap_positions: Optional[jnp.ndarray],
        token_offset: int = 0,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ):
        if snapshots is None or snapshots.shape[1] == 0:
            return jnp.zeros_like(x)

        B, T, D = x.shape
        _, N_snaps, H, r = snapshots.shape
        Dh = self.mem_head_dim

        q = self.q_up(self.q_down(x)).reshape(B, T, H, Dh)
        q = self.q_norm(q).transpose(0, 2, 1, 3)  # (B, H, T, Dh)

        k = self.k_up(snapshots)      # (B, N_snaps, H, Dh)
        v = self.v_up(snapshots)      # (B, N_snaps, H, Dh)
        k = self.k_norm(k).transpose(0, 2, 1, 3)  # (B, H, N_snaps, Dh)
        v = v.transpose(0, 2, 1, 3)             # (B, H, N_snaps, Dh)

        inv_freq = self.rope_inv_freq.value
        t_abs = token_offset + jnp.arange(T, dtype=jnp.int32)
        q = self._apply_rope(q, t_abs, inv_freq)
        k = self._apply_rope(k, snap_positions.astype(jnp.int32), inv_freq)

        scores = jnp.einsum("bhtd,bhnd->bhtn", q, k) / (Dh ** 0.5)

        # Causal mask: snapshot at position p visible to token t iff p < t
        causal_mask = snap_positions[None, :] < t_abs[:, None]  # (T, N_snaps)
        causal_mask = causal_mask[None, None, :, :]  # (1, 1, T, N_snaps)
        scores = jnp.where(causal_mask, scores, float("-inf"))

        # Top-k selection
        k_sel = min(self.top_k, N_snaps)
        if k_sel < N_snaps:
            top_scores, top_idx = jax.lax.top_k(scores, k_sel)
            # Build mask of selected positions
            idx_grid = jnp.arange(N_snaps)[None, None, None, :]
            selected = jnp.zeros(scores.shape, dtype=jnp.bool_)
            for kk in range(k_sel):
                selected = selected | (idx_grid == top_idx[..., kk, None])
            scores = jnp.where(selected, scores, float("-inf"))

        if self.attention_sink:
            scores_f = scores.astype(jnp.float32)
            sink_f = self.sink_logit.astype(jnp.float32)
            aug = jnp.concatenate([
                scores_f,
                sink_f.reshape(1, H, 1, 1) + jnp.zeros((B, H, T, 1), dtype=jnp.float32),
            ], axis=-1)
            weights = jax.nn.softmax(aug, axis=-1)[..., :N_snaps].astype(v.dtype)
        else:
            all_masked = jnp.isinf(scores).all(axis=-1, keepdims=True)
            scores_safe = jnp.where(all_masked, 0.0, scores)
            weights = jax.nn.softmax(
                scores_safe.astype(jnp.float32), axis=-1,
            ).astype(v.dtype)
            weights = jnp.where(all_masked, 0.0, weights)

        out = jnp.einsum("bhtn,bhnd->bhtd", weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, H * Dh)

        gate = jax.nn.sigmoid(self.gate_up(self.gate_down(x)))
        out = out * gate
        out = self.out_up(self.out_down(out))

        if attention_mask is not None:
            out = out * attention_mask[..., None].astype(out.dtype)
        return out
