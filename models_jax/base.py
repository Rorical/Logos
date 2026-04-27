"""Core building blocks for Logos in Flax: RMSNorm, SwiGLU, RotaryEmbedding."""

from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn


def default_kernel_init():
    return nn.initializers.normal(stddev=0.02)


def count_parameters(params) -> int:
    return sum(
        leaf.size for leaf in jax.tree_util.tree_leaves(params)
        if isinstance(leaf, jnp.ndarray)
    )


def model_summary(params) -> str:
    lines = ["Model Summary", "=" * 50]
    total = 0
    for key, subtree in params.items():
        n = sum(
            leaf.size for leaf in jax.tree_util.tree_leaves(subtree)
            if isinstance(leaf, jnp.ndarray)
        )
        total += n
        lines.append(f"{key:25s} {n:>15,} params")
    lines.append("-" * 50)
    lines.append(f"{'Total':25s} {total:>15,} params")
    return "\n".join(lines)


class RMSNorm(nn.Module):
    """RMSNorm matching PyTorch nn.RMSNorm: eps inside sqrt."""
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        rrms = jax.lax.rsqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + self.eps)
        return (x_f32 * rrms * scale.astype(jnp.float32)).astype(dtype)


class RotaryEmbedding:
    """Functional RoPE: precomputed cos/sin table, matches PyTorch version.

    ``inv_freq = 1/(base**(arange(0,d,2)/d))``, ``freqs = t @ inv_freq``,
    ``emb = cat(freqs, freqs)``, cos/sin stored for fast lookup.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
        t = jnp.arange(max_seq_len, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        self.cos = emb.cos()
        self.sin = emb.sin()

    def rotate_half(self, x: jnp.ndarray) -> jnp.ndarray:
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)

    def __call__(self, x: jnp.ndarray, seq_len: int) -> jnp.ndarray:
        if seq_len > self.cos.shape[0]:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds precomputed max_seq_len "
                f"({self.cos.shape[0]})"
            )
        cos = self.cos[:seq_len, :].astype(x.dtype)
        sin = self.sin[:seq_len, :].astype(x.dtype)
        return x * cos + self.rotate_half(x) * sin


class SwiGLU(nn.Module):
    """SwiGLU FFN: silu(gate(x)) * up(x) -> down."""

    d_ff: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        d_model = x.shape[-1]
        gate = nn.Dense(self.d_ff, use_bias=False, kernel_init=default_kernel_init(), name="w_gate")(x)
        up = nn.Dense(self.d_ff, use_bias=False, kernel_init=default_kernel_init(), name="w_up")(x)
        x = jax.nn.silu(gate) * up
        x = nn.Dense(d_model, use_bias=False, kernel_init=default_kernel_init(), name="w_down")(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        return x
