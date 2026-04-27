"""Block Attention Residual (MoonshotAI): depth-wise softmax over completed blocks.

Replaces standard additive residuals. proj is zero-initialised for uniform start.
"""

from __future__ import annotations

from typing import List

import jax
import jax.numpy as jnp
from flax import linen as nn

from models_jax.base import RMSNorm


class BlockAttentionResidual(nn.Module):
    """Depth-wise softmax over completed block states + current partial.

    ``proj`` zero-initialised → uniform average at start, matching
    standard residual sum in expectation. Softmax in fp32.
    """

    d_model: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, blocks: List[jnp.ndarray], partial_block: jnp.ndarray) -> jnp.ndarray:
        if len(blocks) == 0:
            return partial_block

        values = jnp.stack(blocks + [partial_block], axis=0)  # (N, B, T, D)
        norm = RMSNorm(eps=self.eps, name="norm")
        keys = norm(values)

        proj = self.param("proj", nn.initializers.zeros, (self.d_model,))
        logits = (keys * proj).sum(axis=-1, keepdims=True)  # (N, B, T, 1)

        weights = jax.nn.softmax(logits.astype(jnp.float32), axis=0).astype(values.dtype)
        return jnp.sum(weights * values, axis=0)
