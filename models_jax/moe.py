"""Mixture-of-Experts layer with DeepSeek-style aux-loss-free bias balancing.

Static-shape scatter/gather compatible with jax.jit. Bias stored as
a Flax variable so it persists across training steps.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from models_jax.base import default_kernel_init, SwiGLU


class SparseExpertBank(nn.Module):
    """Packed sparse-expert weights: (E, d_ff, d_model) gate/up, (E, d_model, d_ff) down."""
    num_experts: int
    d_model: int
    d_ff: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, expert_in: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        """expert_in: (E, C, d_model) → (E, C, d_model)."""
        E, C, D = expert_in.shape
        DF = self.d_ff
        kinit = default_kernel_init()

        w_gate = self.param("w_gate", kinit, (E, DF, D))
        w_up = self.param("w_up", kinit, (E, DF, D))
        w_down = self.param("w_down", kinit, (E, D, DF))

        h_gate = jnp.einsum("ecd,efd->ecf", expert_in, w_gate)
        h_up = jnp.einsum("ecd,efd->ecf", expert_in, w_up)
        hidden = jax.nn.silu(h_gate) * h_up
        hidden = nn.Dropout(rate=self.dropout)(hidden, deterministic=deterministic)
        return jnp.einsum("ecf,edf->ecd", hidden, w_down)


class Router(nn.Module):
    """Linear router: d_model → num_experts, no bias."""
    num_experts: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(
            self.num_experts, use_bias=False,
            kernel_init=default_kernel_init(),
        )(x)


class MoELayer(nn.Module):
    d_model: int
    num_shared_experts: int
    num_sparse_experts: int
    top_k: int
    expert_d_ff: int
    dropout: float = 0.0
    bias_update_rate: float = 0.01
    capacity_factor: float = 2.0
    num_loops: int = 1
    diversity_factor: float = 0.0

    def setup(self):
        self._router = Router(self.num_sparse_experts)
        self._bias = self.variable(
            "moe_state", "bias",
            jnp.zeros, (self.num_loops, self.num_sparse_experts), jnp.float32,
        )

        self._shared = [
            SwiGLU(d_ff=self.expert_d_ff, dropout=self.dropout,
                   name=f"shared_{i}")
            for i in range(self.num_shared_experts)
        ]

        self._sparse = SparseExpertBank(
            num_experts=self.num_sparse_experts,
            d_model=self.d_model,
            d_ff=self.expert_d_ff,
            dropout=self.dropout,
            name="sparse",
        )

    @property
    def bias(self):
        return self._bias.value

    def __call__(
        self, x: jnp.ndarray, *, loop_idx: int = 0, deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        B, T, D = x.shape
        N = B * T
        E = self.num_sparse_experts
        K = self.top_k
        dtype = x.dtype

        router_logits = self._router(x) + self._bias.value[loop_idx].astype(dtype)

        shared_out = sum(
            expert(x, deterministic=deterministic)
            for expert in self._shared
        ) / max(1, self.num_shared_experts)

        capacity = max(1, int(N * K * self.capacity_factor / E))
        C = capacity

        x_flat = x.reshape(N, D)
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        topk_probs, topk_indices = jax.lax.top_k(router_probs, K)
        topk_probs = topk_probs / (
            topk_probs.sum(axis=-1, keepdims=True) + 1e-9
        )

        topk_flat = topk_indices.reshape(-1)
        probs_flat = topk_probs.reshape(-1)
        token_ids = jnp.broadcast_to(
            jnp.arange(N)[:, None], (N, K)
        ).reshape(-1)

        # Sort by expert ID
        sort_idx = jnp.argsort(topk_flat)
        sorted_exp = topk_flat[sort_idx]
        sorted_tok = token_ids[sort_idx]
        sorted_gate = probs_flat[sort_idx]

        M = sorted_exp.shape[0]
        positions = jnp.arange(M, dtype=jnp.int32)
        diff = sorted_exp[1:] != sorted_exp[:-1]
        is_first = jnp.concatenate([
            jnp.ones(1, dtype=jnp.bool_), diff,
        ])
        group_starts = jnp.maximum.accumulate(
            positions * is_first.astype(jnp.int32)
        )
        slot_indices = positions - group_starts

        valid = slot_indices < C

        # Scatter to (E, C, D): use 1D scatter for unique (expert*C + slot) pairs
        # Only scatter valid slots (< C)
        valid_exp = jnp.where(valid, sorted_exp, 0)
        valid_slot = jnp.where(valid, slot_indices, 0)
        flat_idx = valid_exp * C + valid_slot

        sorted_x = x_flat[sorted_tok]

        expert_in = jnp.zeros((E * C, D), dtype=dtype)
        expert_gate = jnp.zeros((E * C,), dtype=dtype)
        expert_mask = jnp.zeros((E * C,), dtype=jnp.bool_)

        # Scatter: use add for safety (no duplicate handling needed since slots are unique)
        expert_in = expert_in.at[flat_idx].add(
            jnp.where(valid[:, None], sorted_x, 0.0)
        )
        expert_gate = expert_gate.at[flat_idx].add(
            jnp.where(valid, sorted_gate, 0.0)
        )
        expert_mask = expert_mask.at[flat_idx].set(valid)

        expert_in = expert_in.reshape(E, C, D)
        expert_gate = expert_gate.reshape(E, C)
        expert_mask = expert_mask.reshape(E, C)

        expert_out = self._sparse(expert_in, deterministic=deterministic)

        # Gather back: multiply by gate, scatter back to token positions
        gated = expert_out * expert_gate[..., None]
        gated_flat = gated.reshape(E * C, D)
        mask_flat = expert_mask.reshape(-1)

        # Map each (e, c) pair back to its token
        # Build lookup: for each flat positions, which token does it belong to
        tok_lookup = jnp.full((E, C), N, dtype=jnp.int32)
        tok_lookup = tok_lookup.at[valid_exp, valid_slot].set(
            jnp.where(valid, sorted_tok, N)
        )
        tok_flat = tok_lookup.reshape(-1)

        # Scatter to output: sum contributions per token
        safe_dst = jnp.clip(tok_flat, 0, N)
        sparse_out = jnp.zeros((N + 1, D), dtype=dtype)
        sparse_out = sparse_out.at[safe_dst].add(
            jnp.where(mask_flat[:, None], gated_flat, 0.0)
        )
        sparse_out = sparse_out[:N].reshape(B, T, D)

        aux_loss = jnp.zeros((), dtype=dtype)
        return shared_out + sparse_out, aux_loss, topk_indices


def expert_load_from_topk(topk_indices: jnp.ndarray, num_experts: int) -> jnp.ndarray:
    return jnp.bincount(topk_indices.reshape(-1), minlength=num_experts).astype(jnp.float32)


def update_bias(
    bias: jnp.ndarray,
    topk_indices: jnp.ndarray,
    num_sparse_experts: int,
    bias_update_rate: float,
    loop_idx: int = 0,
) -> jnp.ndarray:
    load = expert_load_from_topk(topk_indices, num_sparse_experts)
    total = load.sum() + 1e-9
    load_fraction = load / total
    target_fraction = 1.0 / num_sparse_experts
    update = bias_update_rate * (target_fraction - load_fraction)
    return bias.at[loop_idx].add(update)


def update_bias_per_loop(
    bias: jnp.ndarray,
    topk_per_loop: list,  # list of topk_indices tensors
    num_sparse_experts: int,
    bias_update_rate: float,
    diversity_factor: float,
) -> jnp.ndarray:
    num_loops = len(topk_per_loop)
    loads = jnp.stack([
        expert_load_from_topk(t, num_sparse_experts)
        for t in topk_per_loop
    ], axis=0)
    loads = loads / (loads.sum(axis=1, keepdims=True) + 1e-9)
    target = 1.0 / num_sparse_experts

    if num_loops > 1 and diversity_factor > 0:
        agg_load = loads.mean(axis=0)
        agg_term = target - agg_load
        other_mean = (loads.sum(axis=0, keepdims=True) - loads) / (num_loops - 1)
        diversity_term = -diversity_factor * (other_mean - target)
        update = bias_update_rate * (agg_term[None, :] + diversity_term)
    else:
        update = bias_update_rate * (target - loads)

    return bias + update
