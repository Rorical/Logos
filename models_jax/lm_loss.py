"""LM-head cross-entropy loss helpers for Flax.

Chunked CE avoids materializing full (B*T, vocab) logits.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import optax


def chunked_linear_cross_entropy(
    hidden: jnp.ndarray,  # (B, T, d_model)
    weight: jnp.ndarray,  # (vocab, d_model) — tied with token_emb
    labels: jnp.ndarray,  # (B, T) — labels[i] predicts labels[i+1]
    *,
    chunk_size: int = 1024,
    ignore_index: int = -100,
) -> jnp.ndarray:
    """Compute tied LM-head CE without materializing all logits."""
    seq_len = hidden.shape[1]
    if seq_len < 2:
        return jnp.zeros((), dtype=jnp.float32)

    loss_t = seq_len - 1
    targets = jnp.full((hidden.shape[0], seq_len), ignore_index, dtype=labels.dtype)
    targets = targets.at[:, :loss_t].set(labels[:, 1: 1 + loss_t])

    hidden_flat = hidden.reshape(-1, hidden.shape[-1])
    targets_flat = targets.reshape(-1)

    valid = targets_flat != ignore_index
    safe_targets = jnp.clip(targets_flat, 0)
    count = valid.sum().astype(jnp.float32).clip(1.0)

    def chunk_fn(start, _):
        end = min(start + chunk_size, hidden_flat.shape[0])
        h_chunk = hidden_flat[start:end]
        t_chunk = safe_targets[start:end]
        v_chunk = valid[start:end]

        logits = (h_chunk @ weight.T).astype(jnp.float32)
        log_z = jax.nn.logsumexp(logits, axis=-1)
        target_logits = logits[jnp.arange(end - start), t_chunk]
        chunk_loss = ((log_z - target_logits) * v_chunk).sum()
        return chunk_loss

    starts = jnp.arange(0, hidden_flat.shape[0], chunk_size)
    losses = jax.lax.map(lambda s: chunk_fn(s, None), starts)
    return losses.sum() / count


def standard_lm_cross_entropy(
    logits: jnp.ndarray,  # (B, T, vocab)
    labels: jnp.ndarray,   # (B, T)
    *,
    ignore_index: int = -100,
) -> jnp.ndarray:
    """Standard next-token cross-entropy."""
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
    flat_labels = shift_labels.reshape(-1)

    valid = flat_labels != ignore_index
    flat_labels_safe = jnp.where(valid, flat_labels, 0)

    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
        flat_logits, flat_labels_safe
    )
    loss_sum = (per_token_loss * valid).sum()
    count = valid.sum().clip(1)
    return loss_sum / count
