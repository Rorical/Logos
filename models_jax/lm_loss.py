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
    safe_targets = jnp.clip(targets_flat, min=0)
    count = valid.sum().astype(jnp.float32).clip(min=1.0)

    N = hidden_flat.shape[0]
    pad = (-N) % chunk_size
    if pad > 0:
        hidden_flat = jnp.pad(hidden_flat, ((0, pad), (0, 0)))
        safe_targets = jnp.pad(safe_targets, (0, pad))
        valid = jnp.pad(valid, (0, pad))  # bool pads with False -> zero contribution

    n_chunks = (N + pad) // chunk_size
    h_chunked = hidden_flat.reshape(n_chunks, chunk_size, -1)
    t_chunked = safe_targets.reshape(n_chunks, chunk_size)
    v_chunked = valid.reshape(n_chunks, chunk_size).astype(jnp.float32)

    # Python-unrolled chunk loop: backprop through lax.scan with a closure-
    # captured parameter (here `weight`) has produced fragile HLO under
    # autograd on TPU. n_chunks is concrete (derived from static shapes),
    # so unrolling at trace time is safe and the graph stays bounded.
    total = jnp.zeros((), dtype=jnp.float32)
    for i in range(n_chunks):
        h = h_chunked[i]
        t = t_chunked[i]
        v = v_chunked[i]
        logits = (h @ weight.T).astype(jnp.float32)
        log_z = jax.nn.logsumexp(logits, axis=-1)
        target_logits = jnp.take_along_axis(logits, t[:, None], axis=-1).squeeze(-1)
        total = total + ((log_z - target_logits) * v).sum()

    return total / count


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
    count = valid.sum().clip(min=1)
    return loss_sum / count
