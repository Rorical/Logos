"""Memory-efficient LM-head cross entropy helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _autocast_state(device_type: str) -> Tuple[bool, torch.dtype | None]:
    try:
        enabled = torch.is_autocast_enabled(device_type)
    except TypeError:
        enabled = torch.is_autocast_enabled()
    if not enabled:
        return False, None
    try:
        return True, torch.get_autocast_dtype(device_type)
    except AttributeError:
        if device_type == "cuda":
            return True, torch.get_autocast_gpu_dtype()
        if device_type == "cpu":
            return True, torch.get_autocast_cpu_dtype()
    return True, None


def _chunk_logits(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype | None,
) -> torch.Tensor:
    device_type = hidden.device.type
    if autocast_dtype is not None:
        with torch.amp.autocast(
            device_type=device_type,
            enabled=autocast_enabled,
            dtype=autocast_dtype,
        ):
            return hidden @ weight.t()
    if autocast_enabled:
        with torch.amp.autocast(device_type=device_type, enabled=True):
            return hidden @ weight.t()
    if hidden.dtype != weight.dtype:
        return hidden @ weight.to(hidden.dtype).t()
    return hidden @ weight.t()


def _build_padded_targets(
    labels: torch.Tensor,
    seq_len: int,
    loss_t: int,
    ignore_index: int,
) -> torch.Tensor:
    """Pack labels into a ``(B*seq_len,)`` target tensor aligned with a
    ``hidden.reshape(B*seq_len, D)`` view.

    The last token per sequence (``t == seq_len - 1``) gets ``ignore_index``
    so a flat view of the full ``(B, T, D)`` activation can be passed to the
    kernels directly — no slice-and-reshape copy required.
    """
    batch = labels.size(0)
    targets = labels.new_full((batch, seq_len), ignore_index)
    targets[:, :loss_t] = labels[:, 1: 1 + loss_t]
    return targets.reshape(-1)


@torch.library.custom_op(
    "logos::chunked_linear_cross_entropy", mutates_args=(),
)
def _chunked_lce_op(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int,
    ignore_index: int,
    autocast_enabled: bool,
    autocast_dtype: Optional[torch.dtype],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len = hidden.size(1)
    loss_t = seq_len - 1
    # Pad targets at the (T-1) position per batch so a flat view of the full
    # (B, T, D) activation aligns with target indices. Avoids the
    # hidden[:, :loss_t, :].reshape(...) copy the previous layout required.
    targets = _build_padded_targets(labels, seq_len, loss_t, ignore_index)
    if hidden.is_contiguous():
        hidden_flat = hidden.reshape(-1, hidden.size(-1))
    else:
        hidden_flat = hidden.contiguous().reshape(-1, hidden.size(-1))
    chunk_size = max(1, int(chunk_size))

    loss_sum = torch.zeros((), device=hidden.device, dtype=torch.float32)
    count = (targets != ignore_index).sum().to(torch.float32)
    for start in range(0, hidden_flat.size(0), chunk_size):
        end = min(start + chunk_size, hidden_flat.size(0))
        target_chunk = targets[start:end]
        valid = target_chunk != ignore_index
        safe_target = target_chunk.clamp_min(0)
        logits = _chunk_logits(
            hidden_flat[start:end],
            weight,
            autocast_enabled,
            autocast_dtype,
        ).float()
        log_z = torch.logsumexp(logits, dim=-1)
        target_logits = logits.gather(1, safe_target[:, None]).squeeze(1)
        loss_sum = loss_sum + ((log_z - target_logits) * valid).sum()

    loss = loss_sum / count.clamp_min(1.0)
    empty_lse = hidden.new_empty((0,), dtype=torch.float32)
    return loss, empty_lse, count


@_chunked_lce_op.register_fake
def _chunked_lce_fake(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int,
    ignore_index: int,
    autocast_enabled: bool,
    autocast_dtype: Optional[torch.dtype],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_rows = hidden.size(0) * hidden.size(1)
    return (
        hidden.new_empty((), dtype=torch.float32),
        hidden.new_empty((n_rows,), dtype=torch.float32),
        hidden.new_empty((), dtype=torch.float32),
    )


def _chunked_lce_setup_context(ctx, inputs, output):
    (
        hidden,
        weight,
        labels,
        chunk_size,
        ignore_index,
        autocast_enabled,
        autocast_dtype,
    ) = inputs
    _loss, lse, count = output
    ctx.save_for_backward(hidden, weight, labels, lse, count)
    ctx.chunk_size = max(1, int(chunk_size))
    ctx.ignore_index = int(ignore_index)
    ctx.autocast_enabled = bool(autocast_enabled)
    ctx.autocast_dtype = autocast_dtype


def _chunked_lce_backward(
    ctx,
    grad_output: torch.Tensor,
    _grad_lse: Optional[torch.Tensor] = None,
    _grad_count: Optional[torch.Tensor] = None,
):
    hidden, weight, labels, _lse, count = ctx.saved_tensors
    chunk_size = ctx.chunk_size
    ignore_index = ctx.ignore_index

    seq_len = hidden.size(1)
    loss_t = seq_len - 1
    d_model = hidden.size(-1)
    targets = _build_padded_targets(
        labels, seq_len, loss_t, ignore_index,
    )
    if hidden.is_contiguous():
        hidden_flat = hidden.reshape(-1, d_model)
    else:
        hidden_flat = hidden.contiguous().reshape(-1, d_model)
    # ``count`` was already produced by the forward and saved in ctx; the
    # previous code retrieved it as ``_count`` and recomputed
    # ``(targets != ignore_index).sum()`` — a full GPU reduction that
    # forced a host sync on every backward call.

    grad_hidden = None
    grad_hidden_flat = None
    if ctx.needs_input_grad[0]:
        grad_hidden = torch.zeros_like(hidden)
        grad_hidden_flat = grad_hidden.view(-1, d_model)
    grad_weight = None
    if ctx.needs_input_grad[1]:
        grad_weight = torch.zeros_like(weight)

    scale = (grad_output.float() / count.clamp_min(1.0)).to(torch.float32)
    weight_f = weight.float()
    for start in range(0, hidden_flat.size(0), chunk_size):
        end = min(start + chunk_size, hidden_flat.size(0))
        h_chunk = hidden_flat[start:end]
        target_chunk = targets[start:end]
        valid = target_chunk != ignore_index
        valid_f = valid.to(torch.float32)
        safe_target = target_chunk.clamp_min(0)

        logits = _chunk_logits(
            h_chunk,
            weight,
            ctx.autocast_enabled,
            ctx.autocast_dtype,
        ).float()
        d_logits = torch.softmax(logits, dim=-1)
        d_logits = d_logits * valid_f[:, None]
        rows = torch.arange(
            end - start, device=hidden.device, dtype=torch.long,
        )
        d_logits[rows, safe_target] -= valid_f
        d_logits = d_logits * scale

        if grad_hidden_flat is not None:
            grad_hidden_flat[start:end] = (
                d_logits @ weight_f
            ).to(hidden.dtype)
        if grad_weight is not None:
            grad_weight = grad_weight + (
                d_logits.t() @ h_chunk.float()
            ).to(grad_weight.dtype)

    return grad_hidden, grad_weight, None, None, None, None, None


_chunked_lce_op.register_autograd(
    _chunked_lce_backward,
    setup_context=_chunked_lce_setup_context,
)


def chunked_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = 1024,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute tied LM-head CE without materializing all logits at once.

    ``hidden[..., i, :]`` predicts ``labels[..., i + 1]``. The final hidden
    position is dropped to match standard next-token CE semantics. Backward
    recomputes one logits chunk at a time, trading extra matmuls for much
    lower peak activation memory.
    """
    if hidden.size(1) < 2:
        return hidden.new_zeros((), dtype=torch.float32)
    autocast_enabled, autocast_dtype = _autocast_state(hidden.device.type)
    loss, _lse, _count = _chunked_lce_op(
        hidden,
        weight,
        labels,
        int(chunk_size),
        int(ignore_index),
        bool(autocast_enabled),
        autocast_dtype,
    )
    return loss


def standard_lm_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    # Branchless: tensor-value `if .any()` would force a graph break under
    # torch.compile. With reduction='sum' / clamped count we get 0 when
    # everything is masked, matching the old all-ignored fallback.
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    flat_labels = shift_labels.reshape(-1)
    loss_sum = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        flat_labels,
        ignore_index=ignore_index,
        reduction="sum",
    )
    count = (flat_labels != ignore_index).sum().clamp_min(1)
    return loss_sum / count
