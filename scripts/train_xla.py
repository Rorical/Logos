#!/usr/bin/env python
"""TPU/XLA causal LM pre-training driver.

This script mirrors ``scripts/train.py`` but replaces CUDA DDP/NCCL with
PyTorch/XLA's launcher, ``MpDeviceLoader`` and XLA collectives. The CLI is
intentionally shared with ``train.py`` so model, dataset, optimizer,
observability, EMA and checkpoint options stay aligned.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from datasets import DatasetDict, load_dataset
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

base = importlib.import_module("scripts.train")
baseline_mod = importlib.import_module("models.baseline")


def _import_xla():
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
    except ImportError as exc:
        raise ImportError(
            "`scripts/train_xla.py` requires PyTorch/XLA. Install a torch_xla "
            "build that matches your PyTorch version on the TPU VM, then run "
            "this script there."
        ) from exc
    return torch_xla, xm, pl


def _xla_device(torch_xla, xm) -> torch.device:
    if hasattr(torch_xla, "device"):
        return torch_xla.device()
    return xm.xla_device()


def _xla_world_size(xm) -> int:
    try:
        import torch_xla.runtime as xr

        return int(xr.world_size())
    except Exception:
        return int(xm.xrt_world_size())


def _xla_rank(xm) -> int:
    try:
        import torch_xla.runtime as xr

        return int(xr.global_ordinal())
    except Exception:
        return int(xm.get_ordinal())


def _xla_is_main(xm) -> bool:
    return _xla_rank(xm) == 0


def _xla_barrier(xm, tag: str = "logos_xla_barrier") -> None:
    if _xla_world_size(xm) > 1:
        xm.rendezvous(tag)


def _xla_all_reduce_sum(xm, t: torch.Tensor) -> torch.Tensor:
    if _xla_world_size(xm) <= 1:
        return t
    return xm.all_reduce(xm.REDUCE_SUM, t)


def _xla_all_reduce_mean(xm, t: torch.Tensor) -> torch.Tensor:
    world_size = _xla_world_size(xm)
    if world_size <= 1:
        return t
    return xm.all_reduce(xm.REDUCE_SUM, t, scale=1.0 / world_size)


def _patch_base_process_helpers(xm) -> None:
    """Make imported helpers from train.py rank-aware under XLA."""

    base.is_main_process = lambda: _xla_is_main(xm)
    base.barrier = lambda: _xla_barrier(xm)
    base.all_reduce_mean = lambda t: _xla_all_reduce_mean(xm, t)
    baseline_mod._maybe_all_reduce_load = lambda t: _xla_all_reduce_sum(xm, t)


def _xla_autocast(enabled: bool, dtype: torch.dtype):
    return torch.autocast("xla", dtype=dtype, enabled=enabled)


def _model_for_state_dict(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def _safe_mkdir_main(xm, path: Path) -> None:
    if _xla_is_main(xm):
        path.mkdir(parents=True, exist_ok=True)
    _xla_barrier(xm, f"mkdir:{path}")


def save_checkpoint_xla(
    xm,
    model: nn.Module,
    optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    save_dir: Path,
    is_best: bool = False,
    muon_hp: Optional[Any] = None,
    ema_model: Optional[AveragedModel] = None,
    keep_last_n: int = 0,
    stream_docs_consumed: Optional[int] = None,
):
    _safe_mkdir_main(xm, save_dir)
    inner = _model_for_state_dict(model)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": inner.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    # Within-shard train-stream position for resumable streaming (None on the
    # per-epoch path, which re-reads a finite dataset in full each epoch).
    if stream_docs_consumed is not None:
        checkpoint["stream_docs_consumed"] = int(stream_docs_consumed)
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if muon_hp is not None:
        checkpoint["muon_hp_state_dict"] = muon_hp.state_dict()
    if ema_model is not None:
        checkpoint["ema_state_dict"] = ema_model.state_dict()

    path = save_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    xm.save(checkpoint, str(path), master_only=True, global_master=True)
    if is_best:
        xm.save(
            checkpoint,
            str(save_dir / "checkpoint_best.pt"),
            master_only=True,
            global_master=True,
        )

    if _xla_is_main(xm):
        config_path = save_dir / "config.json"
        if not config_path.exists():
            with config_path.open("w") as f:
                json.dump(vars(inner.config), f, indent=2, default=str)
        base.prune_old_checkpoints(save_dir, keep_last_n)
    _xla_barrier(xm, f"checkpoint:{epoch}")
    return path


def save_final_xla(
    xm,
    payload: Dict[str, Any],
    save_dir: Path,
    filename: str = "checkpoint_final.pt",
) -> Path:
    _safe_mkdir_main(xm, save_dir)
    path = save_dir / filename
    xm.save(payload, str(path), master_only=True, global_master=True)
    _xla_barrier(xm, f"final:{filename}")
    return path


def _reduce_loss_triplet(
    xm,
    loss: torch.Tensor,
    lm_loss: Optional[torch.Tensor],
    aux_loss: Optional[torch.Tensor],
    indexer_loss: Optional[torch.Tensor] = None,
) -> list[float]:
    zero = torch.zeros((), device=loss.device, dtype=loss.dtype)
    parts = [
        loss.detach(),
        (lm_loss if lm_loss is not None else loss).detach(),
        (aux_loss.detach() if aux_loss is not None else zero),
        (indexer_loss.detach() if indexer_loss is not None else zero),
    ]
    pair = torch.stack(parts)
    pair = _xla_all_reduce_mean(xm, pair)
    return pair.cpu().tolist()


def _optimizer_step_xla(
    xm,
    model: nn.Module,
    optimizer,
    grad_clip: float,
) -> Optional[torch.Tensor]:
    grad_norm = None
    if hasattr(xm, "reduce_gradients"):
        xm.reduce_gradients(optimizer)
        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        xm.mark_step()
    else:
        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        xm.optimizer_step(optimizer)
    return grad_norm


@torch.no_grad()
def evaluate_xla(
    xm,
    pl,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
    mp_dtype: torch.dtype,
    max_batches: Optional[int] = None,
    batches_per_execution: int = 1,
) -> Dict[str, float]:
    was_training = model.training
    model.train(False)
    loss_sum = torch.zeros(1, device=device)
    lm_sum = torch.zeros(1, device=device)
    count = torch.zeros(1, device=device)
    device_loader = pl.MpDeviceLoader(
        dataloader, device, batches_per_execution=batches_per_execution,
    )
    for i, batch in enumerate(device_loader):
        if max_batches is not None and i >= max_batches:
            break
        ids = batch["input_ids"]
        lbls = batch["labels"]
        with _xla_autocast(use_amp, mp_dtype):
            out = model(input_ids=ids, attention_mask=None, labels=lbls, is_causal=True)
        loss_sum += out["loss"].detach()
        lm_t = out.get("lm_loss")
        lm_sum += (lm_t.detach() if lm_t is not None else out["loss"].detach())
        count += 1

    model.train(was_training)
    loss_sum = _xla_all_reduce_sum(xm, loss_sum)
    lm_sum = _xla_all_reduce_sum(xm, lm_sum)
    count = _xla_all_reduce_sum(xm, count)
    n = count.item()
    if n == 0:
        return {"loss": float("inf"), "ppl": float("inf")}
    avg = (loss_sum / count).item()
    avg_lm = (lm_sum / count).item()
    return {"loss": avg, "ppl": math.exp(min(avg_lm, 20))}


def run_epoch_xla(
    xm,
    pl,
    model: nn.Module,
    raw_model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    grad_clip: float,
    is_train: bool = True,
    epoch: int = 0,
    use_amp: bool = False,
    mp_dtype: torch.dtype = torch.float32,
    muon_hp: Optional[Any] = None,
    ema_model: Optional[AveragedModel] = None,
    desc: Optional[str] = None,
    batches_per_execution: int = 1,
) -> Dict[str, float]:
    model.train(is_train)
    sampler = getattr(dataloader, "sampler", None)
    if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
        sampler.set_epoch(epoch)

    main = _xla_is_main(xm)
    total_loss = 0.0
    total_lm_loss = 0.0
    total_aux_loss = 0.0
    num_examples = 0
    device_loader = pl.MpDeviceLoader(
        dataloader, device, batches_per_execution=batches_per_execution,
    )
    pbar = tqdm(
        device_loader,
        total=len(dataloader) if hasattr(dataloader, "__len__") else None,
        desc=desc or f"{'Train' if is_train else 'Valid'} Epoch {epoch}",
        leave=False,
        disable=not main,
    )

    for step_in_epoch, batch in enumerate(pbar):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            base.set_moe_training_step(
                raw_model,
                max(0, (epoch - 1) * len(dataloader) + step_in_epoch),
            )

        grad_ctx = contextlib.nullcontext() if is_train else torch.no_grad()
        with grad_ctx, _xla_autocast(use_amp, mp_dtype):
            outputs = model(
                input_ids=input_ids,
                attention_mask=None,
                labels=labels,
                is_causal=True,
            )
        loss = outputs["loss"]

        if is_train:
            loss.backward()
            _optimizer_step_xla(xm, model, optimizer, grad_clip)
            if scheduler is not None:
                scheduler.step()
            if muon_hp is not None:
                muon_hp.step()

            topk_indices = outputs.get("topk_indices")
            if topk_indices is not None:
                raw_model.update_router_biases(topk_indices)
            if ema_model is not None:
                ema_model.update_parameters(raw_model)

        batch_size = input_ids.size(0)
        loss_val, lm_loss_val, aux_loss_val, _ = _reduce_loss_triplet(
            xm, loss, outputs.get("lm_loss"), outputs.get("aux_loss"),
        )
        total_loss += loss_val * batch_size
        total_lm_loss += lm_loss_val * batch_size
        total_aux_loss += aux_loss_val * batch_size
        num_examples += batch_size
        if main:
            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "lm": f"{lm_loss_val:.4f}",
                "aux": f"{aux_loss_val:.2e}",
                "ppl": f"{math.exp(min(lm_loss_val, 20)):.2f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

    pbar.close()
    agg = torch.tensor(
        [total_loss, total_lm_loss, total_aux_loss, float(num_examples)],
        dtype=torch.float64,
        device=device,
    )
    agg = _xla_all_reduce_sum(xm, agg)
    total_loss = agg[0].item()
    total_lm_loss = agg[1].item()
    total_aux_loss = agg[2].item()
    num_examples = int(agg[3].item())
    if num_examples == 0:
        return {"loss": float("inf"), "ppl": float("inf")}
    avg_loss = total_loss / num_examples
    avg_lm_loss = total_lm_loss / num_examples
    avg_aux_loss = total_aux_loss / num_examples
    return {
        "loss": avg_loss,
        "lm_loss": avg_lm_loss,
        "aux_loss": avg_aux_loss,
        "ppl": math.exp(min(avg_lm_loss, 20)),
    }


def _moe_load_metrics_xla(
    xm,
    topk_indices: torch.Tensor,
    num_experts: int,
    capacity_factor: Optional[float] = None,
) -> Dict[str, Any]:
    counts = torch.bincount(topk_indices.reshape(-1), minlength=num_experts).float()
    counts = _xla_all_reduce_sum(xm, counts)
    total = counts.sum().clamp(min=1.0)
    frac = counts / total
    target = 1.0 / num_experts
    eps = 1e-12
    kl = (frac * (torch.log(frac + eps) - math.log(target))).sum()
    entropy = -(frac * torch.log(frac + eps)).sum() / math.log(max(num_experts, 2))
    overflow_frac = torch.zeros((), device=frac.device, dtype=frac.dtype)
    if capacity_factor is not None and capacity_factor > 0:
        capacity = max(1, int(topk_indices.numel() * capacity_factor / num_experts))
        overflow = (counts - capacity).clamp_min(0).sum()
        overflow_frac = overflow / total
    scalars = torch.stack([
        frac.max(),
        frac.min(),
        frac.std(),
        (counts == 0).sum().to(frac.dtype),
        kl,
        entropy,
        overflow_frac,
    ]).detach().cpu().tolist()
    return {
        "frac": frac.detach().cpu(),
        "load_max": scalars[0],
        "load_min": scalars[1],
        "load_std": scalars[2],
        "dead_experts": int(scalars[3]),
        "kl_uniform": scalars[4],
        "load_entropy_norm": scalars[5],
        "capacity_overflow_frac": scalars[6],
    }


def log_moe_load_xla(xm, args: argparse.Namespace, topk_indices_list, step: int) -> None:
    if not args.wandb or topk_indices_list is None:
        return
    names = base._moe_layer_names(args, len(topk_indices_list))
    metrics: Dict[str, Any] = {}
    for name, topk in zip(names, topk_indices_list):
        if topk is None:
            continue
        m = _moe_load_metrics_xla(
            xm, topk, args.num_sparse_experts, args.capacity_factor,
        )
        if not _xla_is_main(xm):
            continue
        prefix = f"moe/{name}"
        metrics[f"{prefix}/load_max"] = m["load_max"]
        metrics[f"{prefix}/load_min"] = m["load_min"]
        metrics[f"{prefix}/load_std"] = m["load_std"]
        metrics[f"{prefix}/dead_experts"] = m["dead_experts"]
        metrics[f"{prefix}/kl_uniform"] = m["kl_uniform"]
        metrics[f"{prefix}/load_entropy_norm"] = m["load_entropy_norm"]
        metrics[f"{prefix}/capacity_overflow_frac"] = m["capacity_overflow_frac"]
        import wandb

        metrics[f"{prefix}/load_hist"] = wandb.Histogram(
            sequence=m["frac"].tolist(), num_bins=args.num_sparse_experts,
        )
    if metrics:
        base.wandb_log(metrics, step=step)


def run_step_training_xla(
    xm,
    pl,
    args: argparse.Namespace,
    model: nn.Module,
    raw_model: nn.Module,
    optimizer,
    scheduler,
    muon_hp,
    ema_model: Optional[AveragedModel],
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    use_amp: bool,
    mp_dtype: torch.dtype,
    save_dir: Path,
    sample_text_fn,
    total_steps: int,
    start_step: int = 0,
    phase1_train_loader: Optional[DataLoader] = None,
) -> Dict[str, Any]:
    grad_clip = args.grad_clip
    log_every = max(1, args.log_every)
    main = _xla_is_main(xm)
    world_size = _xla_world_size(xm)

    if main:
        print("\n" + "=" * 60)
        resume_note = f" | resuming at step {start_step}" if start_step else ""
        print(
            f"XLA step training: {total_steps} steps | "
            f"world_size={world_size}" + resume_note
        )
        phase1_steps = base.token_superposition_phase1_steps(args, total_steps)
        if phase1_steps:
            print(
                "Token Superposition Training: "
                f"phase 1 steps 0-{phase1_steps - 1} "
                f"(bag_size={args.token_superposition_bag_size}), "
                f"phase 2 steps {phase1_steps}-{total_steps - 1} "
                "(standard NTP)"
            )
        print("=" * 60)

    history: list = []
    best_val_loss = float("inf")
    running_loss = 0.0
    running_lm_loss = 0.0
    running_aux_loss = 0.0
    running_count = 0
    step = start_step
    t0 = time.time()
    pbar = tqdm(total=total_steps, initial=start_step, desc="train", disable=not main)

    model.train(True)
    phase1_steps = base.token_superposition_phase1_steps(args, total_steps)
    if phase1_steps > 0 and phase1_train_loader is None:
        raise ValueError("phase1_train_loader is required when TST is enabled")
    train_loaders: Dict[int, DataLoader] = {1: train_loader}
    if phase1_steps > 0 and phase1_train_loader is not None:
        train_loaders[int(args.token_superposition_bag_size)] = phase1_train_loader
    device_loaders: Dict[int, Any] = {}
    train_iters: Dict[int, Any] = {}
    # GLOBAL post-val corpus docs consumed (the unit that feeds back as
    # .skip(stream_docs_consumed)). source_idx is a within-shard local index,
    # so a per-replica count of K maps to ~K*world_size global docs. Seed from
    # the resume position so the first save reflects total progress.
    consumed_resume = int(getattr(args, "stream_docs_consumed", 0) or 0)
    docs_consumed = consumed_resume
    # Running max of within-shard source_idx, kept ON DEVICE so the per-step
    # update is a fused ``torch.maximum`` that never forces a device->host sync
    # before the forward graph builds. Materialized to host only when a
    # checkpoint is about to be written (see ``_materialize_docs_consumed``).
    src_idx_max_dev = torch.full((), -1, dtype=torch.long, device=device)

    def _materialize_docs_consumed() -> int:
        # Host-sync the on-device source_idx max into the global doc-count unit
        # that feeds back as .skip(stream_docs_consumed). source_idx is a
        # within-shard local index, so a per-replica max of M maps to
        # ~(M+1)*world_size global docs. Call ONLY at checkpoint time.
        local_max = int(src_idx_max_dev.item())
        if local_max < 0:
            return docs_consumed
        return max(docs_consumed, consumed_resume + (local_max + 1) * world_size)
    # One-shot resume marker so a run started from a checkpoint records where it
    # picked up (step + corpus position) instead of silently continuing the
    # x-axis. Logged once, before the first optimizer step.
    if main and start_step > 0:
        base.wandb_log({
            "train/resumed_from_step": start_step,
            "train/resumed_docs_consumed": docs_consumed,
        }, step=start_step)
    last_step_time_ms = float("nan")
    last_train_avg_lm = float("nan")  # smoothed train LM loss for the val gap
    while step < total_steps:
        step_t0 = time.time()
        bag_size = base.token_superposition_bag_size_for_step(args, step, total_steps)
        device_loader = device_loaders.get(bag_size)
        if device_loader is None:
            device_loader = pl.MpDeviceLoader(
                train_loaders[bag_size],
                device,
                batches_per_execution=max(1, int(args.xla_batches_per_execution)),
            )
            device_loaders[bag_size] = device_loader
        train_iter = train_iters.get(bag_size)
        if train_iter is None:
            train_iter = iter(device_loader)
            train_iters[bag_size] = train_iter
        try:
            batch = next(train_iter)
        except StopIteration:
            device_loader = pl.MpDeviceLoader(
                train_loaders[bag_size],
                device,
                batches_per_execution=max(1, int(args.xla_batches_per_execution)),
            )
            device_loaders[bag_size] = device_loader
            train_iter = iter(device_loader)
            train_iters[bag_size] = train_iter
            batch = next(train_iter)

        # Pop source_idx OUT of the batch before the model sees it (the
        # MpDeviceLoader already moved it to the XLA device, so popping here
        # also keeps it out of the graph). Fold its max into the on-device
        # running max with ``torch.maximum`` — this stays on the XLA device, so
        # it does NOT force a device->host sync at the top of the step (the old
        # ``int(batch_source_idx.max())`` here serialised every step before the
        # forward graph could build). The host read is deferred to checkpoint
        # time via ``_materialize_docs_consumed``. Replicas advance in lockstep
        # so rank-0's count is a good resume offset for all; a few boundary docs
        # may re-appear once after a restart under multi-worker prefetch —
        # acceptable; systematic stream-head replay is what this kills.
        batch_source_idx = batch.pop("source_idx", None)
        if batch_source_idx is not None and main:
            src_idx_max_dev = torch.maximum(
                src_idx_max_dev, batch_source_idx.max(),
            )

        ids = batch["input_ids"]
        lbls = batch["labels"]
        optimizer.zero_grad(set_to_none=True)
        base.set_moe_training_step(raw_model, step)
        with _xla_autocast(use_amp, mp_dtype):
            outputs = model(
                input_ids=ids,
                attention_mask=None,
                labels=lbls,
                is_causal=True,
                token_superposition_bag_size=bag_size,
            )
        loss = outputs["loss"]
        loss.backward()
        grad_norm = _optimizer_step_xla(xm, model, optimizer, grad_clip)
        if scheduler is not None:
            scheduler.step()
        if muon_hp is not None:
            muon_hp.step()

        topk_indices = outputs.get("topk_indices")
        if topk_indices is not None:
            raw_model.update_router_biases(topk_indices)
        if ema_model is not None:
            ema_model.update_parameters(raw_model)

        if (
            args.moe_log_every > 0
            and (step + 1) % args.moe_log_every == 0
            and topk_indices is not None
        ):
            log_moe_load_xla(xm, args, topk_indices, step + 1)

        step += 1
        loss_val, lm_loss_val, aux_loss_val, indexer_loss_val = _reduce_loss_triplet(
            xm, loss, outputs.get("lm_loss"), outputs.get("aux_loss"),
            outputs.get("indexer_loss"),
        )
        grad_norm_val = grad_norm.detach().cpu().item() if grad_norm is not None else None
        # The loss reduction above forces a device->host sync, so it is the
        # natural step barrier to measure compute time against.
        last_step_time_ms = (time.time() - step_t0) * 1000.0
        running_loss += loss_val
        running_lm_loss += lm_loss_val
        running_aux_loss += aux_loss_val
        running_count += 1
        if main:
            pbar.update(1)

        if main:
            tokens_seen = base.token_superposition_tokens_seen(
                args, step, total_steps, world_size,
            )
            metrics = {
                "train/loss": loss_val,
                "train/lm_loss": lm_loss_val,
                "train/aux_loss": aux_loss_val,
                "train/indexer_loss": indexer_loss_val,
                "train/ppl": math.exp(min(lm_loss_val, 20)),
                "train/tokens_seen": tokens_seen,
                "train/docs_consumed": docs_consumed,
                "train/step_time_ms": last_step_time_ms,
                "train/tst_bag_size": bag_size,
                "train/tst_phase": 1 if bag_size > 1 else 2,
            }
            for i, pg in enumerate(optimizer.param_groups):
                tag = pg.get("name") or f"g{i}"
                metrics[f"train/lr/{tag}"] = pg.get("lr", 0.0)
                if "momentum" in pg:
                    metrics[f"train/momentum/{tag}"] = pg["momentum"]
                if "weight_decay" in pg:
                    metrics[f"train/weight_decay/{tag}"] = pg["weight_decay"]
            if grad_norm_val is not None:
                metrics["train/grad_norm"] = grad_norm_val
            base.wandb_log(metrics, step=step)

        if step % log_every == 0 and main:
            avg = running_loss / running_count
            avg_lm = running_lm_loss / running_count
            avg_aux = running_aux_loss / running_count
            last_train_avg_lm = avg_lm  # carried into the val gap at eval time
            ppl = math.exp(min(avg_lm, 20))
            elapsed = time.time() - t0
            tokens_this_run = (
                base.token_superposition_tokens_seen(args, step, total_steps, world_size)
                - base.token_superposition_tokens_seen(
                    args, start_step, total_steps, world_size,
                )
            )
            tps = tokens_this_run / max(elapsed, 1)
            postfix = {
                "loss": f"{avg:.3f}",
                "lm": f"{avg_lm:.3f}",
                "aux": f"{avg_aux:.1e}",
                "ppl": f"{ppl:.1f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                "tok/s": f"{tps:.0f}",
            }
            if phase1_steps:
                postfix["phase"] = "tst" if bag_size > 1 else "ntp"
                postfix["bag"] = str(bag_size)
            pbar.set_postfix(postfix)
            base.wandb_log({
                "train/avg_loss": avg,
                "train/avg_lm_loss": avg_lm,
                "train/avg_aux_loss": avg_aux,
                "train/avg_ppl": ppl,
                "train/tok_per_sec": tps,
            }, step=step)
            running_loss = 0.0
            running_lm_loss = 0.0
            running_aux_loss = 0.0
            running_count = 0

        eval_due = (
            args.eval_every > 0 and step % args.eval_every == 0
            and val_loader is not None
        )
        save_due = (
            not args.no_save and args.save_every > 0
            and step % args.save_every == 0
        )

        # Materialize the on-device source_idx max into ``docs_consumed`` only
        # when a checkpoint is about to be written (eval saves too, unless
        # --no-save). This is the ONLY per-step device->host sync for the doc
        # count, and it runs at checkpoint cadence rather than every step. Only
        # the master tracks src_idx_max_dev and writes checkpoints, so gate the
        # sync on ``main`` to avoid serialising the other replicas needlessly.
        if (main and args.streaming
                and (save_due or (eval_due and not args.no_save))):
            docs_consumed = _materialize_docs_consumed()

        if eval_due:
            val_metrics = evaluate_xla(
                xm, pl, raw_model, val_loader, device, use_amp, mp_dtype,
                batches_per_execution=max(1, int(args.xla_batches_per_execution)),
            )
            ema_val_metrics = None
            if ema_model is not None:
                ema_val_metrics = evaluate_xla(
                    xm, pl, ema_model, val_loader, device, use_amp, mp_dtype,
                    batches_per_execution=max(1, int(args.xla_batches_per_execution)),
                )
            if main:
                log = (f"\nstep {step:>6} | val_loss {val_metrics['loss']:.4f} "
                       f"(ppl {val_metrics['ppl']:.2f})")
                if ema_val_metrics is not None and ema_val_metrics["loss"] != float("inf"):
                    log += (f" | ema_val {ema_val_metrics['loss']:.4f} "
                            f"(ppl {ema_val_metrics['ppl']:.2f})")
                print(log)
                history.append({
                    "step": step,
                    "val": val_metrics,
                    "ema_val": ema_val_metrics,
                })
                eval_log = {
                    "val/loss": val_metrics["loss"],
                    "val/ppl": val_metrics["ppl"],
                }
                # Train-val gap on the smoothed LM loss (NaN until first log).
                if (math.isfinite(last_train_avg_lm)
                        and val_metrics["loss"] != float("inf")):
                    eval_log["val/train_gap"] = (
                        val_metrics["loss"] - last_train_avg_lm
                    )
                if ema_val_metrics is not None:
                    eval_log["ema_val/loss"] = ema_val_metrics["loss"]
                    eval_log["ema_val/ppl"] = ema_val_metrics["ppl"]
                base.wandb_log(eval_log, step=step)
            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
            if not args.no_save:
                save_checkpoint_xla(
                    xm, raw_model, optimizer, scheduler,
                    epoch=step, metrics=val_metrics,
                    save_dir=save_dir, is_best=is_best,
                    muon_hp=muon_hp, ema_model=ema_model,
                    keep_last_n=args.keep_last_n,
                    stream_docs_consumed=docs_consumed if args.streaming else None,
                )
        elif save_due:
            save_checkpoint_xla(
                xm, raw_model, optimizer, scheduler,
                epoch=step, metrics={"loss": loss_val},
                save_dir=save_dir, is_best=False,
                muon_hp=muon_hp, ema_model=ema_model,
                keep_last_n=args.keep_last_n,
                stream_docs_consumed=docs_consumed if args.streaming else None,
            )

        if (
            args.sample_every > 0
            and step % args.sample_every == 0
            and sample_text_fn is not None
            and main
        ):
            try:
                generated = sample_text_fn(
                    args.sample_prompt,
                    args.sample_max_tokens,
                    args.sample_temperature,
                )
                print(f"  Sample -> {generated}")
                base.wandb_log({"sample/text": generated}, step=step)
            except UnicodeEncodeError as exc:
                print(f"  Sample -> (unicode encode failed: {exc})")
            model.train(True)

    if main:
        pbar.close()
        print(f"\n=== Training complete in {(time.time() - t0) / 60:.1f} min ===")
    # Final host-sync of the doc count for the final checkpoint (the loop only
    # materialized at save cadence, so the tail steps since the last save are
    # not yet reflected).
    if args.streaming and main:
        docs_consumed = _materialize_docs_consumed()
    return {
        "best_val_loss": best_val_loss,
        "history": history,
        "stream_docs_consumed": docs_consumed,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = base.build_arg_parser()
    parser.description = "Train a decoder-only transformer on TPU/XLA"
    parser.add_argument(
        "--xla-num-devices",
        type=int,
        default=None,
        help="Optional nprocs for legacy xmp.spawn fallback. Leave unset for "
             "torch_xla.launch to use the runtime default.",
    )
    parser.add_argument(
        "--xla-batches-per-execution",
        type=int,
        default=1,
        help="MpDeviceLoader batches_per_execution. Larger values can reduce "
             "host overhead but delay tqdm/log updates.",
    )
    parser.add_argument(
        "--xla-single-process",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run one XLA process instead of spawning across every visible TPU "
             "device. Use this on single-device Colab TPU runtimes such as "
             "TPUv6e-1, where multiprocess launch may expect a larger TPU "
             "slice than Colab exposes.",
    )
    parser.add_argument(
        "--xla-no-broadcast-master-param",
        action="store_true",
        help="Skip broadcasting rank-0 initialized weights to other replicas. "
             "Usually only useful for debugging.",
    )
    return parser


def _load_finite_dataset_xla(args, tokenizer, xm, rank: int, world_size: int):
    main_proc = _xla_is_main(xm)

    def _load_dataset_raw():
        if main_proc:
            print(f"Loading dataset: "
                  f"{base._describe_dataset_source(args.dataset, args.dataset_config)}")
        ds = load_dataset(**base._dataset_load_kwargs(
            args.dataset, args.dataset_config, streaming=False,
        ))
        if main_proc:
            print(f"  splits: {list(ds.keys())}")
        if "validation" not in ds and "valid" not in ds:
            if main_proc:
                print("  No validation split found - creating 10% hold-out from train")
            split = ds["train"].train_test_split(test_size=0.1, seed=args.seed)
            ds = DatasetDict({
                "train": split["train"],
                "validation": split["test"],
            })
        if "validation" not in ds and "valid" in ds:
            ds["validation"] = ds["valid"]
        if args.max_train_examples is not None:
            n = min(args.max_train_examples, len(ds["train"]))
            ds["train"] = ds["train"].select(range(n))
            if main_proc:
                print(f"  Subset train to {n} examples")
        if args.max_val_examples is not None:
            n = min(args.max_val_examples, len(ds["validation"]))
            ds["validation"] = ds["validation"].select(range(n))
            if main_proc:
                print(f"  Subset validation to {n} examples")
        return ds

    if main_proc:
        raw_dataset = _load_dataset_raw()
    _xla_barrier(xm, "dataset-preprocess-main")
    if not main_proc:
        raw_dataset = _load_dataset_raw()
    _xla_barrier(xm, "dataset-preprocess-all")

    if main_proc:
        print(f"Preprocessing with block_size={args.max_length} ...")
    dataset = base.preprocess_dataset(
        raw_dataset, tokenizer, args.max_length, text_column=args.text_column,
    )

    phase1_dataset = None
    if base.token_superposition_enabled(args):
        phase1_block_size = base.token_superposition_phase1_block_size(args)
        if main_proc:
            print(
                "Preprocessing TST phase-1 train split with "
                f"block_size={phase1_block_size} ..."
            )
        phase1_dataset = base.preprocess_dataset(
            DatasetDict({"train": raw_dataset["train"]}),
            tokenizer,
            phase1_block_size,
            text_column=args.text_column,
        )

    if main_proc:
        print(f"  train examples:      {len(dataset['train'])}")
        print(f"  validation examples: {len(dataset['validation'])}")
        if phase1_dataset is not None:
            print(f"  TST phase-1 examples: {len(phase1_dataset['train'])}")
    if phase1_dataset is not None and len(phase1_dataset["train"]) == 0:
        raise ValueError(
            "TST phase-1 preprocessing produced zero train examples; "
            "reduce --max-length or --tst-bag-size, or use more data."
        )

    train_loader = base.create_dataloader(
        dataset["train"], args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
    )
    val_loader = base.create_dataloader(
        dataset["validation"], args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
        drop_last=(world_size > 1),
    )
    phase1_train_loader = None
    if phase1_dataset is not None:
        phase1_train_loader = base.create_dataloader(
            phase1_dataset["train"], args.batch_size, shuffle=True,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            seed=args.seed,
        )
    return train_loader, val_loader, phase1_train_loader


def _build_streaming_loaders_xla(args, tokenizer, xm, rank: int, world_size: int):
    if not args.dataset:
        raise ValueError("--streaming requires --dataset (HF name or local path)")
    if _xla_is_main(xm):
        print(f"Streaming dataset: "
              f"{base._describe_dataset_source(args.dataset, args.dataset_config)}")

    load_kwargs = base._dataset_load_kwargs(
        args.dataset, args.dataset_config,
        streaming=True, split="train",
    )
    val_stream = load_dataset(**load_kwargs)
    val_docs_full = list(val_stream.take(args.val_docs))
    if _xla_is_main(xm):
        print(f"  cached {len(val_docs_full)} docs for val")

    # Pipeline: .skip(val_docs) -> .skip(consumed_global) -> .shuffle(seed)
    # -> .shard(world_size, rank). Skip BEFORE shuffle (HF's .skip after a
    # shuffle drops only one shard-block on a single-shard stream and would
    # replay the prefix — see base.shuffle_train_stream). consumed_global is a
    # GLOBAL post-val corpus-doc count, so every replica skips the same prefix
    # and then shards the shuffled tail. Each replica's PackedStream enumerates
    # its own shard, so source_idx reports a within-shard local index here
    # (base_offset=0); the loop scales it back to a global count.
    consumed_global = int(getattr(args, "stream_docs_consumed", 0) or 0)
    train_stream = load_dataset(**load_kwargs).skip(args.val_docs)
    if consumed_global > 0:
        train_stream = train_stream.skip(consumed_global)
    train_stream = base.shuffle_train_stream(args, train_stream)
    if world_size > 1:
        train_stream = train_stream.shard(num_shards=world_size, index=rank)

    train_ds = base.PackedStream(
        train_stream, tokenizer, args.max_length, text_key=args.text_column,
        emit_source_idx=True,
    )
    val_ds = base.PackedStream(
        val_docs_full, tokenizer, args.max_length, text_key=args.text_column,
    )

    def collate(batch):
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

    nw = max(0, int(args.num_workers))
    common_kw: Dict[str, Any] = dict(
        batch_size=args.batch_size,
        collate_fn=collate,
        num_workers=nw,
        pin_memory=False,
    )
    if nw > 0:
        common_kw["persistent_workers"] = True
        common_kw["prefetch_factor"] = max(1, int(args.prefetch_factor))
    train_loader = DataLoader(train_ds, **common_kw)
    val_loader = DataLoader(val_ds, drop_last=False, **common_kw)
    phase1_train_loader = None
    if base.token_superposition_enabled(args):
        phase1_block_size = base.token_superposition_phase1_block_size(args)
        if _xla_is_main(xm):
            print(
                "Building TST phase-1 streaming train loader with "
                f"block_size={phase1_block_size}"
            )
        train_stream_phase1 = load_dataset(**load_kwargs).skip(args.val_docs)
        if consumed_global > 0:
            train_stream_phase1 = train_stream_phase1.skip(consumed_global)
        train_stream_phase1 = base.shuffle_train_stream(args, train_stream_phase1)
        if world_size > 1:
            train_stream_phase1 = train_stream_phase1.shard(
                num_shards=world_size, index=rank,
            )
        phase1_ds = base.PackedStream(
            train_stream_phase1, tokenizer, phase1_block_size,
            text_key=args.text_column, emit_source_idx=True,
        )
        phase1_train_loader = DataLoader(phase1_ds, **common_kw)
    return train_loader, val_loader, phase1_train_loader


def _broadcast_master_params(xm, model: nn.Module, disabled: bool, main_proc: bool) -> None:
    if disabled:
        return
    if hasattr(xm, "broadcast_master_param"):
        xm.broadcast_master_param(model)
        return
    try:
        from torch_xla.experimental import pjrt

        pjrt.broadcast_master_param(model)
    except Exception as exc:
        if main_proc:
            print(f"  Warning: could not broadcast master params ({exc}); "
                  "replicas rely on identical seeding.")


def _xla_worker(index: int, args: argparse.Namespace) -> None:
    torch_xla, xm, pl = _import_xla()
    _patch_base_process_helpers(xm)

    device = _xla_device(torch_xla, xm)
    rank = _xla_rank(xm)
    world_size = _xla_world_size(xm)
    main_proc = _xla_is_main(xm)

    torch.manual_seed(args.seed + rank)
    if main_proc:
        print(f"Using device: {device} | model: {args.model} | XLA world_size={world_size}")
        if args.device not in (None, "xla"):
            print("  Note: --device is ignored by train_xla.py; using torch_xla.device().")
        if args.compile:
            print("  Note: --compile is ignored; XLA performs lazy compilation.")

    mp_dtype = torch.bfloat16 if args.bf16 else torch.float32
    use_amp = args.bf16
    if use_amp and main_proc:
        print("  Mixed precision: torch.autocast('xla', dtype=torch.bfloat16)")

    if args.save_dir is None:
        args.save_dir = f"checkpoints/{args.model}_xla"
    save_dir = Path(args.save_dir)
    _safe_mkdir_main(xm, save_dir)

    wandb_run = base.init_wandb(args) if main_proc else None
    if wandb_run is not None and main_proc:
        print(f"  W&B run: {wandb_run.url}")

    if main_proc:
        print(f"Loading tiktoken encoding: {args.tiktoken_encoding}")
    tokenizer = base.TiktokenTokenizer(args.tiktoken_encoding)
    if main_proc:
        print(f"  vocab_size: {tokenizer.vocab_size}")

    if args.streaming:
        if args.total_steps is not None and args.total_tokens is not None:
            raise ValueError("--total-steps and --total-tokens are mutually exclusive")
        if args.total_tokens is not None:
            tokens_per_step = world_size * args.batch_size * args.max_length
            args.total_steps = base.derive_steps_from_token_budget(
                args, args.total_tokens, world_size,
            )
            if main_proc:
                actual_tokens = base.token_superposition_tokens_seen(
                    args, args.total_steps, args.total_steps, world_size,
                )
                phase_note = ""
                if base.token_superposition_enabled(args):
                    phase_note = (
                        f", phase1 bag_size={args.token_superposition_bag_size}, "
                        f"ratio={args.token_superposition_ratio:g}"
                    )
                print(
                    f"  Token budget: --total-tokens={args.total_tokens:,} "
                    f"-> total_steps={args.total_steps:,} "
                    f"({tokens_per_step:,} standard tokens/step{phase_note}, "
                    f"actual={actual_tokens:,})"
                )
        if args.total_steps is None or args.total_steps <= 0:
            raise ValueError(
                "--streaming requires --total-steps N (N > 0) or --total-tokens"
            )
        # Resolve the resume stream position BEFORE building the loaders so the
        # train pipeline becomes .skip(val_docs) -> .skip(consumed) -> .shuffle
        # -> .shard. The full model/optimizer restore happens later.
        args.stream_docs_consumed = 0
        _resume_peek_path = base.resolve_resume_path(args.resume, save_dir)
        if _resume_peek_path is not None:
            args.stream_docs_consumed = base.read_stream_docs_consumed(
                _resume_peek_path, torch.device("cpu"),
            )
            if main_proc and args.stream_docs_consumed > 0:
                print(
                    "  Resuming stream at global doc "
                    f"{args.stream_docs_consumed:,} (skip val_docs -> "
                    f"skip {args.stream_docs_consumed:,} -> shuffle -> shard)"
                )
        train_loader, val_loader, phase1_train_loader = _build_streaming_loaders_xla(
            args, tokenizer, xm, rank=rank, world_size=world_size,
        )
    else:
        train_loader, val_loader, phase1_train_loader = _load_finite_dataset_xla(
            args, tokenizer, xm, rank=rank, world_size=world_size,
        )

    config, model = base.build_model(args, vocab_size=tokenizer.vocab_size)
    model = model.to(device)
    _broadcast_master_params(xm, model, args.xla_no_broadcast_master_param, main_proc)
    if main_proc:
        print(f"\nModel: {type(model).__name__}")
        print(f"  parameters: {base.count_parameters(model):,}")
        print(f"  use_moe:    {config.use_moe}")
        if base.token_superposition_enabled(args):
            print(
                "  TST:        "
                f"bag_size={args.token_superposition_bag_size}, "
                f"ratio={args.token_superposition_ratio:g}, "
                f"phase1_block={base.token_superposition_phase1_block_size(args)}"
            )
        if config.use_moe:
            print(f"  shared experts: {config.num_shared_experts}")
            print(f"  sparse experts: {config.num_sparse_experts}")
            print(f"  top_k:          {config.top_k}")

    fused_adamw = False
    total_steps = (
        args.total_steps if args.streaming
        else len(train_loader) * args.epochs
    )
    if base.token_superposition_enabled(args) and not args.streaming:
        args.total_steps = total_steps
    muon_params, embed_params, no_decay_params, default_params, router_params = (
        base.split_param_groups(model)
    )
    optimizer, scheduler, muon_hp = base.build_optimizer_and_scheduler(
        args, total_steps, fused_adamw,
        muon_params, embed_params, default_params, router_params,
        no_decay_params=no_decay_params,
    )
    if main_proc:
        n_muon = sum(p.numel() for p in muon_params)
        n_embed = sum(p.numel() for p in embed_params)
        n_no_decay = sum(p.numel() for p in no_decay_params)
        n_default = sum(p.numel() for p in default_params)
        n_router = sum(p.numel() for p in router_params)
        if args.muon and n_muon > 0:
            print("  Optimizer: Muon + AdamW")
            print(f"    Muon tensors: {len(muon_params)}, params: {n_muon:,}")
            print(f"    AdamW embed tensors: {len(embed_params)}, params: {n_embed:,}")
            print(f"    AdamW default tensors: {len(default_params)}, params: {n_default:,}")
            if no_decay_params:
                print(f"    AdamW no_decay tensors: {len(no_decay_params)}, "
                      f"params: {n_no_decay:,} (wd=0)")
            if router_params:
                print(f"    AdamW router tensors: {len(router_params)}, params: {n_router:,}")
        else:
            print(f"  Optimizer: AdamW only "
                  f"({n_muon + n_embed + n_no_decay + n_default + n_router:,} params)")
        print(f"  LR schedule: {args.scheduler} "
              f"(warmup {args.warmup_steps} / {total_steps} total)")

    raw_model = model
    ema_model: Optional[AveragedModel] = None
    if args.ema_decay > 0:
        ema_model = AveragedModel(
            raw_model,
            multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay),
            use_buffers=True,
        )
        ema_model.to(device)
        if main_proc:
            print(f"  EMA enabled (decay={args.ema_decay}, use_buffers=True)")

    start_step = 0
    resume_arg = (args.resume or "none").strip()
    if resume_arg.lower() != "none":
        if resume_arg.lower() == "auto":
            resume_path = base.find_latest_checkpoint(save_dir)
        else:
            resume_path = Path(resume_arg)
            if not resume_path.exists():
                raise FileNotFoundError(
                    f"--resume path {resume_path} does not exist"
                )
        if resume_path is not None:
            if main_proc:
                print(f"  Resuming from {resume_path}")
            # Load to CPU first; load_state_dict copies tensors into XLA params.
            start_step = base.load_resume_checkpoint(
                resume_path, raw_model, optimizer, scheduler,
                muon_hp, ema_model, torch.device("cpu"),
            )
            if main_proc:
                print(f"    resumed at step/epoch {start_step}")
        elif main_proc and resume_arg.lower() == "auto":
            print("  --resume auto: no prior checkpoint under "
                  f"{save_dir}, starting fresh")
    _xla_barrier(xm, "resume")

    def sample_text(prompt: str, max_new_tokens: int, temperature: float) -> str:
        raw_model.train(False)
        prompt_ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long, device=device
        )
        with torch.no_grad(), _xla_autocast(use_amp, mp_dtype):
            generated = raw_model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        xm.mark_step()
        return tokenizer.decode(generated[0].cpu())

    if args.streaming or base.token_superposition_enabled(args):
        if base.token_superposition_enabled(args) and not args.streaming:
            steps_per_epoch = max(1, len(train_loader))
            if main_proc:
                print(
                    "  TST uses step-based training; converting finite-mode "
                    f"eval/save/sample intervals by {steps_per_epoch} "
                    "steps per epoch."
                )
            if args.eval_every > 0:
                args.eval_every *= steps_per_epoch
            if args.save_every > 0:
                args.save_every *= steps_per_epoch
            if args.sample_every > 0:
                args.sample_every *= steps_per_epoch
        result = run_step_training_xla(
            xm, pl, args, model, raw_model, optimizer, scheduler, muon_hp,
            ema_model, train_loader, val_loader, device, use_amp, mp_dtype,
            save_dir, sample_text, total_steps, start_step=start_step,
            phase1_train_loader=phase1_train_loader,
        )
        history = result["history"]
        best_val_loss = result["best_val_loss"]
        final_stream_docs_consumed = result.get("stream_docs_consumed")
        val_metrics = (history[-1].get("val") if history
                       else {"loss": float("inf"), "ppl": float("inf")})
        if main_proc:
            history_payload: Dict[str, Any] = {
                "model": args.model,
                "total_steps": args.total_steps,
                "final_metrics": val_metrics,
                "best_val_loss": best_val_loss,
                "history": history,
            }
            if base.token_superposition_enabled(args):
                history_payload["token_superposition"] = {
                    "bag_size": args.token_superposition_bag_size,
                    "ratio": args.token_superposition_ratio,
                    "phase1_steps": base.token_superposition_phase1_steps(args, total_steps),
                    "phase1_block_size": base.token_superposition_phase1_block_size(args),
                }
            if wandb_run is not None:
                history_payload["wandb"] = {
                    "run_id": wandb_run.id,
                    "run_url": wandb_run.url,
                    "project": wandb_run.project,
                    "entity": wandb_run.entity,
                }
            history_path = save_dir / "history.json"
            with history_path.open("w") as f:
                json.dump(history_payload, f, indent=2, default=str)
            print(f"\nHistory written to {history_path}")
        _xla_barrier(xm, "history-streaming")

        if not args.no_save:
            final_path = save_final_xla(
                xm,
                {
                    "step": total_steps,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "muon_hp_state_dict": muon_hp.state_dict() if muon_hp else None,
                    "ema_state_dict": ema_model.state_dict() if ema_model else None,
                    "metrics": val_metrics,
                    "history": history,
                    "stream_docs_consumed": final_stream_docs_consumed,
                },
                save_dir,
            )
            if main_proc:
                print(f"Final checkpoint saved to {final_path}")
        elif main_proc:
            print("(--no-save active: no model weights written)")
            print("Training complete!")
        if wandb_run is not None:
            wandb_run.finish()
        _xla_barrier(xm, "done-streaming")
        return

    if main_proc:
        print("\n" + "=" * 60)
        print(f"Starting XLA training | world_size={world_size}")
        print("=" * 60)

    best_val_loss = float("inf")
    history: list[Dict[str, Any]] = []
    val_metrics = None
    train_metrics = None
    batches_per_execution = max(1, int(args.xla_batches_per_execution))

    if start_step >= args.epochs and main_proc:
        print(f"  --resume target ({start_step}) is at or past --epochs "
              f"({args.epochs}); nothing to train.")

    for epoch in range(start_step + 1, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = run_epoch_xla(
            xm, pl, model, raw_model, train_loader, optimizer, scheduler, device,
            grad_clip=args.grad_clip, is_train=True, epoch=epoch,
            use_amp=use_amp, mp_dtype=mp_dtype, muon_hp=muon_hp,
            ema_model=ema_model, batches_per_execution=batches_per_execution,
        )

        val_metrics = None
        ema_val_metrics = None
        val_loader_nonempty = len(val_loader.dataset) > 0
        if epoch % args.eval_every == 0 and val_loader_nonempty:
            val_metrics = run_epoch_xla(
                xm, pl, model, raw_model, val_loader, optimizer, None, device,
                grad_clip=0.0, is_train=False, epoch=epoch,
                use_amp=use_amp, mp_dtype=mp_dtype,
                batches_per_execution=batches_per_execution,
            )
            if ema_model is not None:
                ema_val_metrics = run_epoch_xla(
                    xm, pl, ema_model, ema_model, val_loader, optimizer, None, device,
                    grad_clip=0.0, is_train=False, epoch=epoch,
                    use_amp=use_amp, mp_dtype=mp_dtype,
                    desc=f"EMA Valid Epoch {epoch}",
                    batches_per_execution=batches_per_execution,
                )

        epoch_time = time.time() - epoch_start
        if main_proc:
            log_line = (
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"time: {epoch_time:.1f}s | "
                f"train_loss: {train_metrics['loss']:.4f} "
                f"(ppl {train_metrics['ppl']:.2f})"
            )
            if val_metrics and val_metrics["loss"] != float("inf"):
                log_line += (
                    f" | val_loss: {val_metrics['loss']:.4f} "
                    f"(ppl {val_metrics['ppl']:.2f})"
                )
            if ema_val_metrics and ema_val_metrics["loss"] != float("inf"):
                log_line += (
                    f" | ema_val: {ema_val_metrics['loss']:.4f} "
                    f"(ppl {ema_val_metrics['ppl']:.2f})"
                )
            log_line += f" | lr: {optimizer.param_groups[0]['lr']:.2e}"
            print(log_line)

            if config.use_moe:
                balance = raw_model.get_balance_stats()
                if balance:
                    bias_max = max(v for k, v in balance.items() if "bias_max" in k)
                    bias_means = [v for k, v in balance.items() if "bias_mean" in k]
                    bias_mean = sum(bias_means) / len(bias_means)
                    print(f"  MoE balance -> bias_mean: {bias_mean:.4f}, "
                          f"bias_max: {bias_max:.4f}")

        if args.sample_every > 0 and epoch % args.sample_every == 0 and main_proc:
            sample = sample_text(
                args.sample_prompt,
                args.sample_max_tokens,
                args.sample_temperature,
            )
            try:
                print(f"  Sample -> {sample}")
            except UnicodeEncodeError:
                safe = sample.encode("ascii", errors="replace").decode("ascii")
                print(f"  Sample -> {safe}")

        if main_proc:
            history.append({
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "ema_val": ema_val_metrics,
                "time": epoch_time,
            })

        if not args.no_save and epoch % args.save_every == 0:
            is_best = False
            if val_metrics and val_metrics["loss"] != float("inf"):
                is_best = val_metrics["loss"] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics["loss"]
            ckpt_metrics = val_metrics if val_metrics else train_metrics
            path = save_checkpoint_xla(
                xm, raw_model, optimizer, scheduler, epoch,
                ckpt_metrics, save_dir, is_best=is_best, muon_hp=muon_hp,
                ema_model=ema_model, keep_last_n=args.keep_last_n,
            )
            if main_proc:
                print(f"  -> checkpoint saved to {path}")
                if is_best:
                    print(f"  -> new best val_loss: {best_val_loss:.4f}")
        elif val_metrics and val_metrics["loss"] != float("inf"):
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]

    if main_proc:
        history_path = save_dir / "history.json"
        with history_path.open("w") as f:
            json.dump(
                {
                    "model": args.model,
                    "epochs": args.epochs,
                    "final_metrics": val_metrics or train_metrics,
                    "history": history,
                },
                f,
                indent=2,
            )
        print(f"\nHistory written to {history_path}")
    _xla_barrier(xm, "history-epoch")

    if not args.no_save:
        final_path = save_final_xla(
            xm,
            {
                "epoch": args.epochs,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "muon_hp_state_dict": muon_hp.state_dict() if muon_hp else None,
                "ema_state_dict": ema_model.state_dict() if ema_model else None,
                "metrics": val_metrics or train_metrics,
                "history": history,
            },
            save_dir,
        )
        if main_proc:
            print(f"Final checkpoint saved to {final_path}")
    elif main_proc:
        print("(--no-save active: no model weights written)")

    if main_proc:
        print("Training complete!")
    if wandb_run is not None:
        wandb_run.finish()
    _xla_barrier(xm, "done-epoch")


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        args = build_arg_parser().parse_args()
    base.validate_token_superposition_args(args)
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    torch_xla, _, _ = _import_xla()

    if hasattr(torch_xla, "launch") and args.xla_num_devices is None:
        torch_xla.launch(
            _xla_worker,
            args=(args,),
            debug_single_process=args.xla_single_process,
        )
        return

    import torch_xla.distributed.xla_multiprocessing as xmp

    xmp.spawn(
        _xla_worker,
        args=(args,),
        nprocs=args.xla_num_devices,
    )


if __name__ == "__main__":
    main()
