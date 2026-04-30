#!/usr/bin/env python
"""Causal LM pre-training driver. ``--model`` selects the architecture from
``MODEL_REGISTRY``; the same pipeline (tokenisation, chunking, optimiser,
checkpointing, sampling) works for every registered model."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.baseline import BaselineConfig, BaselineTransformer, count_parameters
from models.linear import LinearConfig, LinearTransformer
from models.recursive import RecursiveConfig, RecursiveTransformer
from models.residual import ResidualConfig, ResidualTransformer
from models.superlinear import SuperLinearConfig, SuperLinearTransformer
from models.hybrid import HybridConfig, HybridTransformer
from models.logos import LogosConfig, LogosTransformer
from utils.tokenizer import TiktokenTokenizer


# ---------------------------------------------------------------------------
# Single-host multi-GPU (DDP) helpers
# ---------------------------------------------------------------------------
# Activated only when launched via ``torchrun --nproc_per_node=N`` (or when
# WORLD_SIZE > 1 in the environment). Single-GPU runs go through the same
# code paths with rank=0, world_size=1 and behave exactly as before.

def parse_token_count(value: str) -> int:
    """Parse '10B', '500M', '2.5G', '1e10', or a plain integer into a token count.

    Suffixes are case-insensitive: K=1e3, M=1e6, B=G=1e9, T=1e12. Used by
    --total-tokens so users can type a Chinchilla-style budget directly
    instead of computing the equivalent step count by hand.
    """
    s = value.strip().lower()
    if not s:
        raise argparse.ArgumentTypeError("empty token count")
    multipliers = {"k": 1e3, "m": 1e6, "b": 1e9, "g": 1e9, "t": 1e12}
    if s[-1] in multipliers:
        try:
            n = float(s[:-1]) * multipliers[s[-1]]
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"could not parse token count {value!r}: {e}"
            ) from None
    else:
        try:
            n = float(s)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"could not parse token count {value!r}: {e}"
            ) from None
    if n <= 0:
        raise argparse.ArgumentTypeError(
            f"token count must be positive, got {value!r}"
        )
    return int(n)


def init_distributed() -> tuple:
    """Read torchrun env vars, init NCCL, set the per-rank CUDA device.
    Returns ``(rank, local_rank, world_size)``. Single-GPU returns
    ``(0, 0, 1)`` and skips ``init_process_group``."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    visible = torch.cuda.device_count()
    if local_rank >= visible:
        raise RuntimeError(
            f"DDP rank {rank} (local_rank={local_rank}) cannot bind to "
            f"cuda:{local_rank}: only {visible} CUDA device(s) visible to "
            f"this process. Re-launch with --nproc_per_node={visible} (or "
            f"set CUDA_VISIBLE_DEVICES to expose more GPUs). "
            f"Current CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}."
        )
    # Pin the device BEFORE init_process_group and pass device_id so NCCL
    # locks the PG to this rank's GPU up front. Without this, recent
    # torch versions warn "No device id is provided ... using GPU N as
    # device used by this process is currently unknown."
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)
    return rank, local_rank, world_size


def is_main_process() -> bool:
    return (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_rank() == 0
    )


def configure_compile_logging(show_autotune_logs: bool) -> None:
    """Keep TorchInductor max-autotune from dumping every benchmark choice."""
    if show_autotune_logs:
        return
    try:
        import torch._inductor.config as inductor_config
    except Exception:
        return
    inductor_config.autotune_num_choices_displayed = 0
    inductor_config.max_autotune_report_choices_stats = False

    # Demote Inductor's select_algorithm logger so recoverable autotune
    # fallbacks don't spam stderr; only true failures surface.
    import logging
    logging.getLogger("torch._inductor.select_algorithm").setLevel(
        logging.CRITICAL,
    )

    # "Online softmax is disabled on the fly since Inductor decides to split
    # the reduction" is an informational lowering note, emitted via
    # warnings.warn from torch._inductor.lowering. Filter it so it does not
    # spam stderr once per compiled softmax.
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r"\s*Online softmax is disabled.*",
        category=UserWarning,
    )

    # Inductor's scheduler logs "Layout conflict detected for buf<N>" once per
    # matmul template whose chosen kernel wanted a contiguous stride but the
    # buffer arrived transposed (e.g. KDA per-head views, stacked MoE expert
    # weights). Codegen is still correct — Inductor just falls back to a
    # less-fused path or inserts a copy. Drop only those records so genuine
    # scheduler errors still surface.
    class _DropLayoutConflict(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "Layout conflict detected" not in record.getMessage()

    logging.getLogger("torch._inductor.scheduler").addFilter(
        _DropLayoutConflict(),
    )


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    """Average a scalar/tensor across ranks in place. No-op single-GPU."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Weights & Biases (optional)
# ---------------------------------------------------------------------------
# Lazy import so wandb stays an optional dependency. Only the main process
# initialises a run; non-main ranks see ``wandb_run = None`` and skip every
# logging call. Per-step + per-eval + per-sample metrics flow through the
# global ``wandb.log`` API, so as long as the run exists on rank 0 we can
# call it from anywhere on rank 0 without threading state through.

def init_wandb(args: argparse.Namespace, world_size: int):
    """Initialise a W&B run on the main process. Returns the run handle
    or ``None`` if --wandb is off or we're not rank 0. Errors loudly if
    --wandb is set but the package is missing."""
    if not args.wandb:
        return None
    if not is_main_process():
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "--wandb was set but the `wandb` package is not installed. "
            "Install with `pip install wandb` (or "
            "`pip install -e .[wandb]` from the project root)."
        ) from exc
    config = {**vars(args), "world_size": world_size}
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        config=config,
    )
    return run


def wandb_log(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Rank-0 wandb.log wrapper that no-ops when wandb isn't initialised
    (either --wandb off, non-main rank, or wandb missing)."""
    if not is_main_process():
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    wandb.log(metrics, step=step)


def _moe_layer_names(args: argparse.Namespace, n_entries: int) -> list:
    """Map every position in the model's ``topk_indices_list`` to a
    human-readable layer label. Mirrors ``LogosTransformer.update_router_biases``
    so the wandb panel grouping matches the actual block layout. Falls
    back to ``layer_{i}`` for non-Logos models."""
    if args.model != "logos":
        return [f"layer_{i}" for i in range(n_entries)]
    names: list = []
    for i in range(args.num_entry_layers):
        names.append(f"entry_{i}")
    for l in range(args.num_loops):
        for r in range(args.num_body_layers):
            names.append(f"body_{r}_loop_{l}")
    for i in range(args.num_exit_layers):
        names.append(f"exit_{i}")
    if len(names) > n_entries:
        names = names[:n_entries]
    while len(names) < n_entries:
        names.append(f"layer_{len(names)}")
    return names


def _moe_load_metrics(
    topk_indices: torch.Tensor, num_experts: int,
) -> Dict[str, Any]:
    """Per-MoE-layer load distribution. Uses ``bincount`` over the flat
    expert-id stream (cheaper than the ``one_hot`` round-trip). Sums
    counts across DDP ranks so the reported fractions reflect the
    cluster-wide routing decision, not just rank-0's shard. Returns
    summary scalars (max/min/std/dead/KL-from-uniform) and the raw
    fraction tensor on CPU for ``wandb.Histogram``.

    All scalar reductions are stacked and pulled to CPU in a single
    sync — repeated ``.item()`` calls would each force a GPU stall, and
    with ~22 MoE layers this dominates a logging step otherwise.
    """
    counts = torch.bincount(
        topk_indices.reshape(-1), minlength=num_experts,
    ).float()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    total = counts.sum().clamp(min=1.0)
    frac = counts / total
    target = 1.0 / num_experts
    # KL(frac || uniform) — peaks when load concentrates on few experts.
    eps = 1e-12
    kl = (frac * (torch.log(frac + eps) - math.log(target))).sum()
    scalars = torch.stack([
        frac.max(),
        frac.min(),
        frac.std(),
        (counts == 0).sum().to(frac.dtype),
        kl,
    ]).detach().cpu().tolist()
    return {
        "frac": frac.detach().cpu(),
        "load_max": scalars[0],
        "load_min": scalars[1],
        "load_std": scalars[2],
        "dead_experts": int(scalars[3]),
        "kl_uniform": scalars[4],
    }


def log_moe_load(
    args: argparse.Namespace,
    topk_indices_list,
    step: int,
) -> None:
    """Compute and wandb-log per-MoE-layer load distribution. All DDP ranks
    must participate because ``_moe_load_metrics`` performs all-reduces;
    only rank 0 constructs W&B objects and logs the final metrics. No-op
    when W&B logging is disabled or the model has no MoE in this forward
    (``topk_indices_list`` is None / all-None).

    Cost: one bincount + one all_reduce per layer (~22 layers in the
    default Logos config). Cheap enough to call on a step boundary like
    every 1000 steps."""
    if not args.wandb:
        return
    if topk_indices_list is None:
        return

    names = _moe_layer_names(args, len(topk_indices_list))
    num_experts = args.num_sparse_experts
    metrics: Dict[str, Any] = {}
    for name, topk in zip(names, topk_indices_list):
        if topk is None:
            continue
        m = _moe_load_metrics(topk, num_experts)
        if not is_main_process():
            continue
        prefix = f"moe/{name}"
        metrics[f"{prefix}/load_max"] = m["load_max"]
        metrics[f"{prefix}/load_min"] = m["load_min"]
        metrics[f"{prefix}/load_std"] = m["load_std"]
        metrics[f"{prefix}/dead_experts"] = m["dead_experts"]
        metrics[f"{prefix}/kl_uniform"] = m["kl_uniform"]
        import wandb
        metrics[f"{prefix}/load_hist"] = wandb.Histogram(
            sequence=m["frac"].tolist(), num_bins=num_experts,
        )
    if is_main_process() and metrics:
        import wandb
        if wandb.run is None:
            return
        wandb.log(metrics, step=step)


def _summarize_optimizer_state(optimizer) -> Dict[str, Dict[str, float]]:
    """Per-(sub-optimizer, param-group) summary of optimizer-state magnitudes.

    For each tensor state key (Muon's ``momentum_buffer``, AdamW's
    ``exp_avg`` / ``exp_avg_sq``) we compute one ``mean(|state|)`` per
    parameter and reduce them with a single ``stack().mean()`` per group
    — so the GPU sync cost is O(opt-groups × state-keys), not per-parameter.
    Scalar-style keys (``step``) are reported as the max across the
    group's parameters. Returns a nested ``{label: {field: scalar}}`` dict
    so the caller can both print a one-line summary and wandb-log under
    ``opt/<label>/<field>``.
    """
    out: Dict[str, Dict[str, float]] = {}
    sub_opts = getattr(optimizer, "optimizers", [optimizer])
    for opt in sub_opts:
        kind = type(opt).__name__.lower()
        for gi, g in enumerate(opt.param_groups):
            label = f"{kind}_g{gi}"
            buffers: Dict[str, list] = {}
            steps: list = []
            n_with_state = 0
            for p in g["params"]:
                st = opt.state.get(p)
                if not st:
                    continue
                n_with_state += 1
                for k, v in st.items():
                    if k == "step":
                        if torch.is_tensor(v):
                            steps.append(v.detach().reshape(()).float())
                        else:
                            steps.append(torch.tensor(float(v)))
                    elif torch.is_tensor(v) and v.numel() > 0:
                        buffers.setdefault(k, []).append(
                            v.detach().abs().float().mean()
                        )
            if n_with_state == 0:
                continue
            entry: Dict[str, float] = {
                "lr": float(g.get("lr", 0.0)),
                "n_params": float(len(g["params"])),
                "n_with_state": float(n_with_state),
            }
            for k, items in buffers.items():
                entry[f"mean_abs_{k}"] = torch.stack(items).mean().item()
            if steps:
                entry["step"] = float(torch.stack(steps).max().item())
            out[label] = entry
    return out


def _format_optimizer_state(summary: Dict[str, Dict[str, float]], step: int) -> str:
    parts: list = []
    for label, entry in summary.items():
        chunks: list = [f"step={int(entry.get('step', step))}"]
        for k, v in entry.items():
            if not k.startswith("mean_abs_"):
                continue
            chunks.append(f"|{k[len('mean_abs_'):]}|={v:.3e}")
        parts.append(f"{label} " + " ".join(chunks))
    return "[opt {0}] ".format(step) + " | ".join(parts)


def _moe_max_load_local(
    topk_indices_list, num_experts: int,
) -> Optional[Tuple[float, List[float]]]:
    """Cluster-agnostic local approximation of max-load-fraction across
    layers, intended for tqdm postfix and per-layer wandb keys. Skips
    the all-reduce that ``_moe_load_metrics`` performs (1 sync per layer
    becomes 1 sync per update_every-call total). Returns ``(global_max,
    per_layer_list)`` so callers can log each layer separately and spot
    a single layer collapsing while the global max still looks healthy.
    """
    if topk_indices_list is None or num_experts <= 0:
        return None
    fractions: list = []
    for topk in topk_indices_list:
        if topk is None:
            continue
        counts = torch.bincount(
            topk.reshape(-1), minlength=num_experts,
        ).float()
        fractions.append(counts.max() / counts.sum().clamp_min(1))
    if not fractions:
        return None
    # Single device->host sync: stack once, transfer the whole vector,
    # then reduce on the CPU side.
    per_layer = torch.stack(fractions).cpu().tolist()
    return max(per_layer), per_layer


MODEL_REGISTRY: Dict[str, tuple] = {
    "baseline": (BaselineConfig, BaselineTransformer),
    "linear": (LinearConfig, LinearTransformer),
    "recursive": (RecursiveConfig, RecursiveTransformer),
    "residual": (ResidualConfig, ResidualTransformer),
    "superlinear": (SuperLinearConfig, SuperLinearTransformer),
    "hybrid": (HybridConfig, HybridTransformer),
    "logos": (LogosConfig, LogosTransformer),
}


class MultiOptimizer:
    """Iterates ``step``/``zero_grad``/state across a list of optimizers
    so the training loop stays agnostic to single-vs-multi setups.

    Used to drive Muon (2D weights) and AdamW (1D scalars) jointly: each
    needs its own state and its own scheduler since their lr scales (and
    decay tracks) are independent.
    """

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None) -> None:
        for o in self.optimizers:
            o.step()

    def state_dict(self):
        return {"optimizers": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state) -> None:
        for o, s in zip(self.optimizers, state["optimizers"]):
            o.load_state_dict(s)

    @property
    def param_groups(self):
        groups = []
        for o in self.optimizers:
            groups.extend(o.param_groups)
        return groups


class MultiScheduler:
    """Drives a per-optimizer scheduler list. A single shared scheduler
    would only step one of the underlying optimizers, leaving the other
    on its initial lr."""

    def __init__(self, schedulers):
        self.schedulers = list(schedulers)

    def step(self) -> None:
        for s in self.schedulers:
            s.step()

    def state_dict(self):
        return {"schedulers": [s.state_dict() for s in self.schedulers]}

    def load_state_dict(self, state) -> None:
        for s, st in zip(self.schedulers, state["schedulers"]):
            s.load_state_dict(st)


def split_param_groups(model):
    """Four-way split for the Muon/AdamW recipe.

    * ``muon``    — exactly-2D weights inside transformer blocks
                    (``nn.Linear.weight``). PyTorch's Muon hard-rejects
                    non-2D tensors, so this also enforces correctness.
    * ``embed``   — ``token_emb.weight`` (and ``lm_head.weight`` if the
                    model untied them). Routed to AdamW with a higher
                    base lr because input embeddings receive sparse
                    per-token gradients and benefit from larger steps.
    * ``default`` — RMSNorm scales, attention sink logits, biases (1D),
                    plus 3D ``Conv1d`` kernels in the SuperLinear path.
                    Handled by AdamW at the standard base lr.
    * ``router``  — MoE ``Router.linear.weight`` matrices. Pulled out
                    of ``muon`` so they can run on AdamW with a smaller
                    LR and a longer warmup; the bias-balance mechanism
                    needs time to find a balanced equilibrium before
                    router weights start moving aggressively, otherwise
                    early imbalances lock in a fixed top-K subset.

    Tied tensors are deduped by ``id()`` — when ``lm_head.weight is
    token_emb.weight`` (current default in every model here), the
    shared tensor lands in the ``embed`` group exactly once.
    """
    embed_ids: set[int] = set()
    for attr in ("token_emb", "lm_head"):
        mod = getattr(model, attr, None)
        if mod is not None and hasattr(mod, "weight"):
            embed_ids.add(id(mod.weight))

    router_ids: set[int] = set()
    for name, param in model.named_parameters():
        # Match e.g. ``...ffn.router.linear.weight`` across all model
        # variants that use the shared ``MoELayer`` from baseline.py.
        if name.endswith("router.linear.weight"):
            router_ids.add(id(param))

    seen: set[int] = set()
    muon: list = []
    embed: list = []
    default: list = []
    router: list = []
    for _, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if id(param) in embed_ids:
            embed.append(param)
        elif id(param) in router_ids:
            router.append(param)
        elif param.ndim == 2:
            muon.append(param)
        else:
            default.append(param)
    return muon, embed, default, router


def wsd_lr_lambda(warmup_steps: int, decay_steps: int, total_steps: int):
    """Warmup-Stable-Decay multiplier in [0, 1].

    Linear ramp 0→1 over the first ``warmup_steps``; constant 1
    through the stable phase; linear decay 1→0 over the final
    ``decay_steps``. Designed so a run can be interrupted or extended
    without re-tuning — the stable phase doesn't carry an irreversible
    schedule like cosine does.
    """
    decay_start = max(warmup_steps, total_steps - decay_steps)

    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step < decay_start:
            return 1.0
        if step >= total_steps:
            return 0.0
        return 1.0 - (step - decay_start) / max(1, total_steps - decay_start)

    return fn


def cosine_lr_lambda(warmup_steps: int, total_steps: int, min_ratio: float = 0.1):
    """Linear warmup followed by cosine decay to ``min_ratio`` of base lr."""
    import math

    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


class MuonHyperparamScheduler:
    """Schedules Muon-only ``momentum`` and ``weight_decay`` per step.

    Momentum follows a 3-phase ramp recommended by the Muon recipe:
    ``[0, m1)`` linearly ``mom_start → mom_mid``, ``[m1, m2)`` linearly
    ``mom_mid → mom_end``, then constant ``mom_end``. Weight decay
    linearly anneals ``wd_start → wd_end`` across ``total_steps``
    (early-strong-late-soft regularization).

    Stepped once per optimizer step, alongside the LR scheduler. Has
    no effect on AdamW groups — only Muon param groups are touched.
    """

    def __init__(
        self,
        muon_optimizers: list,
        total_steps: int,
        mom_start: float = 0.85,
        mom_mid: float = 0.90,
        mom_end: float = 0.95,
        m1: int = 150,
        m2: int = 300,
        wd_start: float = 0.2,
        wd_end: float = 0.0,
    ):
        self.muon_optimizers = list(muon_optimizers)
        self.total_steps = max(1, total_steps)
        self.mom_start = mom_start
        self.mom_mid = mom_mid
        self.mom_end = mom_end
        self.m1 = m1
        self.m2 = m2
        self.wd_start = wd_start
        self.wd_end = wd_end
        self._step = 0
        # Apply step-0 values immediately so the first optimizer step
        # uses scheduled hyperparameters, not the constructor defaults.
        self._apply()

    def _apply(self) -> None:
        s = self._step
        if s < self.m1:
            mom = self.mom_start + (self.mom_mid - self.mom_start) * s / max(1, self.m1)
        elif s < self.m2:
            mom = self.mom_mid + (self.mom_end - self.mom_mid) * (s - self.m1) / max(1, self.m2 - self.m1)
        else:
            mom = self.mom_end

        progress = min(1.0, s / self.total_steps)
        wd = self.wd_start + (self.wd_end - self.wd_start) * progress

        for opt in self.muon_optimizers:
            for g in opt.param_groups:
                g["momentum"] = mom
                g["weight_decay"] = wd

    def step(self) -> None:
        self._step += 1
        self._apply()

    def state_dict(self):
        return {"step": self._step}

    def load_state_dict(self, state) -> None:
        self._step = int(state["step"])
        self._apply()


def build_optimizer_and_scheduler(
    args: argparse.Namespace,
    total_steps: int,
    fused_adamw: bool,
    muon_params: list,
    embed_params: list,
    default_params: list,
    router_params: Optional[list] = None,
):
    """Build the Muon + AdamW pair and their schedulers.

    AdamW carries up to three param groups (embed, default, router)
    with different base lrs. The WSD/cosine multiplier scales each
    group's ``lr`` by its own lambda — embed/default share the global
    warmup, while router uses a longer warmup (``--router-warmup-steps``
    or ``3 * --warmup-steps`` by default). ``--no-muon`` collapses the
    matrix bucket into AdamW's default group at the AdamW lr; routers
    keep their dedicated group regardless.
    """
    router_params = list(router_params or [])
    use_muon = args.muon and len(muon_params) > 0
    if not use_muon:
        default_params = list(default_params) + list(muon_params)
        muon_params = []

    optimizers: list = []
    muon_optimizers: list = []
    if muon_params:
        muon_opt = torch.optim.Muon(
            muon_params,
            lr=args.muon_lr,
            weight_decay=args.muon_wd_start,
            momentum=args.muon_momentum_start,
            # No adjust_lr_fn — base lrs in this recipe assume raw Muon
            # scaling, not the AdamW-equivalent rescale.
        )
        optimizers.append(muon_opt)
        muon_optimizers.append(muon_opt)
    adamw_groups: list = []
    # Track per-group "kind" so the scheduler can pick the right lambda
    # (router groups get a longer warmup than embed/default).
    adamw_group_kinds: list = []
    if embed_params:
        adamw_groups.append({"params": embed_params, "lr": args.embed_lr})
        adamw_group_kinds.append("base")
    if default_params:
        adamw_groups.append({"params": default_params, "lr": args.lr})
        adamw_group_kinds.append("base")
    if router_params:
        adamw_groups.append({"params": router_params, "lr": args.router_lr})
        adamw_group_kinds.append("router")
    adamw_opt = None
    if adamw_groups:
        adamw_opt = torch.optim.AdamW(
            adamw_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
            fused=fused_adamw,
        )
        optimizers.append(adamw_opt)
    optimizer = MultiOptimizer(optimizers)

    router_warmup = (
        args.router_warmup_steps
        if args.router_warmup_steps is not None
        else max(args.warmup_steps, 3 * args.warmup_steps)
    )
    if args.scheduler == "wsd":
        decay_steps = (
            args.decay_steps
            if args.decay_steps is not None
            else max(1, int(args.decay_frac * total_steps))
        )
        base_lr_fn = wsd_lr_lambda(args.warmup_steps, decay_steps, total_steps)
        router_lr_fn = wsd_lr_lambda(router_warmup, decay_steps, total_steps)
    elif args.scheduler == "cosine":
        base_lr_fn = cosine_lr_lambda(args.warmup_steps, total_steps, min_ratio=0.1)
        router_lr_fn = cosine_lr_lambda(router_warmup, total_steps, min_ratio=0.1)
    else:
        raise ValueError(f"Unknown --scheduler '{args.scheduler}'")

    schedulers: list = []
    for o in optimizers:
        # LambdaLR multiplies each param-group's base lr (its initial
        # lr at construction) by the lambda's return value. AdamW gets
        # a per-group lambda list so router uses the longer warmup;
        # Muon's sole group uses the base lambda.
        if o is adamw_opt:
            lambdas = [
                router_lr_fn if k == "router" else base_lr_fn
                for k in adamw_group_kinds
            ]
        else:
            lambdas = [base_lr_fn] * len(o.param_groups)
        schedulers.append(
            torch.optim.lr_scheduler.LambdaLR(o, lr_lambda=lambdas)
        )
    scheduler = MultiScheduler(schedulers)

    if muon_optimizers and args.muon_schedule_hyperparams:
        muon_hp = MuonHyperparamScheduler(
            muon_optimizers,
            total_steps=total_steps,
            mom_start=args.muon_momentum_start,
            mom_mid=args.muon_momentum_mid,
            mom_end=args.muon_momentum_end,
            m1=args.muon_momentum_warmup_1,
            m2=args.muon_momentum_warmup_2,
            wd_start=args.muon_wd_start,
            wd_end=args.muon_wd_end,
        )
    else:
        muon_hp = None

    return optimizer, scheduler, muon_hp


def build_model(args: argparse.Namespace, vocab_size: int):
    """Instantiate config + model. Only fields the selected config accepts
    are forwarded; irrelevant flags are silently ignored."""
    if args.model not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{args.model}'. Available: {sorted(MODEL_REGISTRY)}"
        )
    ConfigCls, ModelCls = MODEL_REGISTRY[args.model]

    kwargs: Dict[str, Any] = dict(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_length,
        dropout=args.dropout,
        norm_eps=args.norm_eps,
        use_moe=args.use_moe,
        num_shared_experts=args.num_shared_experts,
        num_sparse_experts=args.num_sparse_experts,
        top_k=args.top_k,
        expert_d_ff=args.expert_d_ff,
        bias_update_rate=args.bias_update_rate,
        moe_diversity_factor=args.moe_diversity_factor,
        lm_head_chunk_size=args.lm_head_chunk_size,
        block_residual_isolate_softmax=args.block_residual_isolate_softmax,
    )
    if getattr(args, "capacity_factor", None) is not None:
        kwargs["capacity_factor"] = args.capacity_factor

    if args.model == "linear":
        head_dim = args.head_dim or (args.d_model // args.num_heads)
        kwargs.update(
            head_dim=head_dim,
            conv_size=args.conv_size,
            chunk_size=args.chunk_size,
        )
    elif args.model == "superlinear":
        head_dim = args.head_dim or (args.d_model // args.num_heads)
        kwargs.update(
            head_dim=head_dim,
            conv_size=args.conv_size,
            chunk_size=args.chunk_size,
            snapshot_interval=args.snapshot_interval,
            snapshot_latent_dim=args.snapshot_latent_dim,
            mem_top_k=args.mem_top_k,
            mem_head_dim=args.mem_head_dim,
            rope_scaling_type=args.rope_scaling_type,
            rope_scaling_factor=args.rope_scaling_factor,
            rope_original_max_position=args.rope_original_max_position,
            yarn_beta_fast=args.yarn_beta_fast,
            yarn_beta_slow=args.yarn_beta_slow,
        )
    elif args.model == "hybrid":
        head_dim = args.head_dim or (args.d_model // args.num_heads)
        kwargs.update(
            head_dim=head_dim,
            conv_size=args.conv_size,
            chunk_size=args.chunk_size,
            snapshot_interval=args.snapshot_interval,
            snapshot_latent_dim=args.snapshot_latent_dim,
            mem_top_k=args.mem_top_k,
            mem_head_dim=args.mem_head_dim,
            rope_scaling_type=args.rope_scaling_type,
            rope_scaling_factor=args.rope_scaling_factor,
            rope_original_max_position=args.rope_original_max_position,
            yarn_beta_fast=args.yarn_beta_fast,
            yarn_beta_slow=args.yarn_beta_slow,
            swa_window=args.swa_window,
            swa_every=args.swa_every,
            swa_offset=args.swa_offset,
        )
    elif args.model == "recursive":
        # num_layers is auto-derived from entry + body + exit.
        kwargs.update(
            num_entry_layers=args.num_entry_layers,
            num_body_layers=args.num_body_layers,
            num_exit_layers=args.num_exit_layers,
            num_loops=args.num_loops,
            body_gate_init_std=args.body_gate_init_std,
        )
    elif args.model == "residual":
        kwargs.update(num_blocks=args.num_blocks)
    elif args.model == "logos":
        # num_layers is auto-derived from entry + body + exit.
        head_dim = args.head_dim or (args.d_model // args.num_heads)
        kwargs.update(
            head_dim=head_dim,
            conv_size=args.conv_size,
            chunk_size=args.chunk_size,
            snapshot_interval=args.snapshot_interval,
            snapshot_latent_dim=args.snapshot_latent_dim,
            mem_top_k=args.mem_top_k,
            mem_head_dim=args.mem_head_dim,
            rope_scaling_type=args.rope_scaling_type,
            rope_scaling_factor=args.rope_scaling_factor,
            rope_original_max_position=args.rope_original_max_position,
            yarn_beta_fast=args.yarn_beta_fast,
            yarn_beta_slow=args.yarn_beta_slow,
            swa_window=args.swa_window,
            swa_every=args.swa_every,
            swa_offset=args.swa_offset,
            num_entry_layers=args.num_entry_layers,
            num_body_layers=args.num_body_layers,
            num_exit_layers=args.num_exit_layers,
            num_loops=args.num_loops,
            gradient_checkpointing=args.gradient_checkpointing,
            ckpt_granularity=args.ckpt_granularity,
        )

    config = ConfigCls(**kwargs)
    model = ModelCls(config)
    return config, model


def preprocess_dataset(
    dataset: DatasetDict,
    tokenizer: TiktokenTokenizer,
    max_length: int,
    text_column: str = "text",
) -> DatasetDict:
    """Tokenise, concatenate, and split into fixed-size blocks of ``max_length``."""

    def tokenize_function(examples: Dict[str, list]) -> Dict[str, list]:
        all_ids = []
        for text in examples[text_column]:
            ids = tokenizer.encode(text)
            ids.append(tokenizer.eos_token_id)
            all_ids.extend(ids)
        return {"input_ids": [all_ids]}

    def group_function(examples: Dict[str, list]) -> Dict[str, list]:
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)

        total_length = (len(concatenated) // max_length) * max_length
        if total_length == 0:
            return {"input_ids": []}

        return {
            "input_ids": [
                concatenated[i : i + max_length]
                for i in range(0, total_length, max_length)
            ]
        }

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset[list(dataset.keys())[0]].column_names,
        desc="Tokenising",
    )
    chunked = tokenized.map(
        group_function,
        batched=True,
        desc="Chunking",
    )
    return chunked


def create_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    prefetch_factor: int = 4,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """Wrap a finite tokenised HF dataset in a PyTorch DataLoader.

    Under DDP (``world_size > 1``) attaches a ``DistributedSampler`` so
    each rank sees a disjoint slice of the dataset every epoch. The
    sampler's ``set_epoch`` is driven from ``run_epoch`` to advance the
    shuffle seed each epoch.
    """

    class _Dataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset):
            self.data = hf_dataset

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            # labels alias input_ids; the subsequent torch.stack in
            # collate_fn produces a fresh tensor and the .to(device) in
            # the train loop materialises an independent device copy, so
            # no in-place mutation hazard reaches them either way.
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids,
            }

    def collate_fn(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }

    ds = _Dataset(dataset)
    sampler: Optional[torch.utils.data.distributed.DistributedSampler] = None
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        # DataLoader.shuffle must be False when a sampler is supplied.
        loader_shuffle = False
    else:
        loader_shuffle = shuffle

    loader_kwargs: Dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=loader_shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )
    # ``persistent_workers`` and ``prefetch_factor`` are only valid with
    # ``num_workers > 0``; passing them otherwise raises in PyTorch 2.x.
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(ds, **loader_kwargs)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
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
    log_every: int = 10,
) -> Dict[str, float]:
    model.train(is_train)

    # DistributedSampler needs ``set_epoch`` each epoch so its shuffle
    # seed advances; without it every epoch sees the same per-rank order.
    sampler = getattr(dataloader, "sampler", None)
    if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
        sampler.set_epoch(epoch)

    main = is_main_process()

    # Accumulate the weighted loss on-device so per-step ``.item()`` calls
    # don't stall the host between CUDA-graph replays under
    # ``--compile-mode reduce-overhead``. A single ``.item()`` at end of
    # epoch (plus a throttled postfix sync every ``log_every`` steps for
    # the progress bar) replaces what used to be one sync per step.
    total_loss_t = torch.zeros((), device=device, dtype=torch.float64)
    num_batches = 0
    log_every = max(1, log_every)

    pbar = tqdm(
        dataloader,
        desc=desc or f"{'Train' if is_train else 'Valid'} Epoch {epoch}",
        leave=False,
        disable=not main,
    )

    # Same CUDA-graphs handshake as the streaming loop: under
    # ``--compile-mode reduce-overhead`` outputs from the previous call
    # would otherwise be overwritten by the next graph replay. No-op
    # under default / max-autotune compile or when compile is off.
    cudagraph_mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)

    for step, batch in enumerate(pbar):
        if cudagraph_mark is not None:
            cudagraph_mark()
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        # Eval needs no_grad so activations don't pile up across iterations.
        grad_ctx = contextlib.nullcontext() if is_train else torch.no_grad()
        with grad_ctx, torch.autocast(device_type=device.type, dtype=mp_dtype, enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                # Pretraining batches are already fixed-length packed
                # chunks with no padding, so no key-padding mask is needed.
                attention_mask=None,
                labels=labels,
                is_causal=True,
            )

        loss = outputs["loss"]

        if is_train:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if muon_hp is not None:
                muon_hp.step()

            # Without this call, MoE bias stays at zero init forever and
            # aux-loss-free balancing never activates.
            topk_indices = outputs.get("topk_indices")
            if topk_indices is not None:
                model.update_router_biases(topk_indices)

            if ema_model is not None:
                # AveragedModel was built around the uncompiled module;
                # feed it the same uncompiled instance so parameter
                # ids match.
                inner = getattr(model, "_orig_mod", model)
                ema_model.update_parameters(inner)

        batch_size = input_ids.size(0)
        # GPU-side accumulation — no host sync. ``loss.detach()`` is read
        # by the addition kernel before the next ``cudagraph_mark_step_begin``
        # so the CUDA-graph output pool is safe to reuse on the next replay.
        total_loss_t += loss.detach().to(torch.float64) * batch_size
        num_batches += batch_size

        if main and step % log_every == 0:
            loss_val = loss.detach().float().item()
            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "ppl": f"{math.exp(min(loss_val, 20)):.2f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

    pbar.close()

    # Reduce sums across ranks so every rank reports the same global
    # average (matches ``evaluate()`` in the streaming path). For train
    # this is cosmetic — the optimizer step already syncs gradients via
    # DDP — but for val it's the difference between a per-rank shard
    # metric and the cluster-wide one. This is the only host sync for
    # the accumulated loss in the entire epoch.
    if dist.is_available() and dist.is_initialized():
        agg = torch.stack([
            total_loss_t,
            torch.tensor(float(num_batches), dtype=torch.float64, device=total_loss_t.device),
        ])
        dist.all_reduce(agg, op=dist.ReduceOp.SUM)
        total_loss = agg[0].item()
        num_batches = int(agg[1].item())
    else:
        total_loss = total_loss_t.item()

    if num_batches == 0:
        return {"loss": float("inf"), "ppl": float("inf")}

    avg_loss = total_loss / num_batches
    return {"loss": avg_loss, "ppl": math.exp(min(avg_loss, 20))}


def prune_old_checkpoints(save_dir: Path, keep_last_n: int) -> None:
    """Drop epoch checkpoints beyond the ``keep_last_n`` most recent.

    Only ``checkpoint_epoch_*.pt`` files are touched — ``best`` and
    ``final`` are preserved unconditionally so they remain reachable
    after pruning. Lex sort works because epoch is zero-padded.
    """
    if keep_last_n <= 0:
        return
    files = sorted(save_dir.glob("checkpoint_epoch_*.pt"))
    if len(files) <= keep_last_n:
        return
    for old in files[:-keep_last_n]:
        try:
            old.unlink()
        except OSError:
            pass


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    save_dir: Path,
    is_best: bool = False,
    muon_hp: Optional[Any] = None,
    ema_model: Optional[AveragedModel] = None,
    keep_last_n: int = 0,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    # Unwrap torch.compile so checkpoint keys load into a fresh, uncompiled model.
    inner = getattr(model, "_orig_mod", model)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": inner.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if muon_hp is not None:
        checkpoint["muon_hp_state_dict"] = muon_hp.state_dict()
    if ema_model is not None:
        checkpoint["ema_state_dict"] = ema_model.state_dict()

    path = save_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(checkpoint, path)

    if is_best:
        torch.save(checkpoint, save_dir / "checkpoint_best.pt")

    config_path = save_dir / "config.json"
    if not config_path.exists():
        with open(config_path, "w") as f:
            json.dump(vars(inner.config), f, indent=2, default=str)

    prune_old_checkpoints(save_dir, keep_last_n)

    return path


def find_latest_checkpoint(save_dir: Path) -> Optional[Path]:
    """Pick the most recent ``checkpoint_epoch_{N}.pt`` under ``save_dir``.

    Naming is shared between the per-epoch and step-streaming paths: the
    streaming path stores the global step in the ``epoch`` slot, so the
    integer suffix orders correctly either way. ``checkpoint_final.pt``
    is intentionally not auto-resumed — its presence means a previous run
    completed, and silently rewinding to it would re-train the tail with a
    re-warmed scheduler. Pass ``--resume <path>`` if that is what you want.
    """
    if not save_dir.exists():
        return None
    candidates = []
    for p in save_dir.glob("checkpoint_epoch_*.pt"):
        m = re.match(r"checkpoint_epoch_(\d+)\.pt$", p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0])[1]


def _summarize_keys(keys: list, label: str, limit: int = 5) -> str:
    if not keys:
        return ""
    head = ", ".join(keys[:limit])
    tail = f", ... (+{len(keys) - limit} more)" if len(keys) > limit else ""
    return f"    {label} ({len(keys)}): {head}{tail}\n"


def load_resume_checkpoint(
    path: Path,
    raw_model: nn.Module,
    optimizer,
    scheduler,
    muon_hp,
    ema_model: Optional[AveragedModel],
    device: torch.device,
) -> int:
    """Load model + optimizer + scheduler + muon_hp + EMA state from a saved
    checkpoint. Returns the step (or epoch) to resume from.

    Tolerant of architecture drift (e.g. a feature was removed in the code
    after the checkpoint was saved): model and EMA loads are non-strict
    and warn about the missing / unexpected keys; optimizer and EMA loads
    that fail an internal validity check (e.g. param-count mismatch) are
    skipped with a warning so the model weights still come back even when
    the optimizer state has to restart from fresh moments. Scheduler and
    Muon-HP states have no architecture coupling and always load cleanly.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # ---- model ----
    incompat = raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    msg = ""
    msg += _summarize_keys(list(incompat.missing_keys),    "missing in checkpoint (using fresh init)")
    msg += _summarize_keys(list(incompat.unexpected_keys), "unexpected in checkpoint (ignored)")
    if msg:
        print(f"  [resume] model load skipped some keys:\n{msg}", end="")

    # ---- optimizer ----
    # If the saved optimizer state references parameters that no longer
    # exist (or vice versa), torch raises ValueError. The model weights
    # already loaded above, so the safe fallback is to start optimizer
    # state fresh (Adam moments / Muon momentum re-warm over a few hundred
    # steps).
    if "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, KeyError, RuntimeError) as exc:
            print(f"  [resume] optimizer state could not be reused "
                  f"({type(exc).__name__}: {exc}); starting moments fresh.")

    # ---- scheduler / muon_hp ----
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if muon_hp is not None and ckpt.get("muon_hp_state_dict") is not None:
        muon_hp.load_state_dict(ckpt["muon_hp_state_dict"])

    # ---- EMA ----
    # AveragedModel's state mirrors the live model, so same drift handling.
    # ``strict=False`` is a kwarg on torch.nn.Module.load_state_dict; it
    # propagates through AveragedModel since it's a plain Module subclass.
    if ema_model is not None and ckpt.get("ema_state_dict") is not None:
        try:
            ema_incompat = ema_model.load_state_dict(
                ckpt["ema_state_dict"], strict=False,
            )
            ema_msg = ""
            ema_msg += _summarize_keys(list(ema_incompat.missing_keys),    "missing in EMA (using fresh init)")
            ema_msg += _summarize_keys(list(ema_incompat.unexpected_keys), "unexpected in EMA (ignored)")
            if ema_msg:
                print(f"  [resume] EMA load skipped some keys:\n{ema_msg}", end="")
        except (RuntimeError, KeyError) as exc:
            print(f"  [resume] EMA state could not be reused "
                  f"({type(exc).__name__}: {exc}); starting EMA fresh.")

    # Streaming path saves ``epoch=step``; per-epoch path saves the epoch
    # number. ``checkpoint_final.pt`` (only reachable through an explicit
    # ``--resume <path>``) writes the step under ``"step"``.
    return int(ckpt.get("step", ckpt.get("epoch", 0)))


# ---------------------------------------------------------------------------
# Streaming corpus support
# ---------------------------------------------------------------------------

class PackedStream(torch.utils.data.IterableDataset):
    """Tokenize-on-the-fly, pack a stream of documents into fixed
    ``block_size`` chunks with EOS between docs. Yields the same
    ``{input_ids, attention_mask, labels}`` dict shape that the
    finite-corpus DataLoader produces, so downstream code is uniform.
    """

    def __init__(self, source, tokenizer, block_size: int, text_key: str = "text"):
        self.source = source
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_key = text_key

    def __iter__(self):
        # Shard across DataLoader workers so ``num_workers > 0`` doesn't
        # have every worker re-emit the same documents. Per-rank sharding
        # is already handled upstream by ``split_dataset_by_node``; this
        # adds the intra-rank stride.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            stride, offset = 1, 0
        else:
            stride, offset = worker_info.num_workers, worker_info.id
        buf: list = []
        for idx, sample in enumerate(self.source):
            if (idx % stride) != offset:
                continue
            ids = self.tokenizer.encode(sample[self.text_key])
            ids.append(self.tokenizer.eos_token_id)
            buf.extend(ids)
            while len(buf) >= self.block_size:
                chunk = buf[: self.block_size]
                buf = buf[self.block_size :]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                yield {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "labels": input_ids,
                }


# ---------------------------------------------------------------------------
# Dataset source resolution (shared by streaming + finite paths)
# ---------------------------------------------------------------------------
# Both training paths accept ``--dataset`` as either a HuggingFace Hub
# name or a local file/directory. The helpers below normalise both forms
# into ``load_dataset`` kwargs so the call site is a single
# ``load_dataset(**kw)`` regardless of source.
#
# For local paths we pin the parquet/json/csv/text builder explicitly
# via ``data_files`` rather than relying on HF's directory
# auto-resolution. Combined with ``streaming=True`` this avoids the
# multi-hundred-GB Arrow cache that the parquet builder would otherwise
# materialise for large partial snapshots (fineweb-edu sample-10BT etc).
#
# Iteration order in this dict picks the winning extension when a
# directory mixes formats; parquet wins because that's what HF ships.
_LOCAL_DATASET_FORMATS: Dict[str, str] = {
    ".parquet": "parquet",
    ".arrow":   "arrow",
    ".jsonl":   "json",
    ".json":    "json",
    ".csv":     "csv",
    ".txt":     "text",
}


def _resolve_local_data_files(path: Path) -> tuple:
    """Map a local file or directory to ``(builder_name, sorted files)``.

    Sorted output keeps every DDP rank's file ordering identical so
    ``split_dataset_by_node`` (streaming) and ``DistributedSampler``
    (finite) produce the same shard split across runs and ranks.
    """
    if path.is_file():
        fmt = _LOCAL_DATASET_FORMATS.get(path.suffix.lower())
        if fmt is None:
            raise ValueError(
                f"Unsupported local file format {path.suffix!r}; "
                f"expected one of {sorted(_LOCAL_DATASET_FORMATS)}"
            )
        return fmt, [str(path)]
    for ext, fmt in _LOCAL_DATASET_FORMATS.items():
        files = sorted(str(p) for p in path.rglob(f"*{ext}"))
        if files:
            return fmt, files
    raise FileNotFoundError(
        f"No supported data files under {path}; expected one of "
        f"{sorted(_LOCAL_DATASET_FORMATS)}"
    )


def _dataset_load_kwargs(
    dataset: str,
    dataset_config: Optional[str],
    streaming: bool,
    split: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the ``load_dataset(**kwargs)`` payload for either a Hub
    name or a local path. ``dataset_config`` is ignored for local
    paths (the explicit builder + ``data_files`` route doesn't take a
    Hub config name)."""
    path = Path(dataset)
    if path.exists():
        fmt, files = _resolve_local_data_files(path)
        kwargs: Dict[str, Any] = dict(
            path=fmt, data_files=files, streaming=streaming,
        )
    else:
        kwargs = dict(
            path=dataset, name=dataset_config, streaming=streaming,
        )
    if split is not None:
        kwargs["split"] = split
    return kwargs


def _describe_dataset_source(
    dataset: str, dataset_config: Optional[str],
) -> str:
    """Human-readable one-line summary for startup logging."""
    path = Path(dataset)
    if path.exists():
        fmt, files = _resolve_local_data_files(path)
        return f"local {dataset} ({fmt}, {len(files)} file(s))"
    return dataset + (f" / {dataset_config}" if dataset_config else "")


def build_streaming_loaders(
    args: argparse.Namespace,
    tokenizer,
    rank: int = 0,
    world_size: int = 1,
):
    """Build (train, val) loaders for a streaming dataset source.

    ``--dataset`` may be either a HuggingFace Hub name or a local
    file/directory; local paths route through the parquet/json/csv/
    text builder with ``streaming=True`` so HF never materialises an
    Arrow cache. The downstream pipeline (val pre-cache, DDP shard
    split, packing) is identical for both sources.

    Caches the first ``--val-docs`` documents into memory for validation
    and uses ``.skip(val_docs)`` on the training stream so the slices
    don't overlap. ``num_workers=0`` because IterableDataset across
    multiple workers needs explicit shard partitioning.

    Under DDP each rank gets a disjoint slice of the same stream via
    ``datasets.distributed.split_dataset_by_node``, so per-step batches
    don't overlap across ranks. The cached val docs are striped by
    ``rank::world_size``.
    """
    if not args.dataset:
        raise ValueError(
            "--streaming requires --dataset (HF name or local path)"
        )
    if is_main_process():
        print(f"Streaming dataset: "
              f"{_describe_dataset_source(args.dataset, args.dataset_config)}")

    load_kwargs = _dataset_load_kwargs(
        args.dataset, args.dataset_config,
        streaming=True, split="train",
    )
    val_stream = load_dataset(**load_kwargs)
    val_docs_full = list(val_stream.take(args.val_docs))
    if is_main_process():
        print(f"  cached {len(val_docs_full)} docs for val")
    train_stream = load_dataset(**load_kwargs).skip(args.val_docs)

    if world_size > 1:
        from datasets.distributed import split_dataset_by_node
        train_stream = split_dataset_by_node(train_stream, rank, world_size)
        val_docs = val_docs_full[rank::world_size]
        if is_main_process():
            print(f"  DDP shard: rank {rank}/{world_size}, "
                  f"val docs per rank ~ {len(val_docs)}")
    else:
        val_docs = val_docs_full

    train_ds = PackedStream(train_stream, tokenizer, args.max_length,
                            text_key=args.text_column)
    val_ds = PackedStream(val_docs, tokenizer, args.max_length,
                          text_key=args.text_column)

    def collate(batch):
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

    pin = torch.cuda.is_available()
    nw = max(0, int(args.num_workers))
    common_kw: Dict[str, Any] = dict(
        batch_size=args.batch_size,
        collate_fn=collate,
        num_workers=nw,
        pin_memory=pin,
    )
    if nw > 0:
        # PackedStream is now ``get_worker_info``-aware so each worker
        # consumes a strided slice of the upstream HF stream and they
        # don't duplicate samples.
        common_kw["persistent_workers"] = True
        common_kw["prefetch_factor"] = max(1, int(args.prefetch_factor))
    train_loader = DataLoader(train_ds, **common_kw)
    # ``drop_last=False`` for val: PackedStream + per-worker sharding can
    # leave each worker with fewer packed sequences than ``batch_size``
    # (especially at large ``max_length``), and ``drop_last=True`` would
    # then yield zero batches and make ``evaluate()`` return inf.
    val_loader = DataLoader(val_ds, drop_last=False, **common_kw)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Step-based training driver (used by the streaming path)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module, dataloader: DataLoader, device: torch.device,
    use_amp: bool, mp_dtype: torch.dtype, max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Single pass over ``dataloader``; returns ``{loss, ppl, n_skipped}``.
    Under DDP every rank evaluates its shard and we all-reduce a per-batch
    running sum so ranks agree on the reported metric. Batches whose loss
    is NaN/Inf are skipped (and counted) instead of poisoning the running
    sum — observed when the val path falls back to eager flex_attention
    after the compiled-cache busts during sampling, where bf16 score_mod
    can occasionally underflow into a -inf softmax row.
    """
    was_training = model.training
    model.train(False)
    loss_sum = torch.zeros(1, device=device, dtype=torch.float64)
    count = torch.zeros(1, device=device, dtype=torch.float64)
    skipped = torch.zeros(1, device=device, dtype=torch.float64)
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        ids = batch["input_ids"].to(device, non_blocking=True)
        lbls = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=mp_dtype, enabled=use_amp):
            out = model(input_ids=ids, attention_mask=None, labels=lbls, is_causal=True)
        loss_b = out["loss"].detach().to(torch.float64)
        finite = torch.isfinite(loss_b)
        loss_sum += torch.where(finite, loss_b, torch.zeros_like(loss_b))
        count += finite.to(torch.float64)
        skipped += (~finite).to(torch.float64)
    model.train(was_training)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        dist.all_reduce(skipped, op=dist.ReduceOp.SUM)
    n = count.item()
    n_skipped = int(skipped.item())
    if n == 0:
        # n_seen lets the caller distinguish "empty loader" (n_skipped==0)
        # from "every batch was non-finite" (n_skipped==N>0); both result
        # in inf but have very different root causes.
        return {
            "loss": float("inf"),
            "ppl": float("inf"),
            "n_skipped": n_skipped,
            "n_seen": int(n_skipped),
        }
    avg = (loss_sum / count).item()
    return {
        "loss": avg,
        "ppl": math.exp(min(avg, 20)),
        "n_skipped": n_skipped,
        "n_seen": int(n + n_skipped),
    }


def run_step_training(
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
) -> Dict[str, Any]:
    """Step-bounded training. Mirrors the per-epoch path's per-step body
    (forward, backward, opt/sched/muon_hp, MoE bias update, EMA update)
    and runs eval/sample/checkpoint on step boundaries set by
    ``--eval-every`` / ``--sample-every`` / ``--save-every``.

    Returns ``{best_val_loss, history}``; the caller writes history.json
    + final checkpoint so the per-epoch and streaming paths produce the
    same on-disk artifacts.
    """
    grad_clip = args.grad_clip
    log_every = max(1, args.log_every)
    main = is_main_process()
    world_size = (
        dist.get_world_size()
        if dist.is_available() and dist.is_initialized()
        else 1
    )

    if main:
        print("\n" + "=" * 60)
        resume_note = f" | resuming at step {start_step}" if start_step else ""
        print(f"Streaming training: {total_steps} steps"
              + (f" | DDP world_size={world_size}" if world_size > 1 else "")
              + resume_note)
        print("=" * 60)

    history: list = []
    best_val_loss = float("inf")
    running_loss = 0.0
    running_count = 0
    nonfinite_skips = 0
    nonfinite_warned = False
    step = start_step
    t0 = time.time()
    # ``initial`` aligns the bar with the resumed position, while ``total``
    # stays at the full target so the eta reflects the remaining work.
    pbar = tqdm(total=total_steps, initial=start_step, desc="train", disable=not main)

    # ``reduce-overhead`` enables CUDA graphs, which reuse the same
    # device-memory pool across invocations. Tensors returned from one
    # compiled forward (loss, topk_indices, ...) are held in that pool
    # and would be overwritten when the next call starts — torch raises
    # "accessing tensor output of CUDAGraphs that has been overwritten by
    # a subsequent run" via update_router_biases / loss.backward(). Marking
    # step start tells CUDA graphs to copy any still-referenced outputs
    # out of the pool first. The call is a no-op under default /
    # max-autotune compile and when compile is off, so unconditional.
    cudagraph_mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)

    model.train(True)
    train_iter = iter(train_loader)
    while step < total_steps:
        if cudagraph_mark is not None:
            cudagraph_mark()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch["input_ids"].to(device, non_blocking=True)
        lbls = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=mp_dtype, enabled=use_amp):
            outputs = model(input_ids=ids, attention_mask=None, labels=lbls, is_causal=True)
        loss = outputs["loss"]
        loss.backward()
        # Capture pre-clip grad norm so wandb can plot it; clip_grad_norm_
        # returns the total norm regardless of whether clipping kicked in.
        grad_norm = (
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if grad_clip > 0 else None
        )

        # NaN/Inf guard: one bad batch (e.g., a -inf softmax row from
        # eager flex_attention fallback) would otherwise propagate
        # NaN gradients into the optimizer and permanently poison the
        # parameters. We sync once per step (loss + grad_norm) and skip
        # optimizer.step / scheduler.step / muon_hp.step / router-bias /
        # EMA when either is non-finite. Set-to-none zero_grad above
        # already left grads cleared; the explicit zero_grad here is a
        # belt-and-suspenders that also clears any NaN grads produced
        # by this step's backward before the next iteration starts.
        finite_scalars = [loss.detach().reshape(1)]
        if grad_norm is not None:
            finite_scalars.append(grad_norm.detach().reshape(1))
        finite_check = torch.isfinite(
            torch.cat(finite_scalars)
        ).all().item()

        topk_indices = outputs.get("topk_indices")
        if finite_check:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if muon_hp is not None:
                muon_hp.step()

            # Router-bias and EMA updates always go through the unwrapped
            # raw model so DDP/compile attribute forwarding can't desync
            # the bias buffer or break AveragedModel's parameter-id
            # snapshot.
            if topk_indices is not None:
                raw_model.update_router_biases(topk_indices)
            if ema_model is not None:
                ema_model.update_parameters(raw_model)
        else:
            optimizer.zero_grad(set_to_none=True)
            nonfinite_skips += 1
            if main and not nonfinite_warned:
                print(
                    f"\n[train] non-finite loss/grad at step {step + 1} — "
                    f"skipping optim step. Subsequent skips will be "
                    f"counted silently and surfaced via the tqdm "
                    f"`skips=` postfix and wandb `train/nonfinite_skips`."
                )
                nonfinite_warned = True

        # Periodic MoE expert-load instrumentation. Aligned to step+1 so
        # the very first step never logs (avoids capturing untrained
        # routing). No-op when wandb isn't initialised or --moe-log-every
        # is 0.
        if (args.moe_log_every > 0
                and (step + 1) % args.moe_log_every == 0
                and topk_indices is not None):
            log_moe_load(args, topk_indices, step + 1)

        step += 1
        # All-reduce the per-rank loss for display so every rank logs the
        # same global average. Detach so the backward graph is untouched;
        # stack with optional grad_norm to pull everything to CPU in a
        # single sync instead of multiple .item() stalls.
        loss_d = loss.detach().reshape(1)
        if world_size > 1:
            loss_d = loss_d.clone()
            all_reduce_mean(loss_d)
        if grad_norm is not None:
            scalars = torch.cat([loss_d, grad_norm.detach().reshape(1)]).cpu().tolist()
            loss_val, grad_norm_val = scalars[0], scalars[1]
        else:
            loss_val = loss_d.cpu().item()
            grad_norm_val = None
        # Exclude non-finite losses from the running average so a single
        # bad batch doesn't poison the displayed loss; running_count
        # tracks finite contributions only.
        if math.isfinite(loss_val):
            running_loss += loss_val
            running_count += 1
        pbar.update(1)

        # Per-step W&B log (rank-0 only via wandb_log). Includes the raw
        # per-step loss, current LR for every param group (Muon /
        # AdamW-default / AdamW-embed), Muon-specific momentum + WD when
        # the muon_hp ramp is active, grad norm, and the running token
        # count so plots can be x-axis'd by tokens instead of steps.
        if main:
            tokens_seen = step * args.batch_size * args.max_length * world_size
            metrics = {
                "train/loss": loss_val,
                "train/ppl": math.exp(min(loss_val, 20)),
                "train/tokens_seen": tokens_seen,
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
            wandb_log(metrics, step=step)

        if step % log_every == 0 and main:
            if running_count > 0:
                avg = running_loss / running_count
                ppl = math.exp(min(avg, 20))
            else:
                # Entire window was non-finite; surface that explicitly
                # rather than dividing by zero.
                avg = float("nan")
                ppl = float("nan")
            elapsed = time.time() - t0
            # Cluster-wide tokens/sec for the *current* run — counts only
            # post-resume steps so an old checkpoint doesn't inflate the
            # rate against the just-now-elapsed wall time. Multiplies
            # per-rank tokens by world_size so the headline reflects
            # total throughput.
            steps_this_run = step - start_step
            tps = (steps_this_run * args.batch_size * args.max_length * world_size) / max(elapsed, 1)
            postfix = {
                "loss": f"{avg:.3f}", "ppl": f"{ppl:.1f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                "tok/s": f"{tps:.0f}",
            }
            # Approximate per-rank max expert-load fraction across all MoE
            # layers; cheap enough to update on every tqdm refresh and
            # gives early warning of router collapse without waiting for
            # the throttled --moe-log-every wandb dump. Also emits
            # per-layer keys so a single layer collapsing while the
            # global max stays healthy is still visible in wandb.
            moe_metrics = _moe_max_load_local(
                topk_indices, args.num_sparse_experts,
            ) if args.use_moe else None
            wandb_extra: Dict[str, Any] = {}
            if moe_metrics is not None:
                moe_max, moe_per_layer = moe_metrics
                postfix["moe_max"] = f"{moe_max:.3f}"
                wandb_extra["train/moe_max_load_local"] = moe_max
                for i, val in enumerate(moe_per_layer):
                    wandb_extra[f"train/moe_max_load_layer_{i:02d}"] = val
            if nonfinite_skips:
                postfix["skips"] = str(nonfinite_skips)
                wandb_extra["train/nonfinite_skips"] = nonfinite_skips
            pbar.set_postfix(postfix)
            wandb_log({
                "train/avg_loss": avg,
                "train/avg_ppl": ppl,
                "train/tok_per_sec": tps,
                **wandb_extra,
            }, step=step)
            running_loss = 0.0
            running_count = 0

        if (args.opt_state_log_every > 0
                and step % args.opt_state_log_every == 0):
            opt_summary = _summarize_optimizer_state(optimizer)
            if main and opt_summary:
                print(_format_optimizer_state(opt_summary, step))
                flat: Dict[str, float] = {}
                for label, entry in opt_summary.items():
                    for field, value in entry.items():
                        flat[f"opt/{label}/{field}"] = value
                wandb_log(flat, step=step)

        eval_due = (
            args.eval_every > 0 and step % args.eval_every == 0
            and val_loader is not None
        )
        save_due = (
            not args.no_save and args.save_every > 0
            and step % args.save_every == 0
        )

        if eval_due:
            # Eval runs on the uncompiled raw_model on every rank — sidesteps
            # any compile-time recompile from a different shape and keeps the
            # graph identical to training. all-reduce inside ``evaluate``
            # synchronises the metric across ranks before logging.
            val_metrics = evaluate(raw_model, val_loader, device, use_amp, mp_dtype)
            ema_val_metrics = None
            if ema_model is not None:
                ema_val_metrics = evaluate(
                    ema_model, val_loader, device, use_amp, mp_dtype,
                )
            if main:
                log = (f"\nstep {step:>6} | val_loss {val_metrics['loss']:.4f} "
                       f"(ppl {val_metrics['ppl']:.2f})")
                n_sk = val_metrics.get("n_skipped", 0)
                n_seen = val_metrics.get("n_seen", 0)
                if n_seen == 0:
                    # Distinguish 0-batch loader from all-non-finite —
                    # both render as inf but have different fixes.
                    log += " | EMPTY val_loader (0 batches)"
                elif n_sk:
                    log += f" | skipped {n_sk}/{n_seen} non-finite val batch(es)"
                if ema_val_metrics is not None and ema_val_metrics["loss"] != float("inf"):
                    log += (f" | ema_val {ema_val_metrics['loss']:.4f} "
                            f"(ppl {ema_val_metrics['ppl']:.2f})")
                    n_sk_ema = ema_val_metrics.get("n_skipped", 0)
                    n_seen_ema = ema_val_metrics.get("n_seen", 0)
                    if n_seen_ema and n_sk_ema:
                        log += f" | ema skipped {n_sk_ema}/{n_seen_ema}"
                print(log)
                history.append({
                    "step": step,
                    "val": val_metrics,
                    "ema_val": ema_val_metrics,
                })
                eval_log = {
                    "val/loss": val_metrics["loss"],
                    "val/ppl": val_metrics["ppl"],
                    "val/n_skipped": val_metrics.get("n_skipped", 0),
                }
                if ema_val_metrics is not None:
                    eval_log["ema_val/loss"] = ema_val_metrics["loss"]
                    eval_log["ema_val/ppl"] = ema_val_metrics["ppl"]
                    eval_log["ema_val/n_skipped"] = ema_val_metrics.get("n_skipped", 0)
                wandb_log(eval_log, step=step)
            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
            if not args.no_save and main:
                save_checkpoint(
                    raw_model, optimizer, scheduler,
                    epoch=step, metrics=val_metrics,
                    save_dir=save_dir, is_best=is_best,
                    muon_hp=muon_hp, ema_model=ema_model,
                    keep_last_n=args.keep_last_n,
                )
            barrier()
        elif save_due:
            if main:
                save_checkpoint(
                    raw_model, optimizer, scheduler,
                    epoch=step,
                    metrics={"loss": loss_val},
                    save_dir=save_dir, is_best=False,
                    muon_hp=muon_hp, ema_model=ema_model,
                    keep_last_n=args.keep_last_n,
                )
            barrier()

        if (args.sample_every > 0 and step % args.sample_every == 0
                and sample_text_fn is not None and main):
            try:
                generated = sample_text_fn(
                    args.sample_prompt, args.sample_max_tokens, args.sample_temperature,
                )
                print(f"  Sample -> {generated}")
                wandb_log({"sample/text": generated}, step=step)
            except UnicodeEncodeError as exc:
                print(f"  Sample -> (unicode encode failed: {exc})")
            model.train(True)

    pbar.close()
    if main:
        print(f"\n=== Training complete in {(time.time() - t0) / 60:.1f} min ===")
    return {"best_val_loss": best_val_loss, "history": history}


def build_arg_parser() -> argparse.ArgumentParser:
    """Single source of truth for the training CLI. Exported so notebooks
    and tests can build the same Namespace shape with
    ``build_arg_parser().parse_args([...])`` instead of duplicating the
    flag definitions."""
    parser = argparse.ArgumentParser(
        description="Train a decoder-only transformer with causal next-token prediction"
    )

    parser.add_argument("--model", type=str, default="baseline",
                        choices=sorted(MODEL_REGISTRY.keys()))

    parser.add_argument("--dataset", type=str, default="tiny_shakespeare",
                        help="HuggingFace Hub name, or a local file or "
                             "directory of parquet/arrow/jsonl/json/csv/"
                             "txt shards (e.g. a partial fineweb-edu "
                             "snapshot). Local paths route through the "
                             "explicit builder + data_files; in "
                             "--streaming mode this avoids HF's Arrow "
                             "cache entirely. ``--dataset-config`` is "
                             "ignored for local paths.")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Sequence block size")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes per rank. Use 0 to "
                             "load in the main process (debug only); 4–8 is a "
                             "good default for tokenized streams.")
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="DataLoader prefetch_factor (only used when "
                             "--num-workers > 0). Each worker keeps this many "
                             "batches queued ahead of the consumer.")
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=None)

    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1364,
                        help="Dense SwiGLU intermediate dim")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--use-moe", action="store_true")
    parser.add_argument("--num-shared-experts", type=int, default=2)
    parser.add_argument("--num-sparse-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--expert-d-ff", type=int, default=256)
    parser.add_argument("--capacity-factor", type=float, default=2.0,
                        help="Capacity factor for static-shape MoE dispatch. "
                             "Lower values reduce expert tensor size and "
                             "compute, but may drop over-capacity routed "
                             "tokens.")
    parser.add_argument("--moe-diversity-factor", type=float, default=0.0,
                        help="Cross-loop expert-diversity weight in the MoE "
                             "bias update. Relevant for ``recursive`` and "
                             "``logos`` (MoE weights reused across loop "
                             "iterations). 0 = standard balance only. Try "
                             "0.5–1.0 to encourage different expert "
                             "selections per loop step.")
    parser.add_argument("--bias-update-rate", type=float, default=0.02,
                        help="DeepSeek-style router bias update rate. "
                             "0.02 (default) gives the bias-balance loop "
                             "enough authority to outpace router-weight "
                             "drift under the body-loop sharing in "
                             "``logos`` / ``recursive`` (each forward "
                             "applies the same router num_loops times). "
                             "Drop to 0.01 for non-looped baselines or "
                             "for tighter convergence late in training.")
    parser.add_argument("--block-residual-isolate-softmax", action="store_true",
                        help="Route the BlockAttentionResidual depth-softmax "
                             "+ weighted-sum through an opaque "
                             "torch.library.custom_op so torch.compile can't "
                             "fuse softmax_backward with the upstream stack / "
                             "RMSNorm / dot-product chain. Fix for "
                             "'OutOfResources: shared memory' on Inductor's "
                             "fused softmax_backward kernel on Ada-class "
                             "consumer GPUs (sm_120 / RTX PRO 6000 Blackwell, "
                             "~99 KB SMEM/SM). Adds ~one graph break per "
                             "BlockAttentionResidual call (~97 in Logos at "
                             "default depth); typically <2%% throughput hit.")
    parser.add_argument("--lm-head-chunk-size", type=int, default=0,
                        help="Chunk token positions when computing LM-head "
                             "cross entropy, avoiding full [B,T,V] logits "
                             "during Logos training. 0 disables.")

    parser.add_argument("--head-dim", type=int, default=None,
                        help="[linear+] Per-head q/k/v dim. Defaults to "
                             "d_model // num_heads.")
    parser.add_argument("--conv-size", type=int, default=4,
                        help="[linear+] Short causal conv1d kernel size")
    parser.add_argument("--chunk-size", type=int, default=128,
                        help="[linear+] Chunk size for the KDA scan. Larger "
                             "chunks halve the Python loop trip count "
                             "(Nc = T/chunk_size) at the cost of O(C^2) "
                             "intra-chunk activation memory. Must be a "
                             "divisor of --snapshot-interval.")

    parser.add_argument("--num-entry-layers", type=int, default=2,
                        help="[recursive,logos] Standard blocks run once before the body loop")
    parser.add_argument("--num-body-layers", type=int, default=4,
                        help="[recursive,logos] Shared blocks applied on every loop step")
    parser.add_argument("--num-exit-layers", type=int, default=2,
                        help="[recursive,logos] Standard blocks run once after the body loop")
    parser.add_argument("--num-loops", type=int, default=4,
                        help="[recursive,logos] Number of times the body is applied")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="[logos] Recompute body activations during "
                             "backward (torch.utils.checkpoint, "
                             "use_reentrant=False). Trades wall time for "
                             "lower activation memory; usually required at "
                             "4K+ context with the body loop active. "
                             "Granularity controlled by --ckpt-granularity.")
    parser.add_argument("--ckpt-granularity", type=str, default="per-block",
                        choices=["per-block", "per-loop"],
                        help="[logos] When --gradient-checkpointing is on, "
                             "this picks the wrap granularity for the body. "
                             "'per-block' (default) wraps each body block "
                             "individually — backward holds at most one "
                             "block's activations at a time, lowest peak "
                             "memory. 'per-loop' wraps one full body-loop "
                             "iteration so torch.compile can fuse across "
                             "body blocks; backward holds num_body_layers "
                             "worth of activations at once, so only viable "
                             "with memory headroom.")
    parser.add_argument("--body-gate-init-std", type=float, default=0.0,
                        help="[recursive] Std for random init of the per-channel "
                             "A gate. 0 (default) reproduces the original "
                             "zero-init behaviour; small positive values "
                             "(e.g. 0.02) break the symmetry that leaves the "
                             "loop's residual mixing inert at step 0")

    parser.add_argument("--num-blocks", type=int, default=4,
                        help="[residual] Number of AttnRes blocks "
                             "(num_layers must be divisible by num_blocks)")

    parser.add_argument("--snapshot-interval", type=int, default=256,
                        help="[superlinear+] Token interval between KDA "
                             "snapshots (multiple of chunk_size)")
    parser.add_argument("--snapshot-latent-dim", type=int, default=128,
                        help="[superlinear+] MLA-compressed latent dim per head")
    parser.add_argument("--mem-top-k", type=int, default=16,
                        help="[superlinear+] Top-k snapshots retrieved per token")
    parser.add_argument("--mem-head-dim", type=int, default=64,
                        help="[superlinear+] Per-head dim for retrieval q/k/v")
    parser.add_argument("--rope-scaling-type", type=str, default="none",
                        choices=["none", "ntk", "yarn"],
                        help="[superlinear+] Retrieval RoPE scaling mode")
    parser.add_argument("--rope-scaling-factor", type=float, default=1.0,
                        help="[superlinear+] s = L_new / L_train. 1.0 disables.")
    parser.add_argument("--rope-original-max-position", type=int, default=None,
                        help="[superlinear+] Reference training context "
                             "length for RoPE scaling (default: max_seq_len)")
    parser.add_argument("--yarn-beta-fast", type=float, default=32.0,
                        help="[superlinear+] YaRN: extrapolation #-rotations threshold")
    parser.add_argument("--yarn-beta-slow", type=float, default=1.0,
                        help="[superlinear+] YaRN: interpolation #-rotations threshold")

    parser.add_argument("--swa-window", type=int, default=256,
                        help="[hybrid,logos] Sliding-window size")
    parser.add_argument("--swa-every", type=int, default=4,
                        help="[hybrid,logos] Place an SWA layer every N positions")
    parser.add_argument("--swa-offset", type=int, default=3,
                        help="[hybrid,logos] Position of SWA within each block "
                             "of size swa_every (Samba-style end-of-block default)")

    parser.add_argument("--tiktoken-encoding", type=str, default="cl100k_base")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=4e-3,
                        help="AdamW base lr for the 'default' group "
                             "(RMSNorm scales, sink logits, biases, conv "
                             "kernels). Acts as the peak lr of the WSD "
                             "stable phase before the global multiplier.")
    parser.add_argument("--embed-lr", type=float, default=0.2,
                        help="AdamW base lr for the embedding group "
                             "(token_emb / lm_head). When weights are tied "
                             "the shared tensor lands here and uses this lr.")
    parser.add_argument("--muon-lr", type=float, default=0.02,
                        help="Muon base lr for transformer-internal 2D "
                             "Linear weights. Raw Muon scaling — not "
                             "AdamW-equivalent rescaling.")
    parser.add_argument("--router-lr", type=float, default=4e-4,
                        help="AdamW base lr for MoE Router.linear.weight "
                             "matrices. Smaller than --lr to give the "
                             "DeepSeek-style bias-balance loop time to "
                             "stabilize load before router weights start "
                             "moving aggressively. Set explicitly when "
                             "tuning; the 10x-smaller-than-lr default is "
                             "a safe starting point, not optimal.")
    parser.add_argument("--router-warmup-steps", type=int, default=None,
                        help="Linear warmup steps for the router param "
                             "group. Defaults to 3 * --warmup-steps when "
                             "unset. Longer warmup gives the bias-balance "
                             "mechanism a head start so initial routing "
                             "imbalances don't lock in via fast router-"
                             "weight gradient updates.")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                        help="AdamW weight decay (constant across training).")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Linear warmup (0 -> 1 multiplier). Take 2-10%% "
                             "of total steps for the WSD recipe.")
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--scheduler", type=str, default="wsd",
                        choices=["wsd", "cosine"],
                        help="WSD = warmup -> stable -> linear cooldown "
                             "(default; supports interrupt/extend). "
                             "Cosine remains for compatibility.")
    parser.add_argument("--decay-steps", type=int, default=None,
                        help="WSD cooldown length in steps. Overrides "
                             "--decay-frac when set.")
    parser.add_argument("--decay-frac", type=float, default=0.2,
                        help="WSD cooldown length as fraction of total "
                             "steps when --decay-steps is unset.")

    parser.add_argument("--muon", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Route 2D Linear weights to Muon. --no-muon "
                             "falls back to AdamW for everything.")
    parser.add_argument("--muon-schedule-hyperparams",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Apply the 3-phase momentum ramp and the "
                             "linear weight-decay decay to Muon. Disable "
                             "to hold both at their starting values.")
    parser.add_argument("--muon-momentum-start", type=float, default=0.85)
    parser.add_argument("--muon-momentum-mid", type=float, default=0.90)
    parser.add_argument("--muon-momentum-end", type=float, default=0.95)
    parser.add_argument("--muon-momentum-warmup-1", type=int, default=150,
                        help="End step of momentum phase 1 (start -> mid).")
    parser.add_argument("--muon-momentum-warmup-2", type=int, default=300,
                        help="End step of momentum phase 2 (mid -> end).")
    parser.add_argument("--muon-wd-start", type=float, default=0.2,
                        help="Initial Muon weight decay; linearly anneals "
                             "to --muon-wd-end over total steps.")
    parser.add_argument("--muon-wd-end", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Default: checkpoints/{model}")
    parser.add_argument("--resume", type=str, default="auto",
                        metavar="auto|none|<path>",
                        help="Resume training from a saved checkpoint. "
                             "'auto' (default) picks the highest "
                             "checkpoint_epoch_{step}.pt under --save-dir "
                             "if any exist, else starts fresh. 'none' always "
                             "starts fresh. Otherwise treated as a path to a "
                             "specific .pt to load. Restores model + "
                             "optimizer + scheduler + muon_hp + EMA + step "
                             "counter so total_steps / WSD decay / muon "
                             "ramp continue from where they stopped. "
                             "Streaming dataloaders re-iterate from the "
                             "shard start — exact data position is not "
                             "checkpointed, so the first few batches after "
                             "resume may repeat.")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--no-save", action="store_true",
                        help="Skip model-weight checkpoints; still write history.json")
    parser.add_argument("--keep-last-n", type=int, default=0,
                        help="Roll over old epoch checkpoints, keeping only "
                             "the N most recent. checkpoint_best.pt and "
                             "checkpoint_final.pt are never pruned. "
                             "0 (default) keeps every epoch.")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="EMA decay for shadow weights. 0 (default) "
                             "disables. Typical values: 0.999 (fast follow) "
                             "or 0.9999 (strong smoothing). When enabled, an "
                             "extra validation pass runs on the EMA weights "
                             "and the EMA state is saved with each checkpoint.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--compile", action="store_true",
                        help="Wrap model with torch.compile")
    parser.add_argument("--compile-mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--compile-autotune-logs", action="store_true",
                        help="Show TorchInductor max-autotune benchmark tables. "
                             "By default they are suppressed because they can "
                             "emit thousands of stderr lines on large models.")
    parser.add_argument("--bf16", action="store_true",
                        help="Enable bfloat16 mixed precision (Ampere+)")
    parser.add_argument("--ddp-find-unused-parameters", action="store_true",
                        help="DDP only: pass find_unused_parameters=True. "
                             "Default False (faster). Enable if a forward "
                             "skips a parameter that has requires_grad=True; "
                             "with Logos top-k=4 over 32 experts at "
                             "bs*seq>=2k tokens per rank, every expert is "
                             "selected each step so this is not needed.")
    parser.add_argument("--ddp-static-graph", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="DDP only: enable static_graph=True so the bucket "
                             "plan is fused after the first iteration. Requires "
                             "every parameter to participate in every step; "
                             "auto-disabled when --ddp-find-unused-parameters "
                             "is set. Default off because DDP's fused compile "
                             "path can fail on graphs with dynamic-shape "
                             "submod boundaries with ``AttributeError: 'int' "
                             "object has no attribute 'meta'``; enable "
                             "explicitly when validated.")

    parser.add_argument("--wandb", action="store_true",
                        help="Log per-step + per-eval metrics to Weights & "
                             "Biases. Rank-0 only under DDP. Requires "
                             "`pip install -e .[wandb]` (or `pip install "
                             "wandb`) and a logged-in WANDB_API_KEY.")
    parser.add_argument("--wandb-project", type=str, default="logos-pretrain")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B team/user; falls back to default entity.")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="Display name for the run; auto-generated if omitted.")
    parser.add_argument("--wandb-tags", nargs="*", default=None,
                        help="Space-separated tags applied to the run.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                        default="online",
                        help="online: stream live; offline: write locally and "
                             "sync later with `wandb sync`.")
    parser.add_argument("--moe-log-every", type=int, default=1000,
                        help="Log MoE per-expert load fraction histogram + "
                             "summary scalars (max/min/std/dead/KL-from-uniform) "
                             "to W&B every N steps. 0 disables. Cheap enough "
                             "(~one bincount + all_reduce per MoE layer) to "
                             "leave on for the full run; useful for catching "
                             "router collapse early.")
    parser.add_argument("--opt-state-log-every", type=int, default=1000,
                        help="Log a one-line optimizer-state summary (per "
                             "sub-optimizer / param-group: step counter and "
                             "mean(|state|) for each tensor state key — "
                             "Muon's momentum_buffer, AdamW's exp_avg / "
                             "exp_avg_sq) every N steps. 0 disables. Useful "
                             "for verifying that resumed runs reload "
                             "optimizer moments (state should be non-zero "
                             "post-warmup) and for spotting collapsing or "
                             "exploding update magnitudes. Single sync per "
                             "(group, state-key) — cheap to leave on.")

    parser.add_argument("--sample-every", type=int, default=1,
                        help="In epoch mode: sample every N epochs. "
                             "In --streaming mode: sample every N steps.")
    parser.add_argument("--sample-prompt", type=str, default="Once upon a time")
    parser.add_argument("--sample-max-tokens", type=int, default=50)
    parser.add_argument("--sample-temperature", type=float, default=0.8)

    parser.add_argument("--streaming", action="store_true",
                        help="Stream the HF dataset (don't materialise it). "
                             "Required for corpora too large to download. "
                             "Switches training to a step-bounded loop "
                             "(--total-steps replaces --epochs); "
                             "--eval-every / --save-every / --sample-every "
                             "are reinterpreted as 'every N steps'.")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Required with --streaming unless --total-tokens "
                             "is given. Number of optimizer steps to run. "
                             "Determines the WSD/cosine schedule horizon and "
                             "the loop termination.")
    parser.add_argument("--total-tokens", type=parse_token_count, default=None,
                        help="Alternative to --total-steps: token budget for "
                             "the run. Accepts 10B / 500M / 2.5G / 1e10 / a "
                             "plain integer. Derives total_steps from "
                             "world_size * batch_size * max_length so a "
                             "fixed token budget stays fixed when you sweep "
                             "batch size or DDP world size. Mutually "
                             "exclusive with --total-steps.")
    parser.add_argument("--val-docs", type=int, default=200,
                        help="With --streaming: how many of the first docs "
                             "to cache in memory for validation. The training "
                             "stream skips the same count to avoid overlap.")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Streaming-mode tqdm postfix update / running-loss "
                             "averaging window in steps.")

    return parser


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        args = build_arg_parser().parse_args()

    rank, local_rank, world_size = init_distributed()
    main_proc = is_main_process()

    # Each rank gets a different RNG offset so streaming-dataset shuffling /
    # dropout / sample-time noise don't lock-step across ranks.
    torch.manual_seed(args.seed + rank)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if world_size > 1 else "cuda")
    else:
        device = torch.device("cpu")
    if main_proc:
        print(f"Using device: {device} | model: {args.model}"
              + (f" | DDP world_size={world_size}" if world_size > 1 else ""))

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    mp_dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32
    use_amp = (mp_dtype == torch.bfloat16)
    if use_amp and main_proc:
        print(f"  Mixed precision: {mp_dtype}")

    if args.save_dir is None:
        args.save_dir = f"checkpoints/{args.model}"
    save_dir = Path(args.save_dir)
    if main_proc:
        save_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    wandb_run = init_wandb(args, world_size)
    if wandb_run is not None:
        print(f"  W&B run: {wandb_run.url}")

    if main_proc:
        print(f"Loading tiktoken encoding: {args.tiktoken_encoding}")
    tokenizer = TiktokenTokenizer(args.tiktoken_encoding)
    if main_proc:
        print(f"  vocab_size: {tokenizer.vocab_size}")

    if args.streaming:
        if args.total_steps is not None and args.total_tokens is not None:
            raise ValueError(
                "--total-steps and --total-tokens are mutually exclusive; "
                "pick one. --total-tokens derives total_steps from "
                "world_size * batch_size * max_length."
            )
        if args.total_tokens is not None:
            tokens_per_step = world_size * args.batch_size * args.max_length
            args.total_steps = max(1, math.ceil(args.total_tokens / tokens_per_step))
            if main_proc:
                actual_tokens = args.total_steps * tokens_per_step
                print(
                    f"  Token budget: --total-tokens={args.total_tokens:,} "
                    f"-> total_steps={args.total_steps:,} "
                    f"({tokens_per_step:,} tokens/step, "
                    f"actual budget after rounding = {actual_tokens:,} "
                    f"tokens)"
                )
        if args.total_steps is None or args.total_steps <= 0:
            raise ValueError(
                "--streaming requires --total-steps N (N > 0) or "
                "--total-tokens (e.g. 10B); the schedule horizon and "
                "loop termination both depend on it."
            )
        if world_size > 1 and main_proc:
            print(f"  DDP token budget: per-step tokens = "
                  f"{world_size} x {args.batch_size} x {args.max_length} = "
                  f"{world_size * args.batch_size * args.max_length:,}. "
                  f"--total-tokens accounts for world_size automatically; "
                  f"if you set --total-steps manually, divide by "
                  f"{world_size} to keep the token budget fixed vs "
                  f"single-GPU.")
        train_loader, val_loader = build_streaming_loaders(
            args, tokenizer, rank=rank, world_size=world_size,
        )
    else:
        # ``--dataset`` may be a Hub name, a local file (.json/.csv/.txt),
        # or a local directory holding a saved-to-disk HF dataset
        # (e.g. a partial snapshot of fineweb-edu sample-10BT). Loading
        # is run on rank-0 first so the HF cache is warm before the
        # other ranks try to populate it; non-main ranks then read the
        # same fingerprinted files instead of racing on .map().
        def _load_and_preprocess():
            if main_proc:
                print(f"Loading dataset: "
                      f"{_describe_dataset_source(args.dataset, args.dataset_config)}")
            ds = load_dataset(**_dataset_load_kwargs(
                args.dataset, args.dataset_config, streaming=False,
            ))

            if main_proc:
                print(f"  splits: {list(ds.keys())}")

            if "validation" not in ds and "valid" not in ds:
                if main_proc:
                    print("  No validation split found – creating 10% hold-out from train")
                split = ds["train"].train_test_split(test_size=0.1, seed=args.seed)
                ds = DatasetDict({
                    "train": split["train"],
                    "validation": split["test"],
                })

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

            if main_proc:
                print(f"Preprocessing with block_size={args.max_length} ...")
            ds = preprocess_dataset(
                ds, tokenizer, args.max_length, text_column=args.text_column
            )
            return ds

        if main_proc:
            dataset = _load_and_preprocess()
        barrier()
        if not main_proc:
            dataset = _load_and_preprocess()
        barrier()

        if main_proc:
            print(f"  train examples:      {len(dataset['train'])}")
            print(f"  validation examples: {len(dataset['validation'])}")

        # Under DDP, attach DistributedSampler so each rank consumes a
        # disjoint shard each epoch. Val drops the trailing partial
        # batch when sharded so per-rank counts match exactly and the
        # all-reduce in ``run_epoch`` averages cleanly.
        train_loader = create_dataloader(
            dataset["train"], args.batch_size, shuffle=True,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            rank=rank, world_size=world_size, seed=args.seed,
        )
        val_loader = create_dataloader(
            dataset["validation"], args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            rank=rank, world_size=world_size, seed=args.seed,
            drop_last=(world_size > 1),
        )

    config, model = build_model(args, vocab_size=tokenizer.vocab_size)
    model = model.to(device)
    if main_proc:
        print(f"\nModel: {type(model).__name__}")
        print(f"  parameters: {count_parameters(model):,}")
        print(f"  use_moe:    {config.use_moe}")
        if config.use_moe:
            print(f"  shared experts: {config.num_shared_experts}")
            print(f"  sparse experts: {config.num_sparse_experts}")
            print(f"  top_k:          {config.top_k}")

    fused_adamw = (device.type == "cuda")
    total_steps = (
        args.total_steps if args.streaming
        else len(train_loader) * args.epochs
    )

    muon_params, embed_params, default_params, router_params = split_param_groups(model)
    optimizer, scheduler, muon_hp = build_optimizer_and_scheduler(
        args, total_steps, fused_adamw,
        muon_params, embed_params, default_params, router_params,
    )
    n_muon = sum(p.numel() for p in muon_params)
    n_embed = sum(p.numel() for p in embed_params)
    n_default = sum(p.numel() for p in default_params)
    n_router = sum(p.numel() for p in router_params)
    router_warmup = (
        args.router_warmup_steps
        if args.router_warmup_steps is not None
        else 3 * args.warmup_steps
    )
    if main_proc:
        if args.muon and n_muon > 0:
            print(f"  Optimizer: Muon + AdamW")
            print(f"    Muon (matrix, lr={args.muon_lr}): "
                  f"{len(muon_params)} tensors, {n_muon:,} params")
            print(f"    AdamW embed (lr={args.embed_lr}): "
                  f"{len(embed_params)} tensors, {n_embed:,} params")
            print(f"    AdamW default (lr={args.lr}): "
                  f"{len(default_params)} tensors, {n_default:,} params"
                  + (" (fused)" if fused_adamw else ""))
            if router_params:
                print(f"    AdamW router (lr={args.router_lr}, "
                      f"warmup {router_warmup}): "
                      f"{len(router_params)} tensors, {n_router:,} params")
            if muon_hp is not None:
                print(f"    Muon momentum: {args.muon_momentum_start} -> "
                      f"{args.muon_momentum_mid} (step {args.muon_momentum_warmup_1}) -> "
                      f"{args.muon_momentum_end} (step {args.muon_momentum_warmup_2})")
                print(f"    Muon weight_decay: {args.muon_wd_start} -> "
                      f"{args.muon_wd_end} linearly over {total_steps} steps")
        else:
            print(f"  Optimizer: AdamW only "
                  f"({n_muon + n_embed + n_default + n_router:,} params)"
                  + (" (fused)" if fused_adamw else ""))
        print(f"  LR schedule: {args.scheduler} "
              f"(warmup {args.warmup_steps}"
              + (f", decay {args.decay_steps}" if args.decay_steps else
                 f", decay {int(args.decay_frac * total_steps)}")
              + f" / {total_steps} total)")

    raw_model = model

    # Build EMA before torch.compile — AveragedModel snapshots the raw
    # parameter list, and the per-step update_parameters call inside
    # run_epoch hands it the same uncompiled instance via _orig_mod.
    # use_buffers=True so the EMA also tracks MoE router-bias buffers,
    # which evolve outside the optimizer via update_router_biases.
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

    # Resume runs into raw_model BEFORE DDP wrap and torch.compile so the
    # state-dict keys match the saved file (no ``module.`` / ``_orig_mod.``
    # prefixes to strip). Optimizer / scheduler / muon_hp / EMA all reload
    # from the same checkpoint so the WSD position, Muon momentum/WD ramp
    # and EMA shadow continue uninterrupted.
    start_step = 0
    resume_arg = (args.resume or "none").strip()
    if resume_arg.lower() != "none":
        if resume_arg.lower() == "auto":
            resume_path = find_latest_checkpoint(save_dir)
        else:
            resume_path = Path(resume_arg)
            if not resume_path.exists():
                raise FileNotFoundError(
                    f"--resume path {resume_path} does not exist"
                )
        if resume_path is not None:
            if main_proc:
                print(f"  Resuming from {resume_path}")
            start_step = load_resume_checkpoint(
                resume_path, raw_model, optimizer, scheduler,
                muon_hp, ema_model, device,
            )
            if main_proc:
                print(f"    resumed at step/epoch {start_step}")
        elif main_proc and resume_arg.lower() == "auto":
            print("  --resume auto: no prior checkpoint under "
                  f"{save_dir}, starting fresh")
    barrier()

    def sample_text(prompt: str, max_new_tokens: int, temperature: float) -> str:
        raw_model.train(False)
        prompt_ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long, device=device
        )
        with torch.no_grad():
            generated = raw_model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        return tokenizer.decode(generated[0])

    # DDP wrap goes BEFORE torch.compile so the compiled graph captures
    # DDP's gradient-bucket hooks. ``raw_model`` keeps a handle to the
    # underlying nn.Module for state_dict / EMA / sample / router-bias.
    if world_size > 1:
        # static_graph fuses the gradient-bucket plan after the first
        # iteration; requires every parameter to participate every step.
        # Mutually exclusive with find_unused_parameters=True.
        ddp_static_graph = (
            args.ddp_static_graph and not args.ddp_find_unused_parameters
        )
        if main_proc:
            print(f"  Wrapping model in DDP "
                  f"(find_unused_parameters={args.ddp_find_unused_parameters}, "
                  f"gradient_as_bucket_view=True, "
                  f"static_graph={ddp_static_graph})")
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=args.ddp_find_unused_parameters,
            gradient_as_bucket_view=True,
            static_graph=ddp_static_graph,
        )

    if args.compile:
        configure_compile_logging(args.compile_autotune_logs)
        # Logos issues distinct call sites at every entry/body/exit step:
        # different block modules, a growing ``blocks`` list (length 1..N),
        # and a per-loop body checkpoint closure that captures ``loop_idx``
        # as a default arg. That easily produces 20+ unique guard signatures
        # with --num-body-layers 6 --num-loops 3. The Dynamo default of 8
        # trips the recompile_limit and falls back to eager — bump the cache
        # so all legitimate variants compile.
        try:
            import torch._dynamo.config as dynamo_config
            for attr in ("recompile_limit", "cache_size_limit"):
                if hasattr(dynamo_config, attr):
                    setattr(dynamo_config, attr, max(64, getattr(dynamo_config, attr)))
        except Exception:
            pass
        if main_proc:
            print(f"  Compiling model with torch.compile (mode={args.compile_mode})...")
        model = torch.compile(model, mode=args.compile_mode)

    if args.streaming:
        result = run_step_training(
            args, model, raw_model, optimizer, scheduler, muon_hp, ema_model,
            train_loader, val_loader, device, use_amp, mp_dtype,
            save_dir, sample_text, total_steps, start_step=start_step,
        )
        history = result["history"]
        best_val_loss = result["best_val_loss"]
        val_metrics = (history[-1].get("val") if history
                       else {"loss": float("inf"), "ppl": float("inf")})
        train_metrics = val_metrics  # streaming path doesn't track per-epoch train metrics

        if main_proc:
            history_path = save_dir / "history.json"
            history_payload: Dict[str, Any] = {
                "model": args.model,
                "total_steps": args.total_steps,
                "final_metrics": val_metrics,
                "best_val_loss": best_val_loss,
                "history": history,
            }
            if wandb_run is not None:
                history_payload["wandb"] = {
                    "run_id": wandb_run.id,
                    "run_url": wandb_run.url,
                    "project": wandb_run.project,
                    "entity": wandb_run.entity,
                }
            with history_path.open("w") as f:
                json.dump(history_payload, f, indent=2, default=str)
            print(f"\nHistory written to {history_path}")

            if not args.no_save:
                final_path = save_dir / "checkpoint_final.pt"
                torch.save({
                    "step": total_steps,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "muon_hp_state_dict": muon_hp.state_dict() if muon_hp else None,
                    "ema_state_dict": ema_model.state_dict() if ema_model else None,
                    "metrics": val_metrics,
                    "history": history,
                }, final_path)
                print(f"Final checkpoint saved to {final_path}")
            else:
                print("(--no-save active: no model weights written)")
            print("Training complete!")
        if wandb_run is not None:
            wandb_run.finish()
        barrier()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        return

    if main_proc:
        print("\n" + "=" * 60)
        print("Starting training"
              + (f" | DDP world_size={world_size}" if world_size > 1 else ""))
        print("=" * 60)

    best_val_loss = float("inf")
    history: list[Dict[str, Any]] = []

    # Per-epoch path: ``start_step`` is the last completed epoch.
    if start_step >= args.epochs and main_proc:
        print(f"  --resume target ({start_step}) is at or past --epochs "
              f"({args.epochs}); nothing to train. Bump --epochs to extend.")
    for epoch in range(start_step + 1, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = run_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_clip=args.grad_clip, is_train=True, epoch=epoch,
            use_amp=use_amp, mp_dtype=mp_dtype, muon_hp=muon_hp,
            ema_model=ema_model, log_every=args.log_every,
        )

        val_metrics = None
        ema_val_metrics = None
        val_loader_nonempty = len(val_loader.dataset) > 0
        if epoch % args.eval_every == 0 and val_loader_nonempty:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            val_metrics = run_epoch(
                model, val_loader, optimizer, None, device,
                grad_clip=0.0, is_train=False, epoch=epoch,
                use_amp=use_amp, mp_dtype=mp_dtype,
                log_every=args.log_every,
            )
            if ema_model is not None:
                ema_val_metrics = run_epoch(
                    ema_model, val_loader, optimizer, None, device,
                    grad_clip=0.0, is_train=False, epoch=epoch,
                    use_amp=use_amp, mp_dtype=mp_dtype,
                    desc=f"EMA Valid Epoch {epoch}",
                    log_every=args.log_every,
                )

        epoch_time = time.time() - epoch_start

        if main_proc:
            log_line = (
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"time: {epoch_time:.1f}s | "
                f"train_loss: {train_metrics['loss']:.4f} (ppl {train_metrics['ppl']:.2f})"
            )
            if val_metrics and val_metrics["loss"] != float("inf"):
                log_line += (
                    f" | val_loss: {val_metrics['loss']:.4f} (ppl {val_metrics['ppl']:.2f})"
                )
            if ema_val_metrics and ema_val_metrics["loss"] != float("inf"):
                log_line += (
                    f" | ema_val: {ema_val_metrics['loss']:.4f} "
                    f"(ppl {ema_val_metrics['ppl']:.2f})"
                )
            log_line += f" | lr: {optimizer.param_groups[0]['lr']:.2e}"
            print(log_line)

        if config.use_moe and main_proc:
            balance = raw_model.get_balance_stats()
            if balance:
                bias_max = max(v for k, v in balance.items() if "bias_max" in k)
                bias_mean = sum(v for k, v in balance.items() if "bias_mean" in k) / len(
                    [k for k in balance if "bias_mean" in k]
                )
                print(f"  MoE balance -> bias_mean: {bias_mean:.4f}, bias_max: {bias_max:.4f}")

        if (args.sample_every > 0 and epoch % args.sample_every == 0
                and main_proc):
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
            if main_proc:
                ckpt_metrics = val_metrics if val_metrics else train_metrics
                path = save_checkpoint(
                    raw_model, optimizer, scheduler, epoch,
                    ckpt_metrics, save_dir, is_best=is_best, muon_hp=muon_hp,
                    ema_model=ema_model, keep_last_n=args.keep_last_n,
                )
                print(f"  -> checkpoint saved to {path}")
                if is_best:
                    print(f"  -> new best val_loss: {best_val_loss:.4f}")
            barrier()
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

        if not args.no_save:
            final_path = save_dir / "checkpoint_final.pt"
            torch.save(
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
                final_path,
            )
            print(f"Final checkpoint saved to {final_path}")
        else:
            print("(--no-save active: no model weights written)")
        print("Training complete!")
    if wandb_run is not None:
        wandb_run.finish()
    barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
