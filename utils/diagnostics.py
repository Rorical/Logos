"""Training-dynamics diagnostic monitor for the Logos model family.

Collects weight, gradient, and load-balance statistics from the raw model
at configurable intervals and emits warnings when metrics cross thresholds
that correlate with known training pathologies (plateau, NaN, collapse).

Usage from ``scripts/train.py``:

    monitor = DiagnosticsMonitor(args)
    ...
    if step % args.diagnostic_every == 0 and main:
        monitor.check(
            raw_model=raw_model, optimizer=optimizer,
            topk_indices=topk_indices, config=model_config,
            step=step, loss_val=loss_val,
            nonfinite_skips=nonfinite_skips, total_steps=total_steps,
        )

All operations are ``@torch.no_grad()`` and lightweight — typically < 5 ms
on a small model — so the default interval of 100 steps incurs negligible
wall-time overhead while still catching collapse before it propagates.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

if TYPE_CHECKING:
    from torch import nn


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


# Sentinel for values that couldn't be measured (e.g. model doesn't have the
# component, or it was skipped via config).
_UNMEASURED = float("nan")


@dataclass
class _Issue:
    message: str
    key: str          # stable key for deduplication / rate-limiting
    severe: bool      # True = red ALERT, False = yellow WARNING

    def format(self, step: int) -> str:
        prefix = _red("[ALERT]") if self.severe else _yellow("[WARN ]")
        return f"  {prefix} {self.message}"


# ---------------------------------------------------------------------------
# Threshold constants — tuned against observed plateaus in the Logos family
# ---------------------------------------------------------------------------

# Gradient ratios. The body loop applies the same weights ``num_loops`` times
# so body gradients are naturally ~ N× larger than entry/exit.  Beyond ~3×
# the per-group clip may be silently suppressing body updates.
GRAD_RATIO_WARN = 3.0
GRAD_RATIO_ALERT = 5.0

# BlockAttentionResidual ``proj`` stays at zero → routing never specialises.
PROJ_WARN_ABS_MEAN = 1e-4    # |proj| < this → warn
PROJ_WARN_AFTER_STEP = 200   # only after this many steps

# SnapshotRetrieval ``out_up`` near zero → memory branch is a no-op.
OUT_UP_WARN_ABS_MEAN = 1e-4   # was initialised at 1e-3 (Logos) or 0 (hybrid)
OUT_UP_WARN_AFTER_STEP = 200

# A/B gates in RecursiveBlock.  |A| > 1 for any channel makes the loop
# potentially unstable (h_{t+1} = A*h + ... repeated num_loops times).
GATE_A_WARN_MAX = 0.5
GATE_A_ALERT_MAX = 1.0
GATE_B_WARN_ABS_MEAN = 1e-4  # still near zero after warmup → never activated
GATE_B_WARN_AFTER_STEP = 500

# KDA decay: log_decay = -exp(A_log) * softplus(...).  If mean_per_step_decay
# is very close to 1 the state forgets nothing; if very close to 0 it erases
# everything. Both extremes cause the linear attention to behave like a fixed
# moving average with poor recall.  We monitor the raw A_log magnitude.
KDA_A_LOG_WARN_MIN = -4.0     # exp(-(-4)) = exp(4) → very fast decay
KDA_A_LOG_WARN_MAX = 3.0      # exp(-3) = 0.05 → very slow decay
KDA_DT_BIAS_WARN_NEG = -5.0   # dt_bias < -5 → softplus ≈ 0 → no decay updates

# MoE expert load collapse.
MOE_MAX_LOAD_WARN = 0.5
MOE_MAX_LOAD_ALERT = 0.8

# Non-finite skip rate.
SKIP_RATE_WARN = 0.01
SKIP_RATE_ALERT = 0.10

# PPL plateau detection: no improvement in loss for N diagnostic intervals.
PLATEAU_PATIENCE = 10          # number of intervals
PLATEAU_MIN_IMPROVEMENT = 0.001  # minimum absolute loss improvement
PLATEAU_COOLDOWN_STEPS = 500   # don't re-alert within this many steps

# Model capacity: warn when embedding dominates total params (no room for
# the transformer body to learn anything beyond a lookup table).
EMBED_FRACTION_WARN = 0.80     # embedding > 80% of params
EMBED_FRACTION_ALERT = 0.90    # embedding > 90% of params
BODY_PARAM_MIN = 500_000       # body < 500k params is too thin for 100k vocab


# ---------------------------------------------------------------------------
# Diagnostic monitor
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticsMonitor:
    """Periodic diagnostic scanner for Logos-family training runs."""

    # CLI knobs
    every: int = 100
    quiet: bool = False
    disable: bool = False
    warn_model_mismatch: bool = True

    # Internal state for rate-limiting and plateau detection
    _issued: Set[str] = field(default_factory=set, repr=False)
    _loss_history: List[float] = field(default_factory=list, repr=False)
    _last_plateau_step: int = field(default=-PLATEAU_COOLDOWN_STEPS, repr=False)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def check(
        self,
        *,
        raw_model: "nn.Module",
        optimizer,
        topk_indices: Optional[List[Optional[Any]]],
        config: Any,                    # BaselineConfig subclass
        step: int,
        loss_val: float,
        nonfinite_skips: int,
        total_steps: int,
    ) -> List[str]:
        """Run all diagnostics. Returns human-readable message lines (empty
        list when nothing to report). The caller is responsible for printing
        or logging them."""
        if self.disable:
            return []

        issues: List[_Issue] = []

        # --- Weight / parameter inspections ---
        issues.extend(self._check_proj(raw_model, step))
        issues.extend(self._check_out_up(raw_model, step))
        issues.extend(self._check_ab_gates(raw_model, config, step))
        issues.extend(self._check_kda_params(raw_model, config, step))
        issues.extend(self._check_capacity(raw_model, config, step))

        # --- Gradient inspections ---
        issues.extend(self._check_section_grads(raw_model, config, step))

        # --- MoE load balance ---
        issues.extend(self._check_moe_load(raw_model, config, topk_indices, step))

        # --- Non-finite skip rate ---
        issues.extend(self._check_skip_rate(nonfinite_skips, step))

        # --- PPL plateau ---
        issues.extend(self._check_plateau(loss_val, step, total_steps))

        # --- Optimizer state stats (brief) ---
        issues.extend(self._check_opt_state(optimizer, step))

        # Format
        if not issues:
            return []

        lines: List[str] = [f"[diag @ step {step}]"]
        for issue in issues:
            lines.append(issue.format(step))
            self._issued.add(issue.key)
        return lines

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_proj(self, model, step: int) -> List[_Issue]:
        """BlockAttentionResidual proj — should grow away from zero."""
        vals: List[float] = []
        for mod in model.modules():
            if hasattr(mod, "proj") and hasattr(mod, "norm"):
                w = mod.proj
                if isinstance(w, torch.Tensor) and w.numel() > 0:
                    vals.append(w.detach().abs().float().mean().item())
        if not vals:
            return []
        avg = sum(vals) / len(vals)
        if step < PROJ_WARN_AFTER_STEP:
            return []
        key = "proj_near_zero"
        if avg < PROJ_WARN_ABS_MEAN:
            if key not in self._issued:
                self._issued.add(key)
                return [_Issue(
                    key=key, severe=False,
                    message=(
                        f"BlockAttentionResidual proj mean |w| = {avg:.2e} — "
                        f"routing remains near-uniform. Expected >> 0 after "
                        f"step {PROJ_WARN_AFTER_STEP}."
                    ),
                )]
            return []
        self._issued.discard(key)
        return []

    def _check_out_up(self, model, step: int) -> List[_Issue]:
        """SnapshotRetrieval out_up — should grow away from zero."""
        vals: List[float] = []
        for name, mod in model.named_modules():
            # out_up is an nn.Linear with bias=False inside SnapshotRetrieval
            if hasattr(mod, "out_up") and hasattr(mod.out_up, "weight"):
                w = mod.out_up.weight
                if isinstance(w, torch.Tensor) and w.numel() > 0:
                    vals.append(w.detach().abs().float().mean().item())
        if not vals:
            return []
        avg = sum(vals) / len(vals)
        if step < OUT_UP_WARN_AFTER_STEP:
            return []
        key = "out_up_near_zero"
        if avg < OUT_UP_WARN_ABS_MEAN:
            if key not in self._issued:
                self._issued.add(key)
                return [_Issue(
                    key=key, severe=False,
                    message=(
                        f"SnapshotRetrieval out_up mean |w| = {avg:.2e} — "
                        f"memory branch still near-zero. Was initialised at "
                        f"1e-3 (Logos) or 0 (hybrid). If 0-initialised, try "
                        f"setting a small nonzero init in the model __init__."
                    ),
                )]
            return []
        self._issued.discard(key)
        return []

    def _check_ab_gates(self, model, config, step: int) -> List[_Issue]:
        """RecursiveBlock A/B per-channel gates."""

        issues: List[_Issue] = []
        has_body = hasattr(model, "body") and hasattr(model.body, "A")

        if not has_body:
            return []

        A = model.body.A
        B = model.body.B
        if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
            return []

        a_abs = A.detach().abs().float()
        b_abs = B.detach().abs().float()

        # A gate: warn if any channel exceeds thresholds
        a_max = a_abs.max().item()
        key_a_hi = "gate_A_high"
        if a_max > GATE_A_ALERT_MAX:
            issues.append(_Issue(
                key=key_a_hi, severe=True,
                message=(
                    f"Body A per-channel gate max |A| = {a_max:.3f} > "
                    f"{GATE_A_ALERT_MAX}. Loop h_{t+1} = A*h + ... may be "
                    f"UNSTABLE — A acts multiplicatively over {getattr(config, 'num_loops', '?')} "
                    f"steps. Consider clamping or sigmoid-wrapping A."
                ),
            ))
        elif a_max > GATE_A_WARN_MAX:
            issues.append(_Issue(
                key=key_a_hi, severe=False,
                message=(
                    f"Body A per-channel gate max |A| = {a_max:.3f} > "
                    f"{GATE_A_WARN_MAX}. Monitor for growth."
                ),
            ))
        else:
            self._issued.discard(key_a_hi)

        # B gate: warn if still near zero after warmup (deduped)
        key_b_lo = "gate_B_zero"
        if step > GATE_B_WARN_AFTER_STEP and b_abs.mean().item() < GATE_B_WARN_ABS_MEAN:
            if key_b_lo not in self._issued:
                self._issued.add(key_b_lo)
                issues.append(_Issue(
                    key=key_b_lo, severe=False,
                    message=(
                        f"Body B per-channel gate mean |B| = {b_abs.mean().item():.2e} — "
                        f"still near zero after step {step}. The injection signal 'e' "
                        f"may be ineffective. Try --body-gate-init-std 0.02."
                    ),
                ))
        else:
            self._issued.discard(key_b_lo)

        # A gate: warn if still zero (symmetry never broken) — deduped
        key_a_zero = "gate_A_zero"
        if step > GATE_B_WARN_AFTER_STEP and a_abs.mean().item() < GATE_B_WARN_ABS_MEAN:
            if key_a_zero not in self._issued:
                self._issued.add(key_a_zero)
                issues.append(_Issue(
                    key=key_a_zero, severe=False,
                    message=(
                        f"Body A per-channel gate mean |A| = {a_abs.mean().item():.2e} — "
                        f"still near zero after step {step}. Residual mixing is INERT. "
                        f"Try --body-gate-init-std 0.02."
                    ),
                ))
        else:
            self._issued.discard(key_a_zero)

        return issues

    def _check_kda_params(self, model, config, step: int) -> List[_Issue]:
        """KDA A_log and dt_bias parameters."""
        issues: List[_Issue] = []

        has_linear = False
        for mod in model.modules():
            if hasattr(mod, "A_log") and hasattr(mod, "dt_bias"):
                has_linear = True
                break
        if not has_linear:
            return []

        a_logs: List[float] = []
        dt_biases: List[float] = []
        for mod in model.modules():
            if hasattr(mod, "A_log"):
                a = mod.A_log
                if isinstance(a, torch.Tensor) and a.numel() > 0:
                    a_logs.append(a.detach().float().mean().item())
            if hasattr(mod, "dt_bias"):
                d = mod.dt_bias
                if isinstance(d, torch.Tensor) and d.numel() > 0:
                    dt_biases.append(d.detach().float().mean().item())

        if a_logs:
            avg_a_log = sum(a_logs) / len(a_logs)
            key_a_log = "kda_A_log_extreme"
            if avg_a_log < KDA_A_LOG_WARN_MIN:
                issues.append(_Issue(
                    key=key_a_log, severe=False,
                    message=(
                        f"KDA A_log mean = {avg_a_log:.3f} < {KDA_A_LOG_WARN_MIN} — "
                        f"per-step decay exp(-exp(A_log)) ≈ 0, state erases too fast. "
                        f"Attention may be unable to retain context."
                    ),
                ))
            elif avg_a_log > KDA_A_LOG_WARN_MAX:
                issues.append(_Issue(
                    key=key_a_log, severe=False,
                    message=(
                        f"KDA A_log mean = {avg_a_log:.3f} > {KDA_A_LOG_WARN_MAX} — "
                        f"per-step decay ≈ 1, state never forgets. "
                        f"Attention saturates after ~head_dim tokens."
                    ),
                ))
            else:
                self._issued.discard(key_a_log)

        if dt_biases:
            avg_dt = sum(dt_biases) / len(dt_biases)
            key_dt = "kda_dt_bias_low"
            if avg_dt < KDA_DT_BIAS_WARN_NEG:
                issues.append(_Issue(
                    key=key_dt, severe=False,
                    message=(
                        f"KDA dt_bias mean = {avg_dt:.3f} < {KDA_DT_BIAS_WARN_NEG} — "
                        f"softplus(dt_bias) ≈ 0, per-step decay frozen."
                    ),
                ))
            else:
                self._issued.discard(key_dt)

        return issues

    def _check_capacity(self, model, config, step: int) -> List[_Issue]:
        """Warn if the model's parameter budget is skewed toward the embedding,
        leaving too little capacity for the transformer body to learn."""
        issues: List[_Issue] = []

        total = 0
        embed_n = 0
        body_n = 0
        for name, p in model.named_parameters():
            n = p.numel()
            total += n
            if "token_emb" in name or "lm_head" in name:
                embed_n += n
            elif any(k in name for k in (".body.", ".entry.", ".exit.", "body.", "entry.", "exit.")):
                body_n += n
            elif "body." in name or "blocks." in name or "attn" in name.lower() or "ffn" in name.lower() or "moe" in name.lower():
                body_n += n

        if total == 0 or step <= 100:
            return issues

        embed_frac = embed_n / total

        key = "embed_dominance"
        if embed_frac > EMBED_FRACTION_ALERT:
            if key not in self._issued:
                self._issued.add(key)
                issues.append(_Issue(
                    key=key, severe=True,
                    message=(
                        f"Embedding is {embed_frac:.0%} of params ({embed_n:,}/{total:,}) — "
                        f"transformer body has only {body_n:,} params. "
                        f"The model is essentially a lookup table. "
                        f"Increase d_model (>={int(embed_n ** 0.5):,}) or "
                        f"use a smaller tokenizer."
                    ),
                ))
        elif embed_frac > EMBED_FRACTION_WARN:
            if key not in self._issued:
                self._issued.add(key)
                issues.append(_Issue(
                    key=key, severe=False,
                    message=(
                        f"Embedding is {embed_frac:.0%} of params ({embed_n:,}/{total:,}) — "
                        f"body has {body_n:,} params. "
                        f"Consider larger d_model for this vocab size."
                    ),
                ))
        else:
            self._issued.discard(key)

        if body_n > 0 and body_n < BODY_PARAM_MIN:
            key_body = "body_too_small"
            if key_body not in self._issued:
                self._issued.add(key_body)
                issues.append(_Issue(
                    key=key_body, severe=True,
                    message=(
                        f"Transformer body has only {body_n:,} params — "
                        f"below {BODY_PARAM_MIN:,} minimum for meaningful "
                        f"learning. Increase d_model, num_body_layers, or d_ff."
                    ),
                ))
        elif body_n >= BODY_PARAM_MIN:
            self._issued.discard("body_too_small")

        return issues

    def _check_section_grads(self, model, config, step: int) -> List[_Issue]:
        """Per-section (entry/body/exit) gradient norm comparison."""
        issues: List[_Issue] = []

        sections: Dict[str, List[float]] = {"entry": [], "body": [], "exit": []}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            n = param.grad.detach().norm().item()
            # Classify by parameter name.  Naming conventions:
            #  - LogosTransformer:  self.entry, self.body, self.exit
            #  - RecursiveTransformer:  self.entry, self.body, self.exit
            #  - ResidualTransformer:  self.blocks (entry/body/exit not applicable)
            for sec in sections:
                if f".{sec}." in name or name.startswith(f"{sec}."):
                    sections[sec].append(n)
                    break

        # Only warn if the model actually has separate body/entry/exit stacks.
        has_loops = (
            hasattr(model, "body")
            and hasattr(model, "entry")
            and hasattr(model, "exit")
        )
        if not has_loops:
            return []

        avg_entry = _safe_mean(sections["entry"])
        avg_body = _safe_mean(sections["body"])
        avg_exit = _safe_mean(sections["exit"])

        if avg_entry is None or avg_body is None:
            return []

        ratio_be = avg_body / max(avg_entry, 1e-12)
        ratio_bx = avg_body / max(avg_exit, 1e-12) if avg_exit is not None else None

        key = "grad_ratio_body_entry"
        if ratio_be > GRAD_RATIO_ALERT:
            if key not in self._issued:
                self._issued.add(key)
                issues.append(_Issue(
                    key=key, severe=True,
                    message=(
                        f"Body grad norm ({avg_body:.3f}) is "
                        f"{ratio_be:.1f}× entry ({avg_entry:.3f}). "
                        f"Shared body weights accumulate ~{getattr(config, 'num_loops', '?')}× "
                        f"gradient — the per-group grad_clip may be silently "
                        f"crushing body updates. "
                        f"Raise --grad-clip or scale body LR down."
                    ),
                ))
        elif ratio_be > GRAD_RATIO_WARN:
            if key not in self._issued:
                self._issued.add(key)
                issues.append(_Issue(
                    key=key, severe=False,
                    message=(
                        f"Body grad norm ({avg_body:.3f}) is "
                        f"{ratio_be:.1f}× entry ({avg_entry:.3f}) — "
                        f"expected asymmetry from loop sharing "
                        f"(num_loops={getattr(config, 'num_loops', '?')}). "
                        f"Monitor for growth above {GRAD_RATIO_WARN:.0f}×."
                    ),
                ))
        else:
            self._issued.discard(key)

        if ratio_bx is not None and ratio_bx > GRAD_RATIO_WARN:
            self._issued.discard(key)  # already warned via body/entry

        return issues

    def _check_moe_load(
        self, model, config, topk_indices, step: int,
    ) -> List[_Issue]:
        """MoE expert load balance warnings."""
        issues: List[_Issue] = []

        if not getattr(config, "use_moe", False):
            return []
        if topk_indices is None or len(topk_indices) == 0:
            return []

        num_experts = getattr(config, "num_sparse_experts", 0)
        if num_experts <= 0:
            return []

        # Compute per-layer max load fraction (local approx — no all-reduce).
        max_loads: List[float] = []
        collapsed_layers: List[int] = []
        for i, topk in enumerate(topk_indices):
            if topk is None:
                continue
            counts = torch.bincount(
                topk.reshape(-1), minlength=num_experts,
            ).float()
            total = counts.sum().clamp_min(1)
            frac = counts / total
            max_frac = frac.max().item()
            max_loads.append(max_frac)
            if max_frac > MOE_MAX_LOAD_ALERT:
                collapsed_layers.append(i)

        if not max_loads:
            return []

        global_max = max(max_loads)
        key = "moe_load_collapse"

        if global_max > MOE_MAX_LOAD_ALERT:
            issues.append(_Issue(
                key=key, severe=True,
                message=(
                    f"MoE expert load collapse: max load fraction = "
                    f"{global_max:.3f} > {MOE_MAX_LOAD_ALERT} in "
                    f"{len(collapsed_layers)} layer(s) "
                    f"({collapsed_layers[:5]}{'...' if len(collapsed_layers) > 5 else ''}). "
                    f"Raise --bias-update-rate or increase --top-k."
                ),
            ))
        elif global_max > MOE_MAX_LOAD_WARN:
            issues.append(_Issue(
                key=key, severe=False,
                message=(
                    f"MoE expert load concentration: max load fraction = "
                    f"{global_max:.3f} > {MOE_MAX_LOAD_WARN}. Monitor for collapse."
                ),
            ))
        else:
            self._issued.discard(key)

        return issues

    def _check_skip_rate(self, nonfinite_skips: int, step: int) -> List[_Issue]:
        issues: List[_Issue] = []
        if step < 10:
            return issues

        rate = nonfinite_skips / max(step, 1)
        key = "skip_rate"

        if rate > SKIP_RATE_ALERT:
            issues.append(_Issue(
                key=key, severe=True,
                message=(
                    f"Non-finite skip rate = {rate:.2%} ({nonfinite_skips}/{step}). "
                    f"Training is likely diverging — check data, LR, and gradient clip."
                ),
            ))
        elif rate > SKIP_RATE_WARN:
            issues.append(_Issue(
                key=key, severe=False,
                message=(
                    f"Non-finite skip rate = {rate:.2%} ({nonfinite_skips}/{step}). "
                    f"Occasional skips from flex_attention fallback are normal, "
                    f"but >1% suggests instability."
                ),
            ))
        else:
            self._issued.discard(key)

        return issues

    def _check_plateau(
        self, loss_val: float, step: int, total_steps: int,
    ) -> List[_Issue]:
        issues: List[_Issue] = []

        if not math.isfinite(loss_val):
            return issues

        self._loss_history.append(loss_val)
        # Keep a bounded window.
        max_window = PLATEAU_PATIENCE + 5
        if len(self._loss_history) > max_window:
            self._loss_history = self._loss_history[-max_window:]

        if len(self._loss_history) < PLATEAU_PATIENCE:
            return issues

        # Only flag plateau if we're well past warmup.
        if step < 500:
            self._loss_history.clear()
            return issues

        recent = self._loss_history[-PLATEAU_PATIENCE:]
        oldest = recent[0]
        newest = recent[-1]
        improvement = oldest - newest

        key = "ppl_plateau"
        if improvement < PLATEAU_MIN_IMPROVEMENT and newest > 0.5:
            # Cooldown: don't re-alert within PLATEAU_COOLDOWN_STEPS.
            if step - self._last_plateau_step >= PLATEAU_COOLDOWN_STEPS:
                ppl = math.exp(min(newest, 20))
                # Build a hint about what's NOT wrong, to narrow down causes.
                hints: list = []
                hints.append("Check --diagnostic-every output above for root cause")
                hints.append("try: --body-gate-init-std 0.02, reduce --num-loops, raise --bias-update-rate, or increase d_model")
                issues.append(_Issue(
                    key=key, severe=True,
                    message=(
                        f"Loss plateau detected: no improvement in "
                        f"{PLATEAU_PATIENCE} intervals "
                        f"(loss {oldest:.3f} → {newest:.3f}, PPL ≈ {ppl:.1f}). "
                        + " ".join(hints)
                    ),
                ))
                self._last_plateau_step = step
            # Always clear history after a plateau is detected — the
            # next PLATEAU_PATIENCE intervals will form a fresh window.
            self._loss_history.clear()
        else:
            self._issued.discard(key)

        return issues

    def _check_opt_state(self, optimizer, step: int) -> List[_Issue]:
        """Brief optimizer state health checks."""
        issues: List[_Issue] = []

        # MultiOptimizer (train.py) wraps several optimizers; iterate them.
        inner_opts: list = getattr(optimizer, "optimizers", [optimizer])

        for opt in inner_opts:
            for g in opt.param_groups:
                lr = g.get("lr", 0.0)
                initial_lr = g.get("initial_lr", lr)
                # lr=0 during warmup is normal (scheduler hasn't ramped yet);
                # only warn if initial_lr itself is zero or negative.
                if initial_lr <= 0:
                    name = g.get("name", "?")
                    issues.append(_Issue(
                        key=f"opt_lr_zero_{name}", severe=True,
                        message=(
                            f"Optimizer group '{name}' has initial_lr={initial_lr} — "
                            f"this param group will never update."
                        ),
                    ))
                # Warn if AdamW beta2 (exp_avg_sq) is blowing up
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    state = opt.state.get(p)
                    if state is None:
                        continue
                    exp_avg_sq = state.get("exp_avg_sq")
                    if exp_avg_sq is not None and isinstance(exp_avg_sq, torch.Tensor):
                        esq_max = exp_avg_sq.max().item()
                        if esq_max > 100.0:
                            name = g.get("name", "?")
                            issues.append(_Issue(
                                key=f"opt_esq_high_{name}", severe=False,
                                message=(
                                    f"AdamW exp_avg_sq max = {esq_max:.1f} in "
                                    f"group '{name}' — large second moment suggests "
                                    f"gradient spikes. Consider reducing LR or "
                                    f"increasing grad_clip."
                                ),
                            ))
                            break  # one warning per group is enough
        return issues

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# One-shot summary (used at the start of training to confirm model structure)
# ---------------------------------------------------------------------------

def print_model_diagnostic_summary(model: "nn.Module", config: Any) -> None:
    """Print a one-time summary of model structure and initialisation state,
    useful for verifying that novel components (proj, out_up, A/B gates) are
    initialised as expected."""

    lines: List[str] = [_bold("[diag] Model diagnostic summary:")]

    # Model type
    model_type = type(model).__name__
    lines.append(f"  Model: {model_type}")

    # Count components
    n_proj = 0
    n_out_up = 0
    n_kda = 0
    n_moe = 0
    n_swa = 0
    has_ab = False
    for mod in model.modules():
        t = type(mod).__name__
        if hasattr(mod, "proj") and hasattr(mod, "norm"):
            n_proj += 1
        if hasattr(mod, "out_up"):
            n_out_up += 1
        if hasattr(mod, "A_log"):
            n_kda += 1
        if t == "MoELayer":
            n_moe += 1
        if t == "LocalAttention":
            n_swa += 1
    if hasattr(model, "body") and hasattr(model.body, "A"):
        has_ab = True

    lines.append(f"  BlockAttentionResidual modules: {n_proj}")
    lines.append(f"  SnapshotRetrieval modules:      {n_out_up}")
    lines.append(f"  KDA attention layers:           {n_kda}")
    lines.append(f"  MoE layers:                     {n_moe}")
    lines.append(f"  SWA (local attention) layers:   {n_swa}")
    lines.append(f"  Recursive A/B gates present:    {has_ab}")

    # Looped config
    if hasattr(config, "num_loops"):
        lines.append(f"  num_loops: {config.num_loops}")
        lines.append(f"  num_entry_layers: {getattr(config, 'num_entry_layers', '?')}")
        lines.append(f"  num_body_layers: {getattr(config, 'num_body_layers', '?')}")
        lines.append(f"  num_exit_layers: {getattr(config, 'num_exit_layers', '?')}")

    # Init state of key components
    for mod in model.modules():
        if hasattr(mod, "proj") and hasattr(mod, "norm"):
            w = mod.proj
            if isinstance(w, torch.Tensor):
                lines.append(
                    f"  BlockAttentionResidual.proj init: "
                    f"mean={w.float().mean().item():.2e} "
                    f"max_abs={w.float().abs().max().item():.2e}"
                )
            break
    for mod in model.modules():
        if hasattr(mod, "out_up"):
            w = mod.out_up.weight
            if isinstance(w, torch.Tensor):
                lines.append(
                    f"  SnapshotRetrieval.out_up init: "
                    f"mean={w.float().mean().item():.2e} "
                    f"std={w.float().std().item():.2e}"
                )
            break
    if has_ab:
        A = model.body.A
        B = model.body.B
        if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
            lines.append(
                f"  Body A gate init: "
                f"mean={A.float().mean().item():.2e} "
                f"max_abs={A.float().abs().max().item():.2e}"
            )
            lines.append(
                f"  Body B gate init: "
                f"mean={B.float().mean().item():.2e} "
                f"max_abs={B.float().abs().max().item():.2e}"
            )

    for line in lines:
        print(line)
