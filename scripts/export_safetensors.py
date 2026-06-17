#!/usr/bin/env python
"""Export a clean, inference-ready ``.safetensors`` file from a training checkpoint.

Training checkpoints (``checkpoint_epoch_*.pt`` / ``checkpoint_best.pt`` /
``checkpoint_final.pt`` written by ``scripts/train.py``) bundle the model
weights together with optimizer, scheduler, Muon hyper-parameter, EMA, and
streaming-resume state (~8GB). For inference or release you only want the model
weights. This tool strips everything else and writes ``safetensors``.

Two details make a naive ``safetensors.save_file`` fail or mislead, both handled
here:

* **Tied weights.** Every model ties ``lm_head.weight = token_emb.weight`` (one
  shared storage under two keys). ``safetensors`` refuses tensors that share
  storage, so we clone to give each key independent, contiguous memory.
* **Wrapper prefixes.** The final checkpoint stores raw-model keys, but periodic
  checkpoints or hand-made ones may carry ``_orig_mod.`` (torch.compile) or
  ``module.`` (DDP) prefixes. We strip them defensively.

Usage::

    uv run python scripts/export_safetensors.py runs/.../checkpoint_final.pt
    uv run python scripts/export_safetensors.py ckpt.pt -o model.safetensors --dtype bfloat16
    uv run python scripts/export_safetensors.py ckpt.pt --ema   # export EMA weights
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import torch
from safetensors.torch import save_file

_DTYPES = {
    "keep": None,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Wrapper prefixes that may leak into a state dict, longest first.
_PREFIXES = ("_orig_mod.", "module.")


def _strip_prefix(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in _PREFIXES:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def _select_state_dict(ckpt: object, use_ema: bool) -> Dict[str, torch.Tensor]:
    """Pull the model (or EMA) weights out of a loaded checkpoint object."""
    if use_ema:
        if not isinstance(ckpt, dict) or "ema_state_dict" not in ckpt:
            raise SystemExit(
                "--ema requested but the checkpoint has no 'ema_state_dict' "
                "(the run was trained with --ema-decay 0)."
            )
        raw = ckpt["ema_state_dict"]
        # AveragedModel stores weights under 'module.' plus an 'n_averaged' buffer.
        return {
            k: v for k, v in raw.items()
            if k != "n_averaged" and isinstance(v, torch.Tensor)
        }

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    # Allow exporting a bare state dict (already just weights).
    if isinstance(ckpt, dict) and all(
        isinstance(v, torch.Tensor) for v in ckpt.values()
    ) and ckpt:
        return ckpt
    raise SystemExit(
        "Checkpoint has no 'model_state_dict' and is not a bare tensor dict; "
        "cannot locate model weights."
    )


def _find_tied_groups(sd: Dict[str, torch.Tensor]) -> List[List[str]]:
    """Group keys that share the same underlying storage (tied weights)."""
    by_storage: Dict[int, List[str]] = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor) or v.numel() == 0:
            continue
        ptr = v.untyped_storage().data_ptr()
        by_storage.setdefault(ptr, []).append(k)
    return [sorted(keys) for keys in by_storage.values() if len(keys) > 1]


def export(
    checkpoint: Path,
    output: Path,
    *,
    use_ema: bool = False,
    dtype: str = "keep",
    copy_config: bool = True,
) -> None:
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    print(f"Loading checkpoint: {checkpoint}")
    # weights_only=False: our own trusted file containing optimizer/Muon python
    # objects that the (torch>=2.6) safe unpickler would otherwise reject.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    raw_sd = _select_state_dict(ckpt, use_ema)
    cast = _DTYPES[dtype]

    tied = _find_tied_groups(raw_sd)
    if tied:
        pretty = "; ".join("=".join(g) for g in tied)
        print(f"Tied weights detected (cloning to de-duplicate storage): {pretty}")

    tensors: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    total_params = 0
    for key, val in raw_sd.items():
        clean = _strip_prefix(key)
        if not isinstance(val, torch.Tensor):
            skipped.append(key)
            continue
        t = val.detach().to("cpu")
        if cast is not None and t.is_floating_point():
            t = t.to(cast)
        # .contiguous().clone() guarantees independent, non-shared storage so
        # safetensors accepts tied tensors and the file is self-contained.
        tensors[clean] = t.contiguous().clone()
        total_params += t.numel()

    if skipped:
        print(f"Skipped {len(skipped)} non-tensor entr(ies): {skipped[:8]}"
              + (" ..." if len(skipped) > 8 else ""))

    dtypes = sorted({str(t.dtype).replace("torch.", "") for t in tensors.values()})

    metadata: Dict[str, str] = {
        "format": "pt",
        "producer": "logos/scripts/export_safetensors.py",
        "source_checkpoint": checkpoint.name,
        "weights": "ema" if use_ema else "model",
        "num_tensors": str(len(tensors)),
        "total_params": str(total_params),
        "dtypes": ",".join(dtypes),
    }
    if isinstance(ckpt, dict):
        if ckpt.get("epoch") is not None:
            metadata["step"] = str(ckpt["epoch"])
        metrics = ckpt.get("metrics")
        if isinstance(metrics, dict):
            for mk, mv in metrics.items():
                if isinstance(mv, (int, float, str, bool)):
                    metadata[f"metric_{mk}"] = str(mv)
    if tied:
        metadata["tied_weights"] = "; ".join("=".join(g) for g in tied)

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(tensors)} tensors ({total_params:,} params, "
          f"dtype={'/'.join(dtypes)}) -> {output}")
    save_file(tensors, str(output), metadata=metadata)

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Wrote {output} ({size_mb:.1f} MiB)")

    if copy_config:
        src_config = checkpoint.parent / "config.json"
        dst_config = output.parent / "config.json"
        if src_config.is_file() and src_config.resolve() != dst_config.resolve():
            shutil.copyfile(src_config, dst_config)
            print(f"Copied model config -> {dst_config}")
        elif not src_config.is_file():
            print("No sibling config.json found next to the checkpoint "
                  "(skipping config copy).")

    # Emit a small provenance sidecar so the export is self-describing.
    sidecar = output.with_suffix(output.suffix + ".json")
    with open(sidecar, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote export metadata -> {sidecar}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export model weights from a training checkpoint to safetensors.",
    )
    p.add_argument("checkpoint", type=Path,
                   help="Path to a checkpoint_*.pt produced by scripts/train.py")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output .safetensors path "
                        "(default: <checkpoint_stem>.safetensors next to the input)")
    p.add_argument("--ema", action="store_true",
                   help="Export EMA-averaged weights instead of the live model "
                        "(requires a run trained with --ema-decay > 0).")
    p.add_argument("--dtype", choices=sorted(_DTYPES), default="keep",
                   help="Cast floating-point tensors on export (default: keep "
                        "original, typically bfloat16).")
    p.add_argument("--no-config-copy", action="store_true",
                   help="Do not copy the sibling config.json next to the output.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    output = args.output or args.checkpoint.with_suffix(".safetensors")
    export(
        args.checkpoint,
        output,
        use_ema=args.ema,
        dtype=args.dtype,
        copy_config=not args.no_config_copy,
    )


if __name__ == "__main__":
    main()
