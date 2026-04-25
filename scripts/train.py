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
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
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


MODEL_REGISTRY: Dict[str, tuple] = {
    "baseline": (BaselineConfig, BaselineTransformer),
    "linear": (LinearConfig, LinearTransformer),
    "recursive": (RecursiveConfig, RecursiveTransformer),
    "residual": (ResidualConfig, ResidualTransformer),
    "superlinear": (SuperLinearConfig, SuperLinearTransformer),
    "hybrid": (HybridConfig, HybridTransformer),
    "logos": (LogosConfig, LogosTransformer),
}


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
) -> DataLoader:

    class _Dataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset):
            self.data = hf_dataset

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    def collate_fn(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }

    return DataLoader(
        _Dataset(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


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
) -> Dict[str, float]:
    model.train(is_train)

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        dataloader,
        desc=f"{'Train' if is_train else 'Valid'} Epoch {epoch}",
        leave=False,
    )

    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        # Eval needs no_grad so activations don't pile up across iterations.
        grad_ctx = contextlib.nullcontext() if is_train else torch.no_grad()
        with grad_ctx, torch.autocast(device_type=device.type, dtype=mp_dtype, enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
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

            # Without this call, MoE bias stays at zero init forever and
            # aux-loss-free balancing never activates.
            topk_indices = outputs.get("topk_indices")
            if topk_indices is not None:
                model.update_router_biases(topk_indices)

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        num_batches += batch_size

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "ppl": f"{math.exp(min(loss.item(), 20)):.2f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    pbar.close()

    if num_batches == 0:
        return {"loss": float("inf"), "ppl": float("inf")}

    avg_loss = total_loss / num_batches
    return {"loss": avg_loss, "ppl": math.exp(min(avg_loss, 20))}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    save_dir: Path,
    is_best: bool = False,
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

    path = save_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(checkpoint, path)

    if is_best:
        torch.save(checkpoint, save_dir / "checkpoint_best.pt")

    config_path = save_dir / "config.json"
    if not config_path.exists():
        with open(config_path, "w") as f:
            json.dump(vars(inner.config), f, indent=2, default=str)

    return path


def main():
    parser = argparse.ArgumentParser(
        description="Train a decoder-only transformer with causal next-token prediction"
    )

    parser.add_argument("--model", type=str, default="baseline",
                        choices=sorted(MODEL_REGISTRY.keys()))

    parser.add_argument("--dataset", type=str, default="tiny_shakespeare",
                        help="HuggingFace dataset name OR local file path "
                             "(.json, .csv, .txt).")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Sequence block size")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
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
    parser.add_argument("--moe-diversity-factor", type=float, default=0.0,
                        help="Cross-loop expert-diversity weight in the MoE "
                             "bias update. Relevant for ``recursive`` and "
                             "``logos`` (MoE weights reused across loop "
                             "iterations). 0 = standard balance only. Try "
                             "0.5–1.0 to encourage different expert "
                             "selections per loop step.")
    parser.add_argument("--bias-update-rate", type=float, default=0.01,
                        help="DeepSeek-style router bias update rate")

    parser.add_argument("--head-dim", type=int, default=None,
                        help="[linear+] Per-head q/k/v dim. Defaults to "
                             "d_model // num_heads.")
    parser.add_argument("--conv-size", type=int, default=4,
                        help="[linear+] Short causal conv1d kernel size")
    parser.add_argument("--chunk-size", type=int, default=64,
                        help="[linear+] Chunk size for the KDA scan")

    parser.add_argument("--num-entry-layers", type=int, default=2,
                        help="[recursive,logos] Standard blocks run once before the body loop")
    parser.add_argument("--num-body-layers", type=int, default=4,
                        help="[recursive,logos] Shared blocks applied on every loop step")
    parser.add_argument("--num-exit-layers", type=int, default=2,
                        help="[recursive,logos] Standard blocks run once after the body loop")
    parser.add_argument("--num-loops", type=int, default=4,
                        help="[recursive,logos] Number of times the body is applied")
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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Default: checkpoints/{model}")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--no-save", action="store_true",
                        help="Skip model-weight checkpoints; still write history.json")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--compile", action="store_true",
                        help="Wrap model with torch.compile")
    parser.add_argument("--compile-mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--bf16", action="store_true",
                        help="Enable bfloat16 mixed precision (Ampere+)")

    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--sample-prompt", type=str, default="Once upon a time")
    parser.add_argument("--sample-max-tokens", type=int, default=50)
    parser.add_argument("--sample-temperature", type=float, default=0.8)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device} | model: {args.model}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    mp_dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32
    use_amp = (mp_dtype == torch.bfloat16)
    if use_amp:
        print(f"  Mixed precision: {mp_dtype}")

    if args.save_dir is None:
        args.save_dir = f"checkpoints/{args.model}"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tiktoken encoding: {args.tiktoken_encoding}")
    tokenizer = TiktokenTokenizer(args.tiktoken_encoding)
    print(f"  vocab_size: {tokenizer.vocab_size}")

    dataset_path = Path(args.dataset)
    if dataset_path.exists():
        print(f"Loading local dataset: {args.dataset}")
        suffix = dataset_path.suffix.lower()
        if suffix == ".json":
            dataset = load_dataset("json", data_files=str(dataset_path))
        elif suffix == ".csv":
            dataset = load_dataset("csv", data_files=str(dataset_path))
        elif suffix == ".txt":
            dataset = load_dataset("text", data_files=str(dataset_path))
        else:
            raise ValueError(f"Unsupported local file format: {suffix}")
    else:
        print(f"Loading HuggingFace dataset: {args.dataset}")
        dataset = load_dataset(args.dataset, args.dataset_config)

    print(f"  splits: {list(dataset.keys())}")

    if "validation" not in dataset and "valid" not in dataset:
        print("  No validation split found – creating 10% hold-out from train")
        split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=args.seed)
        dataset = DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"],
        })

    if args.max_train_examples is not None:
        n = min(args.max_train_examples, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(n))
        print(f"  Subset train to {n} examples")
    if args.max_val_examples is not None:
        n = min(args.max_val_examples, len(dataset["validation"]))
        dataset["validation"] = dataset["validation"].select(range(n))
        print(f"  Subset validation to {n} examples")

    print(f"Preprocessing with block_size={args.max_length} ...")
    dataset = preprocess_dataset(
        dataset, tokenizer, args.max_length, text_column=args.text_column
    )
    print(f"  train examples:    {len(dataset['train'])}")
    print(f"  validation examples: {len(dataset['validation'])}")

    train_loader = create_dataloader(
        dataset["train"], args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = create_dataloader(
        dataset["validation"], args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    config, model = build_model(args, vocab_size=tokenizer.vocab_size)
    model = model.to(device)
    print(f"\nModel: {type(model).__name__}")
    print(f"  parameters: {count_parameters(model):,}")
    print(f"  use_moe:    {config.use_moe}")
    if config.use_moe:
        print(f"  shared experts: {config.num_shared_experts}")
        print(f"  sparse experts: {config.num_sparse_experts}")
        print(f"  top_k:          {config.top_k}")

    fused_adamw = (device.type == "cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        fused=fused_adamw,
    )
    if fused_adamw:
        print("  Using fused AdamW")

    total_steps = len(train_loader) * args.epochs
    if args.warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-10, end_factor=1.0, total_iters=args.warmup_steps
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=total_steps - args.warmup_steps, eta_min=args.lr * 0.1
                ),
            ],
            milestones=[args.warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.lr * 0.1
        )

    raw_model = model

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

    if args.compile:
        print(f"  Compiling model with torch.compile (mode={args.compile_mode})...")
        model = torch.compile(model, mode=args.compile_mode)

    print("\n" + "=" * 60)
    print("Starting training")
    print("=" * 60)

    best_val_loss = float("inf")
    history: list[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = run_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_clip=args.grad_clip, is_train=True, epoch=epoch,
            use_amp=use_amp, mp_dtype=mp_dtype,
        )

        val_metrics = None
        val_loader_nonempty = len(val_loader.dataset) > 0
        if epoch % args.eval_every == 0 and val_loader_nonempty:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            val_metrics = run_epoch(
                model, val_loader, optimizer, None, device,
                grad_clip=0.0, is_train=False, epoch=epoch,
                use_amp=use_amp, mp_dtype=mp_dtype,
            )

        epoch_time = time.time() - epoch_start

        log_line = (
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"time: {epoch_time:.1f}s | "
            f"train_loss: {train_metrics['loss']:.4f} (ppl {train_metrics['ppl']:.2f})"
        )
        if val_metrics and val_metrics["loss"] != float("inf"):
            log_line += (
                f" | val_loss: {val_metrics['loss']:.4f} (ppl {val_metrics['ppl']:.2f})"
            )
        log_line += f" | lr: {optimizer.param_groups[0]['lr']:.2e}"
        print(log_line)

        if config.use_moe:
            balance = model.get_balance_stats()
            if balance:
                bias_max = max(v for k, v in balance.items() if "bias_max" in k)
                bias_mean = sum(v for k, v in balance.items() if "bias_mean" in k) / len(
                    [k for k in balance if "bias_mean" in k]
                )
                print(f"  MoE balance -> bias_mean: {bias_mean:.4f}, bias_max: {bias_max:.4f}")

        if args.sample_every > 0 and epoch % args.sample_every == 0:
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

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "time": epoch_time,
        })

        if not args.no_save and epoch % args.save_every == 0:
            is_best = False
            if val_metrics and val_metrics["loss"] != float("inf"):
                is_best = val_metrics["loss"] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics["loss"]
            ckpt_metrics = val_metrics if val_metrics else train_metrics
            path = save_checkpoint(
                model, optimizer, scheduler, epoch,
                ckpt_metrics, save_dir, is_best=is_best,
            )
            print(f"  -> checkpoint saved to {path}")
            if is_best:
                print(f"  -> new best val_loss: {best_val_loss:.4f}")
        elif val_metrics and val_metrics["loss"] != float("inf"):
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]

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
                "metrics": val_metrics or train_metrics,
                "history": history,
            },
            final_path,
        )
        print(f"Final checkpoint saved to {final_path}")
    else:
        print("(--no-save active: no model weights written)")
    print("Training complete!")


if __name__ == "__main__":
    main()
