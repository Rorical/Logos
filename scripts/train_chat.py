#!/usr/bin/env python
"""ChatML chat fine-tuning driver. Loads HuggingFace datasets with a
``messages`` column, applies ChatML formatting, and masks all but the
assistant tokens from the loss."""

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
from utils.tokenizer import TiktokenTokenizer


# =============================================================================
# Model registry
# =============================================================================

MODEL_REGISTRY: Dict[str, tuple] = {
    "baseline": (BaselineConfig, BaselineTransformer),
    "linear": (LinearConfig, LinearTransformer),
    "recursive": (RecursiveConfig, RecursiveTransformer),
}


def build_model(args: argparse.Namespace, vocab_size: int):
    """Instantiate the config + model for ``args.model``.

    Only fields defined on the selected config dataclass are forwarded, so
    flags that are irrelevant to a particular model are silently ignored.
    """
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
        capacity_factor=args.capacity_factor,
        block_residual_isolate_softmax=getattr(
            args, "block_residual_isolate_softmax", False,
        ),
    )

    if args.model == "linear":
        head_dim = args.head_dim or (args.d_model // args.num_heads)
        kwargs.update(
            head_dim=head_dim,
            conv_size=args.conv_size,
            chunk_size=args.chunk_size,
        )
    elif args.model == "recursive":
        kwargs.update(
            num_entry_layers=args.num_entry_layers,
            num_body_layers=args.num_body_layers,
            num_exit_layers=args.num_exit_layers,
            num_loops=args.num_loops,
        )

    config = ConfigCls(**kwargs)
    model = ModelCls(config)
    return config, model


# =============================================================================
# Data helpers
# =============================================================================

def preprocess_chat_dataset(
    dataset: DatasetDict,
    tokenizer: TiktokenTokenizer,
    max_length: int,
    cache_dir: Path,
) -> DatasetDict:
    """
    Tokenise every conversation with ChatML formatting.
    Only assistant content is included in labels.
    """

    def tokenize_example(example: Dict[str, Any]) -> Dict[str, list]:
        messages = example.get("messages", [])
        if not messages:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        encoded = tokenizer.encode_chat(messages, max_length=max_length)
        return {
            "input_ids": encoded["input_ids"].tolist(),
            "attention_mask": encoded["attention_mask"].tolist(),
            "labels": encoded["labels"].tolist(),
        }

    tokenized = dataset.map(
        tokenize_example,
        remove_columns=dataset[list(dataset.keys())[0]].column_names,
        desc="Tokenising chat",
    )

    # Filter out empty or all-masked examples (would cause NaN in cross-entropy)
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0, desc="Filtering empty")
    tokenized = tokenized.filter(
        lambda x: any(l != -100 for l in x["labels"]),
        desc="Filtering all-masked",
    )
    return tokenized


def collate_fn(batch):
    """Dynamic padding to the longest sequence in the batch."""
    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids = []
    attention_masks = []
    labels = []
    pad_token_id = batch[0].get("pad_token_id", 100257)  # cl100k_base eos

    for b in batch:
        ids = b["input_ids"]
        lbls = b["labels"]
        pad_len = max_len - len(ids)

        if pad_len > 0:
            input_ids.append(ids + [pad_token_id] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)
            labels.append(lbls + [-100] * pad_len)
        else:
            input_ids.append(ids)
            attention_masks.append([1] * len(ids))
            labels.append(lbls)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class _Dataset(torch.utils.data.Dataset):
    """Simple wrapper for a HuggingFace dataset with pre-tokenized columns.
    Defined at module level so Windows spawn multiprocessing can pickle it."""
    def __init__(self, hf_dataset, pad_token_id: int):
        self.data = hf_dataset
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
            "pad_token_id": self.pad_token_id,
        }


def create_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pad_token_id: int = 100257,
):
    return DataLoader(
        _Dataset(dataset, pad_token_id=pad_token_id),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# =============================================================================
# Training helpers
# =============================================================================

def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    grad_clip: float,
    grad_accum: int = 1,
    is_train: bool = True,
    epoch: int = 0,
    use_amp: bool = False,
    mp_dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    model.train(is_train)

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    if is_train:
        optimizer.zero_grad()

    pbar = tqdm(
        dataloader,
        desc=f"{'Train' if is_train else 'Valid'} Epoch {epoch}",
        leave=False,
    )

    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # Only build the autograd graph when we actually need it — otherwise
        # activations pile up until the next iteration replaces `outputs`,
        # doubling peak memory and OOM-ing large models during validation.
        grad_ctx = contextlib.nullcontext() if is_train else torch.no_grad()
        with grad_ctx, torch.autocast(device_type=device.type, dtype=mp_dtype, enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                is_causal=True,
            )

        loss = outputs["loss"]

        if is_train and loss is not None:
            # Scale loss for gradient accumulation
            scaled_loss = loss / grad_accum
            scaled_loss.backward()

            # DeepSeek-style aux-loss-free router-bias update. Fired every
            # micro-batch (not only at the grad-accum boundary) so all
            # routing decisions in the accumulation cycle contribute to
            # load balancing. This is a small in-place update on a
            # non-learnable Parameter, independent of the optimiser step.
            topk_indices = outputs.get("topk_indices")
            if topk_indices is not None:
                model.update_router_biases(topk_indices)

            # Only update weights after accumulating enough gradients
            if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        total_loss += loss.item() * batch_size
        total_tokens += batch_size * seq_len
        num_batches += batch_size

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "ppl": f"{math.exp(min(loss.item(), 20)):.2f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
        )

    pbar.close()

    if num_batches == 0:
        return {"loss": float("inf"), "ppl": float("inf")}

    avg_loss = total_loss / num_batches
    avg_ppl = math.exp(min(avg_loss, 20))
    return {"loss": avg_loss, "ppl": avg_ppl}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    save_dir: Path,
    is_best: bool = False,
):
    """Save model state + training metadata.

    If ``model`` is a ``torch.compile`` wrapper, its state-dict keys would be
    prefixed with ``_orig_mod.`` and would fail to load into a fresh,
    uncompiled instance. Unwrap once so checkpoints stay portable.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
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
        best_path = save_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)

    config_path = save_dir / "config.json"
    if not config_path.exists():
        with open(config_path, "w") as f:
            json.dump(vars(inner.config), f, indent=2, default=str)

    return path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a decoder-only transformer on ChatML chat data"
    )

    # --- Model selection -----------------------------------------------------
    parser.add_argument("--model", type=str, default="baseline",
                        choices=sorted(MODEL_REGISTRY.keys()),
                        help="Which architecture to train")

    # --- Data ----------------------------------------------------------------
    parser.add_argument("--dataset", type=str, required=True,
                        help="HuggingFace dataset name with 'messages' column")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max sequence length (truncate from right)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers (0 = main process)")
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=None)

    # --- Model ---------------------------------------------------------------
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1364)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--use-moe", action="store_true")
    parser.add_argument("--num-shared-experts", type=int, default=2)
    parser.add_argument("--num-sparse-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--expert-d-ff", type=int, default=256)
    parser.add_argument("--bias-update-rate", type=float, default=0.01)
    parser.add_argument("--capacity-factor", type=float, default=2.0,
                        help="Capacity factor for static-shape MoE dispatch (>=1.0)")
    parser.add_argument("--block-residual-isolate-softmax", action="store_true",
                        help="Route the BlockAttentionResidual depth-softmax "
                             "+ weighted-sum through an opaque custom_op so "
                             "torch.compile doesn't fuse softmax_backward "
                             "with the upstream stack/RMSNorm chain. Set on "
                             "Ada-class GPUs hitting Inductor SMEM caps.")

    # --- KDA / linear-model-specific ----------------------------------------
    parser.add_argument("--head-dim", type=int, default=None,
                        help="[linear] Per-head dimension for q/k/v. "
                             "Defaults to d_model // num_heads.")
    parser.add_argument("--conv-size", type=int, default=4,
                        help="[linear] Short causal conv1d kernel size")
    parser.add_argument("--chunk-size", type=int, default=64,
                        help="[linear] Chunk size for the KDA scan")

    parser.add_argument("--num-entry-layers", type=int, default=2,
                        help="[recursive] Blocks run once before the body loop")
    parser.add_argument("--num-body-layers", type=int, default=4,
                        help="[recursive] Shared blocks applied on every loop step")
    parser.add_argument("--num-exit-layers", type=int, default=2,
                        help="[recursive] Blocks run once after the body loop")
    parser.add_argument("--num-loops", type=int, default=4,
                        help="[recursive] Number of times the body is applied")

    # --- Tokenizer -----------------------------------------------------------
    parser.add_argument("--tiktoken-encoding", type=str, default="cl100k_base")

    # --- Optimiser -----------------------------------------------------------
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # --- Sampling ------------------------------------------------------------
    parser.add_argument("--sample-every", type=int, default=1,
                        help="Generate sample every N epochs (0 = disabled)")
    parser.add_argument("--sample-prompt", type=str,
                        default="Explain quantum computing in simple terms.")
    parser.add_argument("--sample-max-tokens", type=int, default=100)
    parser.add_argument("--sample-temperature", type=float, default=0.8)

    # --- Infrastructure ------------------------------------------------------
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save checkpoints "
                             "(default: checkpoints/{model}_chat)")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--compile", action="store_true",
                        help="Wrap model with torch.compile for static-graph execution "
                             "(1.5-2x speedup after warmup)")
    parser.add_argument("--compile-mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode. 'reduce-overhead' uses CUDA graphs; "
                             "'max-autotune' searches for the fastest kernel variants "
                             "(slowest first step, fastest steady state)")
    parser.add_argument("--bf16", action="store_true",
                        help="Enable bf16 mixed precision (recommended for Ampere+)")

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    torch.manual_seed(args.seed)
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device} | model: {args.model}")

    # Enable TF32 on Ampere+ for faster float32 matmul (free speedup)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # Mixed precision dtype
    mp_dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32
    if args.bf16:
        print(f"  Mixed precision: {mp_dtype}")
    use_amp = (mp_dtype == torch.bfloat16)

    if args.save_dir is None:
        args.save_dir = f"checkpoints/{args.model}_chat"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Tokenizer
    # ------------------------------------------------------------------ #
    print(f"Loading tiktoken encoding: {args.tiktoken_encoding}")
    tokenizer = TiktokenTokenizer(args.tiktoken_encoding)
    print(f"  vocab_size: {tokenizer.vocab_size}")

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        cache_dir=str(cache_dir),
    )
    print(f"  splits: {list(dataset.keys())}")

    if "validation" not in dataset and "valid" not in dataset:
        print("  No validation split found - creating 10% hold-out from train")
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

    print(f"Preprocessing with ChatML, max_length={args.max_length} ...")
    dataset = preprocess_chat_dataset(dataset, tokenizer, args.max_length, cache_dir)
    print(f"  train examples:    {len(dataset['train'])}")
    print(f"  validation examples: {len(dataset['validation'])}")

    train_loader = create_dataloader(
        dataset["train"], args.batch_size, shuffle=True,
        num_workers=args.num_workers, pad_token_id=tokenizer.pad_token_id,
    )
    val_loader = create_dataloader(
        dataset["validation"], args.batch_size, shuffle=False,
        num_workers=args.num_workers, pad_token_id=tokenizer.pad_token_id,
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    config, model = build_model(args, vocab_size=tokenizer.vocab_size)
    model = model.to(device)
    print(f"\nModel: {type(model).__name__}")
    print(f"  parameters: {count_parameters(model):,}")
    print(f"  use_moe:    {config.use_moe}")
    if config.use_moe:
        print(f"  shared experts: {config.num_shared_experts}")
        print(f"  sparse experts: {config.num_sparse_experts}")
        print(f"  top_k:          {config.top_k}")

    # ------------------------------------------------------------------ #
    # Optimiser & scheduler
    # ------------------------------------------------------------------ #
    # Use fused AdamW when available (single kernel, faster)
    fused = (device.type == "cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        fused=fused,
    )
    if fused:
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

    # ------------------------------------------------------------------ #
    # Sampling helper (uses the uncompiled model so every generation step
    # doesn't pay a recompile cost for each new prefix length)
    # ------------------------------------------------------------------ #
    raw_model = model

    def sample_chat(prompt_text: str, max_new_tokens: int, temperature: float) -> str:
        raw_model.train(False)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ]
        prompt = tokenizer.apply_chat_template(messages)
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

    # ------------------------------------------------------------------ #
    # torch.compile — static graph + fused kernels via TorchInductor.
    # Attribute access (model.config / get_balance_stats / update_router_biases)
    # is transparently forwarded by OptimizedModule.
    # ------------------------------------------------------------------ #
    if args.compile:
        print(f"  Compiling model with torch.compile (mode={args.compile_mode})...")
        model = torch.compile(model, mode=args.compile_mode)
        print("  Compilation wrapped — first step will be slow while kernels warm up")

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Starting training")
    print("=" * 60)

    best_val_loss = float("inf")
    history: list[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = run_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_clip=args.grad_clip, grad_accum=args.grad_accum,
            is_train=True, epoch=epoch, use_amp=use_amp, mp_dtype=mp_dtype,
        )

        val_metrics = None
        val_loader_nonempty = len(val_loader.dataset) > 0
        if epoch % args.eval_every == 0 and val_loader_nonempty:
            # Reclaim the training graph's reserved blocks before the
            # eval compile path allocates its own.
            if device.type == "cuda":
                torch.cuda.empty_cache()
            val_metrics = run_epoch(
                model, val_loader, optimizer, None, device,
                grad_clip=0.0, grad_accum=1,
                is_train=False, epoch=epoch, use_amp=False, mp_dtype=mp_dtype,
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
            sample = sample_chat(
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

        if epoch % args.save_every == 0:
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

    # ------------------------------------------------------------------ #
    # Final save
    # ------------------------------------------------------------------ #
    final_path = save_dir / "checkpoint_final.pt"
    torch.save(
        {
            "epoch": args.epochs,
            # Save the underlying (uncompiled) weights so the checkpoint is
            # portable whether --compile was used or not.
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": val_metrics or train_metrics,
            "history": history,
        },
        final_path,
    )
    print(f"\nFinal checkpoint saved to {final_path}")
    print("Training complete!")


if __name__ == "__main__":
    main()
