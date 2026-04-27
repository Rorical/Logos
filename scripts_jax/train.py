"""Flax + Optax + Orbax training script for Logos on TPU v6e.

Single-file training loop with:
- Streaming HuggingFace datasets
- Muon (2D Linear weights) + AdamW (1D params, embeddings) via optax.contrib.muon
- WSD (warmup → stable → decay) schedule
- Orbax checkpointing
- jax.jit compiled train/eval steps
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models_jax import LogosConfig, LogosTransformer, count_parameters
from utils.tokenizer import TiktokenTokenizer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Logos pretraining (Jax/Flax)")
    p.add_argument("--model", type=str, default="logos")

    # Data
    p.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", type=str, default="sample-10BT")
    p.add_argument("--text-column", type=str, default="text")
    p.add_argument("--tiktoken-encoding", type=str, default="cl100k_base")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--total-tokens", type=str, default="10B")
    p.add_argument("--num-workers", type=int, default=4)

    # Architecture
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--d-ff", type=int, default=2730)
    p.add_argument("--num-entry-layers", type=int, default=2)
    p.add_argument("--num-body-layers", type=int, default=6)
    p.add_argument("--num-exit-layers", type=int, default=2)
    p.add_argument("--num-loops", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--norm-eps", type=float, default=1e-6)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--conv-size", type=int, default=4)
    p.add_argument("--snapshot-interval", type=int, default=512)
    p.add_argument("--snapshot-latent-dim", type=int, default=128)
    p.add_argument("--mem-top-k", type=int, default=16)
    p.add_argument("--mem-head-dim", type=int, default=64)
    p.add_argument("--swa-window", type=int, default=256)
    p.add_argument("--swa-every", type=int, default=4)
    p.add_argument("--swa-offset", type=int, default=3)

    # MoE
    p.add_argument("--use-moe", action="store_true", default=True)
    p.add_argument("--num-shared-experts", type=int, default=2)
    p.add_argument("--num-sparse-experts", type=int, default=32)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--expert-d-ff", type=int, default=768)
    p.add_argument("--moe-diversity-factor", type=float, default=0.5)
    p.add_argument("--bias-update-rate", type=float, default=0.01)

    # RoPE
    p.add_argument("--rope-base", type=float, default=10000.0)
    p.add_argument("--rope-scaling-type", type=str, default="none")
    p.add_argument("--rope-scaling-factor", type=float, default=1.0)

    # Optimizer — Muon + AdamW
    p.add_argument("--muon", action="store_true", default=True)
    p.add_argument("--lr", type=float, default=0.004)     # AdamW group (norms/biases/etc.)
    p.add_argument("--embed-lr", type=float, default=0.2)  # Embedding group
    p.add_argument("--muon-lr", type=float, default=0.02)  # Muon (2D Linear weights)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=3500)
    p.add_argument("--decay-steps", type=int, default=70000)
    p.add_argument("--scheduler", type=str, default="wsd")

    # Training loop
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=10000)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--keep-last-n", type=int, default=5)

    # Output
    p.add_argument("--save-dir", type=str, default="./checkpoints")
    p.add_argument("--seed", type=int, default=42)

    # LM head
    p.add_argument("--lm-head-chunk-size", type=int, default=4096)

    return p


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

class StreamingDataLoader:
    """Streaming dataset wrapper: yields (input_ids, labels) batches."""

    def __init__(
        self,
        dataset_name: str,
        config_name: str,
        text_column: str,
        tokenizer: TiktokenTokenizer,
        batch_size: int,
        max_length: int,
        seed: int = 42,
    ):
        self.ds = load_dataset(
            dataset_name, config_name, split="train", streaming=True,
        ).shuffle(seed=seed, buffer_size=10000)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.text_column = text_column
        self.seq_len = max_length + 1

    def __iter__(self):
        buffer = []
        for example in self.ds:
            ids = self.tokenizer.encode(
                example[self.text_column],
                allowed_special={"<|endoftext|>"},
            )
            buffer.extend(ids)
            while len(buffer) >= self.batch_size * self.seq_len:
                chunk = buffer[: self.batch_size * self.seq_len]
                buffer = buffer[self.batch_size * self.seq_len :]
                tokens = np.array(chunk, dtype=np.int32).reshape(
                    self.batch_size, self.seq_len
                )
                input_ids = tokens[:, :-1]
                labels = tokens[:, 1:]
                yield input_ids, labels


# ---------------------------------------------------------------------------
# Parameter grouping for Muon + AdamW
# ---------------------------------------------------------------------------

def split_param_groups_muon(params: Dict) -> Tuple[Dict, Dict, Dict]:
    """Split params into embedding, muon (2D), and adamw (1D) groups.

    Matches the PyTorch Muon split logic:
    - embedding: token_emb params, lm_head params (tied, so same lr)
    - muon: 2D linear kernel params (ndim == 2)
    - adamw: everything else (norms, biases, 1D params, conv kernels, etc.)
    """

    def _is_embedding(path, leaf):
        path_str = "/".join(str(k.key) if hasattr(k, 'key') else str(k) for k in path)
        return (
            "token_emb" in path_str
            or "lm_head/kernel" in path_str  # tied with token_emb
        )

    def _is_muon(path, leaf):
        """Muon applies to 2D matrix params (ndim == 2) that aren't embeddings."""
        if not isinstance(leaf, jnp.ndarray):
            return False
        if leaf.ndim != 2:
            return False
        path_str = "/".join(str(k.key) if hasattr(k, 'key') else str(k) for k in path)
        if "token_emb" in path_str:
            return False
        if "lm_head" in path_str and "kernel" in path_str:
            return False
        # All 2D linear kernel weights go to Muon
        return True

    muon_mask = jax.tree_util.tree_map_with_path(
        lambda p, l: _is_muon(p, l) and isinstance(l, jnp.ndarray), params
    )
    embed_mask = jax.tree_util.tree_map_with_path(
        lambda p, l: _is_embedding(p, l) and isinstance(l, jnp.ndarray), params
    )

    def _split(mask_fn):
        out = {}
        for path, leaf in jax.tree_util.tree_map_with_path(
            lambda p, l: (p, l) if mask_fn(p, l) and isinstance(l, jnp.ndarray) else None, params
        ).values():
            pass
        return out

    # Actually, optax.contrib.muon handles grouping internally via muon_weight_dimension_numbers
    # We just need to construct the dimension_specs
    # For muon: all 2D params
    muon_specs = {}
    embed_params = {}

    def _traverse(path, leaf, muon_path, embed_path):
        nonlocal muon_specs, embed_params
        if not isinstance(leaf, jnp.ndarray):
            return
        if leaf.ndim == 2 and not _is_embedding(path, leaf):
            muon_specs[muon_path] = optax.contrib.MuonDimensionNumbers()
            return leaf
        if _is_embedding(path, leaf):
            embed_params[embed_path] = leaf
            return None  # mark as non-muon
        return None  # non-muon -> AdamW

    # We'll use a simpler approach: manually construct the specs
    muon_dim_nums = jax.tree_util.tree_map_with_path(
        lambda p, leaf: (
            optax.contrib.MuonDimensionNumbers()
            if isinstance(leaf, jnp.ndarray) and leaf.ndim == 2
            and not _is_embedding(p, leaf)
            else None
        ),
        params,
    )

    return muon_dim_nums


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def create_wsd_schedule(
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
    peak_lr: float,
) -> optax.Schedule:
    """WSD: linear warmup → constant → linear decay."""
    stable_steps = total_steps - warmup_steps - decay_steps

    def schedule(step):
        step = step.astype(jnp.float32)
        warmup = peak_lr * (step / warmup_steps)
        stable = peak_lr
        decay_end = warmup_steps + stable_steps
        decay_frac = (step - warmup_steps - stable_steps) / decay_steps
        decay = peak_lr * (1.0 - jnp.clip(decay_frac, 0.0, 1.0))

        return jnp.where(
            step < warmup_steps,
            warmup,
            jnp.where(step < warmup_steps + stable_steps, stable, decay),
        )

    return schedule


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------

@dataclass
class TrainState:
    step: int
    params: Dict[str, Any]
    opt_state: Any
    variables: Dict[str, Any]
    key: jax.random.PRNGKey

    @property
    def model_vars(self):
        return {"params": self.params, **self.variables}

    def to_checkpoint(self):
        return {
            "params": self.params,
            "variables": self.variables,
            "step": self.step,
        }

    @classmethod
    def from_checkpoint(cls, ckpt, opt_state, key):
        return cls(
            step=ckpt.get("step", 0),
            params=ckpt["params"],
            opt_state=opt_state,
            variables=ckpt.get("variables", {}),
            key=key,
        )


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def compute_loss(
    params: Dict[str, Any],
    variables: Dict[str, Any],
    model: LogosTransformer,
    input_ids: jnp.ndarray,
    labels: jnp.ndarray,
    key: jax.random.PRNGKey,
    training: bool = True,
) -> Tuple[jnp.ndarray, Tuple[Any, ...]]:
    """CE loss + aux loss from MoE."""
    all_vars = {"params": params, **variables}
    mutable = list(variables.keys()) if training else []

    output = model.apply(
        all_vars,
        input_ids,
        labels=labels,
        deterministic=not training,
        mutable=mutable,
        rngs={"dropout": key} if training else {},
    )

    if mutable:
        outputs_dict, updated_vars = output
    else:
        outputs_dict = output
        updated_vars = {}

    loss = outputs_dict["loss"]
    aux = outputs_dict.get("aux_loss", jnp.zeros((), dtype=loss.dtype))
    total_loss = loss + aux

    return total_loss, (loss, aux, updated_vars)


# ---------------------------------------------------------------------------
# Train step (jit-compiled)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("model", "optimizer", "training"))
def train_step(
    state: TrainState,
    model: LogosTransformer,
    optimizer: optax.GradientTransformation,
    input_ids: jnp.ndarray,
    labels: jnp.ndarray,
    training: bool = True,
) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray]:
    """Single step: forward, backward, optimizer update."""
    key, subkey = jax.random.split(state.key)

    (total_loss, (ce_loss, aux_loss, updated_vars)), grads = jax.value_and_grad(
        compute_loss, argnums=0, has_aux=True,
    )(
        state.params, state.variables, model,
        input_ids, labels, subkey, training,
    )

    if training:
        updates, new_opt_state = optimizer.update(
            grads, state.opt_state, state.params,
        )
        new_params = optax.apply_updates(state.params, updates)
        new_variables = {**state.variables, **updated_vars}
    else:
        new_params = state.params
        new_opt_state = state.opt_state
        new_variables = state.variables

    new_state = TrainState(
        step=state.step + 1 if training else state.step,
        params=new_params,
        opt_state=new_opt_state,
        variables=new_variables,
        key=key,
    )

    return new_state, ce_loss, aux_loss


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _find_latest(save_dir: Path) -> int:
    steps = []
    for d in save_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                steps.append(int(d.name.split("_")[1]))
            except (ValueError, IndexError):
                pass
    return max(steps) if steps else -1


def _save(state: TrainState, checkpointer, save_dir: Path, keep_n: int, final: bool = False):
    step = state.step
    ckpt_dir = str(save_dir / f"step_{step}")

    checkpointer.save(ckpt_dir, args=ocp.args.StandardSave(state.to_checkpoint()))

    steps = sorted(
        int(d.name.split("_")[1])
        for d in save_dir.iterdir()
        if d.is_dir() and d.name.startswith("step_")
    )
    while len(steps) > keep_n + 2:
        old = save_dir / f"step_{steps[0]}"
        if old.exists():
            shutil.rmtree(str(old))
        steps = steps[1:]

    if final:
        fd = save_dir / "final"
        if fd.exists():
            shutil.rmtree(str(fd))
        shutil.copytree(ckpt_dir, str(fd))


# ---------------------------------------------------------------------------
# Token parsing
# ---------------------------------------------------------------------------

def _parse_tokens(s: str) -> int:
    s = s.strip().upper()
    if s.endswith("B"):
        return int(float(s[:-1]) * 1e9)
    elif s.endswith("M"):
        return int(float(s[:-1]) * 1e6)
    elif s.endswith("K"):
        return int(float(s[:-1]) * 1e3)
    return int(s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_config(args) -> LogosConfig:
    tokenizer = TiktokenTokenizer(args.tiktoken_encoding)
    return LogosConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        max_seq_len=args.max_length,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        dropout=args.dropout,
        norm_eps=args.norm_eps,
        d_ff=args.d_ff,
        use_moe=args.use_moe,
        num_shared_experts=args.num_shared_experts,
        num_sparse_experts=args.num_sparse_experts,
        top_k=args.top_k,
        expert_d_ff=args.expert_d_ff,
        bias_update_rate=args.bias_update_rate,
        moe_diversity_factor=args.moe_diversity_factor,
        chunk_size=args.chunk_size,
        conv_size=args.conv_size,
        snapshot_interval=args.snapshot_interval,
        snapshot_latent_dim=args.snapshot_latent_dim,
        mem_top_k=args.mem_top_k,
        mem_head_dim=args.mem_head_dim,
        rope_base=args.rope_base,
        rope_scaling_type=args.rope_scaling_type,
        rope_scaling_factor=args.rope_scaling_factor,
        swa_window=args.swa_window,
        swa_every=args.swa_every,
        swa_offset=args.swa_offset,
        num_entry_layers=args.num_entry_layers,
        num_body_layers=args.num_body_layers,
        num_exit_layers=args.num_exit_layers,
        num_loops=args.num_loops,
        lm_head_chunk_size=args.lm_head_chunk_size,
    )


def build_optimizer(
    args, params: Dict, total_steps: int,
) -> Tuple[optax.GradientTransformation, Dict]:
    """Build Muon + AdamW optimizer with per-group LRs and WSD schedules."""

    # Build LR schedules for each group
    muon_schedule = create_wsd_schedule(
        total_steps, args.warmup_steps, args.decay_steps, args.muon_lr,
    )
    adamw_schedule = create_wsd_schedule(
        total_steps, args.warmup_steps, args.decay_steps, args.lr,
    )
    embed_schedule = create_wsd_schedule(
        total_steps, args.warmup_steps, args.decay_steps, args.embed_lr,
    )

    # Build muon weight dimension numbers: 2D params get Muon, rest get AdamW
    def _is_embed(path, _leaf):
        ps = "/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)
        return "token_emb" in ps or "lm_head" in ps

    def _is_weight_decay(path, leaf):
        """Weight decay applies to weights, not biases/norms."""
        if not isinstance(leaf, jnp.ndarray):
            return False
        ps = "/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)
        skip_patterns = ("scale", "bias", "norm", "_w", "sink_logit", "A_log", "dt_bias")
        return not any(p in ps for p in skip_patterns) and leaf.ndim >= 1

    # Partition into muon (2D weights) and adamw (everything else)
    def _make_muon_spec(path, leaf):
        if not isinstance(leaf, jnp.ndarray) or leaf.ndim != 2:
            return None
        ps = "/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)
        if "token_emb" in ps:
            return None
        if "lm_head" in ps:
            return None
        # 2D linear weight → Muon
        return optax.contrib.MuonDimensionNumbers()

    muon_dim_nums = jax.tree_util.tree_map_with_path(_make_muon_spec, params)

    # Check if muon was requested
    if args.muon:
        optimizer = optax.contrib.muon(
            learning_rate=muon_schedule,
            weight_decay=args.weight_decay,
            muon_weight_dimension_numbers=muon_dim_nums,
            adam_learning_rate=adamw_schedule,
            adam_b1=0.9,
            adam_b2=0.95,
            adam_weight_decay=args.weight_decay,
        )
    else:
        optimizer = optax.adamw(
            learning_rate=adamw_schedule,
            weight_decay=args.weight_decay,
            b1=0.9,
            b2=0.95,
        )

    # Clipping
    if args.grad_clip > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optimizer,
        )

    return optimizer


def main(args: argparse.Namespace):
    # ---- Device info ----
    print(f"Jax devices: {jax.devices()}")
    print(f"Jax backend: {jax.default_backend()}")

    # ---- Device mesh for data parallelism ----
    devices = np.array(jax.devices())
    n_devices = len(devices)
    mesh = Mesh(devices, ("data",))
    data_sharding = NamedSharding(mesh, P("data", None))  # (B, T) sharded on B
    repl_sharding = NamedSharding(mesh, P())              # replicated everywhere

    if args.batch_size % n_devices != 0:
        raise ValueError(
            f"batch_size ({args.batch_size}) must be divisible by num devices ({n_devices})"
        )
    print(
        f"Devices: {n_devices}, global batch: {args.batch_size}, "
        f"per-device batch: {args.batch_size // n_devices}"
    )

    # ---- Config & model ----
    config = build_config(args)
    model = LogosTransformer(config)

    # Init (on default device; we replicate immediately after)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    dummy_ids = jnp.ones((args.batch_size, args.max_length), dtype=jnp.int32)
    dummy_labels = jnp.ones((args.batch_size, args.max_length), dtype=jnp.int32)

    variables = model.init(init_key, dummy_ids, dummy_labels, deterministic=False)
    params = variables["params"]
    moe_vars = {k: v for k, v in variables.items() if k != "params"}

    print(f"Trainable params: {count_parameters(params):,}")

    # Replicate params + non-param variables across the mesh
    params = jax.device_put(params, repl_sharding)
    moe_vars = jax.device_put(moe_vars, repl_sharding)
    key = jax.device_put(key, repl_sharding)

    # ---- Schedule ----
    total_tokens = _parse_tokens(args.total_tokens)
    tokens_per_step = args.batch_size * args.max_length
    total_steps = int(math.ceil(total_tokens / tokens_per_step))
    print(f"Total tokens: {total_tokens:,}, per step: {tokens_per_step}")
    print(f"Total steps: {total_steps:,}")

    # ---- Optimizer ----
    optimizer = build_optimizer(args, params, total_steps)
    opt_state = optimizer.init(params)
    opt_state = jax.device_put(opt_state, repl_sharding)

    # ---- Checkpointing ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_handler = ocp.StandardCheckpointHandler()
    checkpointer = ocp.Checkpointer(ckpt_handler)

    # ---- Resume if possible ----
    latest_step = _find_latest(save_dir)
    if latest_step > 0:
        print(f"Resuming from step {latest_step}")
        # Create empty template for structure restoration
        template = TrainState(
            step=0, params=params, opt_state=opt_state,
            variables=moe_vars, key=key,
        )
        restored = checkpointer.restore(
            save_dir / f"step_{latest_step}",
            args=ocp.args.StandardRestore(template.to_checkpoint()),
        )
        state = TrainState.from_checkpoint(restored, opt_state, key)
        # Re-shard restored leaves onto the mesh (Orbax may return host arrays)
        state.params = jax.device_put(state.params, repl_sharding)
        state.variables = jax.device_put(state.variables, repl_sharding)
        state.opt_state = jax.device_put(state.opt_state, repl_sharding)
    else:
        print("Starting fresh training")
        state = TrainState(step=0, params=params, opt_state=opt_state,
                           variables=moe_vars, key=key)

    # ---- DataLoader ----
    tokenizer = TiktokenTokenizer(args.tiktoken_encoding)
    loader = StreamingDataLoader(
        args.dataset, args.dataset_config, args.text_column,
        tokenizer, args.batch_size, args.max_length, args.seed,
    )

    # ---- Training loop ----
    log_every = args.log_every
    save_every = args.save_every
    history = []

    pbar = tqdm(total=total_steps, initial=state.step, desc="steps")
    step_t0 = time.time()
    running_loss = 0.0
    running_aux = 0.0
    steps_since_log = 0

    for batch_idx, (input_ids, labels) in enumerate(loader):
        input_ids = jax.device_put(jnp.asarray(input_ids), data_sharding)
        labels = jax.device_put(jnp.asarray(labels), data_sharding)
        state, ce_loss, aux_loss = train_step(
            state, model, optimizer,
            input_ids, labels,
            training=True,
        )

        running_loss += float(ce_loss)
        running_aux += float(aux_loss)
        steps_since_log += 1

        if state.step % log_every == 0:
            dt = max(time.time() - step_t0, 0.01)
            tok_s = (args.batch_size * args.max_length * steps_since_log) / dt
            avg_loss = running_loss / steps_since_log
            avg_aux = running_aux / steps_since_log

            pbar.set_postfix_str(
                f"loss={avg_loss:.4f} aux={avg_aux:.4f} tok/s={tok_s:.0f}"
            )
            pbar.update(steps_since_log)

            history.append({
                "step": int(state.step),
                "loss": avg_loss,
                "aux_loss": avg_aux,
            })
            running_loss = 0.0
            running_aux = 0.0
            steps_since_log = 0
            step_t0 = time.time()

        if state.step % save_every == 0:
            _save(state, checkpointer, save_dir, args.keep_last_n)

        if state.step >= total_steps:
            break

    # Final save
    _save(state, checkpointer, save_dir, args.keep_last_n, final=True)

    with (save_dir / "history.json").open("w") as f:
        json.dump({"history": history}, f, indent=2)

    pbar.close()
    print(f"Done. Checkpoints: {save_dir}")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
