# Logos

A sub-quadratic decoder-only transformer.

Each parameter-block is one of two kinds:

- **KDA + Retrieval** Kimi Delta Attention scan, followed by sparse top-k attention over MLA-compressed state snapshots.
- **Local SWA** Causal sliding-window softmax attention with a fixed window of `swa_window` tokens.

Block kind is determined structurally by `layer_idx % swa_every == swa_offset`.

The model is partitioned into three sections — **Entry → Body → Exit** — where the Body is `num_body_layers` shared-weight blocks reused `num_loops` times per forward. Sublayer inputs are computed by Block Attention Residual: a learned depth-wise softmax over the list of completed-block representations and the current partial sum. Section and loop boundaries close partial sums into new completed blocks; a final Block Attention Residual produces the LM-head input.

Body MoE layers carry per-loop router-bias rows and a cross-loop expert-diversity term (`moe_diversity_factor`), so shared weights specialise differently across loop iterations.

The repository also ships six ancestor variants — `baseline`, `linear`, `recursive`, `residual`, `superlinear`, `hybrid` — usable on their own and serving as ablation references for Logos.

## Block list

The depth-wise softmax inside every sublayer attends over an evolving `blocks` list whose entries correspond to natural model boundaries:

```text
block 0      : token embedding (after dropout)
block 1      : entry output
blocks 2..L+1: each loop iteration's closed state (L = num_loops)
final        : exit's partial accumulation, fed through one more
               BlockAttentionResidual to produce the lm_head input
```

| Sublayer | KDA blocks | SWA blocks |
| --- | --- | --- |
| Attention input | `kda_res(blocks, partial)` → `SuperKimiDeltaAttention` | `attn_res(blocks, partial)` → `LocalAttention` |
| Memory input | `mem_res(blocks, partial)` → `SnapshotRetrieval` | — |
| FFN input | `ffn_res(blocks, partial)` → `MoELayer` / `SwiGLU` | `ffn_res(blocks, partial)` → `MoELayer` / `SwiGLU` |

## Initialization

Initialization is chosen so the model behaves like a familiar baseline at step 0:

- `BlockAttentionResidual.proj` is zero-initialised, so each depth-wise softmax starts uniform and matches a standard residual sum in expectation.
- `SnapshotRetrieval.out_up` is zero-initialised, so the memory branch is an exact no-op at step 0.

## Ancestor variants

Each ancestor lives in its own file and is a usable model in its own right. Logos imports concrete pieces from each rather than copying code:

| Variant | Contributes to Logos |
| --- | --- |
| `baseline` | `RMSNorm`, `SwiGLU`, `MoELayer` (per-loop bias rows + cross-loop diversity), shared MoE static-shape dispatch |
| `linear` | Kimi Delta Attention primitives (`_ShortConvolution`, `_kda_gate`, `_kda_recurrent_step`) |
| `superlinear` | `SuperKimiDeltaAttention` (KDA + state snapshots), `SnapshotRetrieval` (sparse top-k attention over MLA-compressed latents) |
| `hybrid` | `HybridConfig` (parent of `LogosConfig`), `LocalAttention` for SWA layers |
| `recursive` | The Entry / Body / Exit looped-depth orchestration |
| `residual` | `BlockAttentionResidual` (depth-wise softmax over completed-block representations + partial) |

## Architecture options shared across variants

- **Q/K RMSNorm** (`qk_norm`): per-head RMSNorm on Q and K before scoring. Default on; KDA already L2-normalises q/k inside the chunkwise scan.
- **Partial RoPE** (`partial_rope_dim`): rotate only the last *N* channels per head, leaving the rest as content-only features. Default `None` (full rotation).
- **Attention sink** (`attention_sink`): per-head learnable logit appended to the softmax denominator (StreamingLLM / GPT-OSS-style). Default on.
- **MoE static dispatch**: expert assignment uses `(positions × is_first).cummax` + sentinel-column scatter — no `nonzero`, no boolean indexing, no graph breaks under `torch.compile`.
- **NTK / YaRN scaling** (retrieval RoPE): three modes (`none` / `ntk` / `yarn`) for long-context extrapolation. YaRN uses the corrected pair-index clamp (the fix for the unit error in the original / HF YaRN reference).

## Project structure

```text
logos/
├── models/
│   ├── baseline.py        # BaselineTransformer + shared building blocks
│   ├── linear.py          # LinearTransformer (Kimi Delta Attention)
│   ├── recursive.py       # RecursiveTransformer (looped depth)
│   ├── residual.py        # ResidualTransformer (Block AttnRes)
│   ├── superlinear.py     # SuperLinearTransformer (KDA + snapshot memory)
│   ├── hybrid.py          # HybridTransformer (KDA + snapshot + local SWA)
│   └── logos.py           # LogosTransformer
├── scripts/
│   ├── train.py           # Unified causal LM training, --model {baseline, linear, ...}
│   └── train_chat.py      # ChatML-formatted fine-tune (assistant-only loss masking)
├── notebooks/
│   └── train_colab.ipynb  # 1 B-MoE bake-off across all seven variants on A100
├── utils/
│   └── tokenizer.py       # Tiktoken wrapper with ChatML support
├── pyproject.toml         # uv-managed dependencies
└── README.md
```

## Quick start

Constructing a Logos model directly:

```python
from models.logos import LogosConfig, LogosTransformer

cfg = LogosConfig(
    vocab_size=32000, d_model=512, num_heads=8, head_dim=64,
    num_entry_layers=2, num_body_layers=4, num_exit_layers=2,
    num_loops=4,
    swa_every=4, swa_offset=3, swa_window=256,
    chunk_size=64, snapshot_interval=256,
    snapshot_latent_dim=128, mem_top_k=16, mem_head_dim=64, mem_latent_dim=128,
    use_moe=True, num_sparse_experts=64, top_k=6, expert_d_ff=256,
    moe_diversity_factor=0.05,
)
model = LogosTransformer(cfg)
out = model(input_ids, attention_mask=attention_mask, labels=labels)

# After optimizer.step(), update the bias-balancing routers:
model.update_router_biases(out["topk_indices"])
```

End-to-end training via `scripts/train.py --model logos`:

```bash
uv sync

uv run python scripts/train.py \
    --model logos \
    --dataset tiny_shakespeare \
    --d-model 512 --num-heads 8 --head-dim 64 \
    --num-entry-layers 2 --num-body-layers 4 \
        --num-exit-layers 2 --num-loops 4 \
    --snapshot-interval 256 --snapshot-latent-dim 128 \
        --mem-top-k 16 --mem-head-dim 64 \
    --swa-every 4 --swa-offset 3 --swa-window 256 \
    --use-moe --num-sparse-experts 64 --top-k 6 \
        --moe-diversity-factor 0.5 \
    --batch-size 4 --epochs 20 --lr 3e-4
```

`scripts/train.py --help` lists every flag (per-variant flags are gated to the relevant `--model`).

## Bake-off notebook

`notebooks/train_colab.ipynb` runs all seven variants at ~1 B parameters with matched MoE settings on WikiText-103, then loads each run's `runs/<name>/history.json` and produces a side-by-side loss-curve plot. Each variant uses `--no-save` so only the lightweight `history.json` files are written; drop the flag to keep weights.

## torch.compile compatibility

The training forward and backward compile cleanly under `torch.compile` with `--compile`:

- MoE expert dispatch uses static-shape scatter (sentinel-column trick) — no `nonzero`, no boolean indexing.
- KDA chunkwise scan is loop-unrolled at compile time; snapshot emission inside the scan is a compile-time-resolved condition.
- `RotaryEmbedding.forward` raises a clear error when `seq_len > max_seq_len` instead of silently truncating.
- `BlockAttentionResidual` is a thin `RMSNorm + Linear(D, 1) + softmax`; softmax runs in fp32 for stability.

Inference and cached generation (KV cache for SWA, recurrent state for KDA, growing snapshot history for retrieval) fall back to eager. Logos's `generate()` does naive re-forward across the growing prefix; per-step cache reuse is not implemented because the body re-runs its KDA scan on every loop iteration.

## Requirements

- Python ≥ 3.13
- PyTorch with CUDA support
- `uv` for dependency management

## License

MIT
