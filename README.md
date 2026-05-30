# Logos

A decoder-only language-model playground centered on **Logos**, a looped hybrid-attention transformer.

Logos now uses direct attention variants only:

- **KDA** Kimi Delta Attention recurrent/chunkwise scan.
- **SWA** causal sliding-window softmax attention.
- **CSA** DeepSeek-V4-style compressed sparse global attention: learned 4-token KV compression, first-stage indexer scoring, then full attention over top-k compressed entries.
- **HCA** DeepSeek-V4-style heavily compressed global attention: learned large-ratio KV compression, then dense attention over all compressed entries.

The old snapshot-memory path and `superlinear` model variant have been removed.

## Logos Layout

The model is partitioned into three sections:

```text
Entry -> Body x num_loops -> Exit
```

Entry and Exit run once. Body is a shared-weight stack reused `num_loops` times per forward pass.

Sublayer inputs use Block Attention Residual: a learned depth-wise softmax over completed block representations plus the current partial block. Section and loop boundaries close partial sums into the completed-block list, and a final Block Attention Residual produces the LM-head input.

```text
block 0      : token embedding
block 1      : entry output
blocks 2..L+1: each body loop output, L = num_loops
final        : exit partial plus completed blocks -> final_res -> lm_head
```

## Attention Scheduling

Every Logos block can execute `kda`, `swa`, `csa`, or `hca`.

Fine-grained schedules are comma- or semicolon-separated patterns:

- `--entry-attn-pattern`: expanded across `num_entry_layers`.
- `--body-attn-pattern`: expanded across `num_loops * num_body_layers` in loop-major order.
- `--exit-attn-pattern`: expanded across `num_exit_layers`.
- `--attn-pattern`: global fallback for Hybrid, and for Logos when section-specific patterns are unset.

Example body order for `num_body_layers=4`, `num_loops=3`:

```text
loop0.block0, loop0.block1, loop0.block2, loop0.block3,
loop1.block0, loop1.block1, loop1.block2, loop1.block3,
loop2.block0, loop2.block1, loop2.block2, loop2.block3
```

If no explicit pattern is provided, Logos preserves the old structural KDA/SWA schedule controlled by `swa_every` and `swa_offset`.

## Architecture Options

- `--csa-compression`: CSA compression ratio, default `4`.
- `--csa-top-k`: CSA sparse recall top-k over compressed entries, default `1024`.
- `--csa-indexer-heads`: CSA first-stage indexer heads, default `4`.
- `--csa-indexer-dim`: CSA first-stage indexer head dimension, default `32`.
- `--hca-compression`: HCA compression ratio, default `128`.
- `--compressed-head-dim`: CSA/HCA shared-KV per-head dimension, default `--head-dim`.
- `--compressed-query-dim`: CSA/HCA low-rank query bottleneck, default `--compressed-head-dim`.
- `--swa-window`: SWA local window, default `256`.
- `--chunk-size`: KDA chunk size.

Shared options across variants include Q/K RMSNorm, partial RoPE for standard attention, attention sink logits, MoE static dispatch, and optional Block Attention Residual softmax isolation for `torch.compile` stability on SMEM-constrained GPUs.

### CSA and HCA Design Notes

**CSA overlap.** CSA compresses tokens into fixed-size groups using two learned projections (A and B). The B projection applies a group-aligned offset so each compressed entry pools tokens from two consecutive windows (current A-group + previous B-group), providing two independent views of adjacent token blocks within a single compressed representation. This structure reduces boundary artefacts where information would otherwise be lost at group edges.

**HCA causal delay.** Compressed attention entries are only visible once their token group is complete. With `hca_compression=128`, the first compressed group covers tokens 0..127 and becomes available at position 127, so positions 0..126 see no compressed attention from that group. With `csa_compression=4` (the CSA default), this delay is only 4 tokens and is negligible in practice. Other attention types (SWA, KDA) provide local and intermediate-range coverage within the same block.

## Project Structure

```text
logos/
├── models/
│   ├── baseline.py     # BaselineTransformer + shared RMSNorm/RoPE/MoE pieces
│   ├── linear.py       # LinearTransformer with Kimi Delta Attention
│   ├── recursive.py    # RecursiveTransformer with looped body depth
│   ├── residual.py     # ResidualTransformer + BlockAttentionResidual
│   ├── hybrid.py       # KDA/SWA/CSA/HCA HybridTransformer pieces
│   └── logos.py        # LogosTransformer
├── scripts/
│   └── train.py        # Unified causal LM training
├── notebooks/          # Colab wrappers and bake-off notebooks
├── utils/
│   ├── diagnostics.py  # Training diagnostics
│   └── tokenizer.py    # Tiktoken wrapper with ChatML support
├── pyproject.toml
└── README.md
```

## Quick Start

Construct a Logos model directly:

```python
from models.logos import LogosConfig, LogosTransformer

cfg = LogosConfig(
    vocab_size=32000,
    d_model=512,
    num_heads=8,
    head_dim=64,
    num_entry_layers=2,
    num_body_layers=4,
    num_exit_layers=2,
    num_loops=4,
    entry_attn_pattern="hca,kda",
    body_attn_pattern="hca,csa,kda,swa,csa,csa,kda,swa",
    exit_attn_pattern="csa,swa",
    csa_compression=4,
    csa_top_k=64,
    hca_compression=128,
    use_moe=True,
    num_sparse_experts=64,
    top_k=6,
    expert_d_ff=256,
)
model = LogosTransformer(cfg)
out = model(input_ids, attention_mask=attention_mask, labels=labels)

model.update_router_biases(out["topk_indices"])
```

Train from the CLI:

```bash
uv sync

uv run python scripts/train.py \
    --model logos \
    --dataset tiny_shakespeare \
    --d-model 512 --num-heads 8 --head-dim 64 \
    --num-entry-layers 2 --num-body-layers 4 \
    --num-exit-layers 2 --num-loops 4 \
    --entry-attn-pattern hca,kda \
    --body-attn-pattern hca,csa,kda,swa,csa,csa,kda,swa \
    --exit-attn-pattern csa,swa \
    --csa-compression 4 --csa-top-k 64 \
    --hca-compression 128 --swa-window 256 \
    --use-moe --num-sparse-experts 64 --top-k 6 \
    --batch-size 4 --epochs 20 --lr 3e-4
```

`scripts/train.py --help` lists all flags.

## Variants

The training registry currently includes:

- `baseline`
- `linear`
- `recursive`
- `residual`
- `hybrid`
- `logos`

`hybrid` is a non-looped stack using the same `kda/swa/csa/hca` attention modules. `logos` adds Entry/Body/Exit looping plus Block Attention Residuals.

## Requirements

- Python >= 3.13
- PyTorch with CUDA support for GPU training
- `uv` for dependency management

## License

MIT
