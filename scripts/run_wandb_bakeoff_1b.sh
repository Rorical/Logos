#!/usr/bin/env bash
set -euo pipefail

# Serial W&B bakeoff for the Logos model family.
#
# Goal: an ISO-FLOP architecture comparison. Every variant is SMALL and executes
# the SAME number of transformer blocks (EFFECTIVE_DEPTH) at the SAME width
# (D_MODEL / NUM_HEADS / D_FF), with a DENSE SwiGLU FFN (no MoE). That keeps the
# attention-projection + FFN + depth FLOPs identical across variants, so the only
# variable is the attention mechanism / depth-routing structure of each family.
#
#   baseline / linear / hybrid : EFFECTIVE_DEPTH plain blocks
#   residual                   : EFFECTIVE_DEPTH blocks, grouped into RESIDUAL_NUM_BLOCKS
#   recursive / logos          : entry + body * loops + exit  ==  EFFECTIVE_DEPTH
#
# Defaults target a 4x A800/H100-80GB node reading a LOCAL parquet corpus, but
# every knob is overridable from the environment, e.g.:
#
#   BATCH_SIZE=8 TOTAL_TOKENS=100M ./scripts/run_wandb_bakeoff_1b.sh
#   DRY_RUN=1 ./scripts/run_wandb_bakeoff_1b.sh logos hybrid baseline

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPUS="${GPUS:-4}"
TOTAL_TOKENS="${TOTAL_TOKENS:-25M}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-2048}"

# Local parquet corpus (text column "content"). A HF hub name also works.
DATASET="${DATASET:-/root/autodl-tmp/datasets/ultrafineweb-l3-en-qa}"
DATASET_CONFIG="${DATASET_CONFIG:-}"
TEXT_COLUMN="${TEXT_COLUMN:-content}"
TOKENIZER="${TOKENIZER:-cl100k_base}"
VAL_DOCS="${VAL_DOCS:-200}"

WANDB_PROJECT="${WANDB_PROJECT:-logos-bakeoff-isoflop}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-offline}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-isoflop}"
RUNS_ROOT="${RUNS_ROOT:-runs/bakeoff-isoflop}"

NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-8}"
LOG_EVERY="${LOG_EVERY:-25}"
EVAL_EVERY="${EVAL_EVERY:-200}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
SAMPLE_EVERY="${SAMPLE_EVERY:-1000}"
KEEP_LAST_N="${KEEP_LAST_N:-2}"
DIAGNOSTIC_EVERY="${DIAGNOSTIC_EVERY:-100}"
OPT_STATE_LOG_EVERY="${OPT_STATE_LOG_EVERY:-500}"
RESUME="${RESUME:-auto}"
DRY_RUN="${DRY_RUN:-0}"
COMPILE="${COMPILE:-1}"
COMPILE_MODE="${COMPILE_MODE:-default}"

# --- Shared (iso-FLOP) model geometry ------------------------------------------
D_MODEL="${D_MODEL:-512}"
NUM_HEADS="${NUM_HEADS:-8}"
HEAD_DIM="${HEAD_DIM:-64}"
D_FF="${D_FF:-1364}"          # ~ (8/3) * d_model, dense SwiGLU intermediate

# Effective executed transformer blocks per forward pass (held equal for all).
EFFECTIVE_DEPTH="${EFFECTIVE_DEPTH:-12}"
RESIDUAL_NUM_BLOCKS="${RESIDUAL_NUM_BLOCKS:-4}"   # must divide EFFECTIVE_DEPTH

# Looped families: entry + body * loops + exit must equal EFFECTIVE_DEPTH.
NUM_ENTRY_LAYERS="${NUM_ENTRY_LAYERS:-2}"
NUM_BODY_LAYERS="${NUM_BODY_LAYERS:-2}"
NUM_EXIT_LAYERS="${NUM_EXIT_LAYERS:-2}"
NUM_LOOPS="${NUM_LOOPS:-4}"

# --- Attention sub-mechanism knobs (hybrid / logos / linear) -------------------
CHUNK_SIZE="${CHUNK_SIZE:-128}"
CONV_SIZE="${CONV_SIZE:-4}"
SWA_WINDOW="${SWA_WINDOW:-256}"
SWA_EVERY="${SWA_EVERY:-4}"
SWA_OFFSET="${SWA_OFFSET:-3}"
CSA_COMPRESSION="${CSA_COMPRESSION:-4}"
CSA_TOP_K="${CSA_TOP_K:-64}"
CSA_INDEXER_HEADS="${CSA_INDEXER_HEADS:-4}"
CSA_INDEXER_DIM="${CSA_INDEXER_DIM:-32}"
HCA_COMPRESSION="${HCA_COMPRESSION:-128}"
LM_HEAD_CHUNK_SIZE="${LM_HEAD_CHUNK_SIZE:-2048}"
CKPT_GRANULARITY="${CKPT_GRANULARITY:-per-block}"

# Attention schedules. hybrid cycles over EFFECTIVE_DEPTH; logos sections expand to
# entry / (body*loops) / exit. All four kinds (hca,csa,swa,kda) appear in each.
HYBRID_ATTN_PATTERN="${HYBRID_ATTN_PATTERN:-hca,kda,swa,kda,csa,kda}"
ENTRY_ATTN_PATTERN="${ENTRY_ATTN_PATTERN:-hca,kda}"
BODY_ATTN_PATTERN="${BODY_ATTN_PATTERN:-swa,kda,csa,kda,swa,kda,hca,kda}"
EXIT_ATTN_PATTERN="${EXIT_ATTN_PATTERN:-csa,swa}"

# --- Optimization --------------------------------------------------------------
LR="${LR:-0.002}"
EMBED_LR="${EMBED_LR:-0.04}"
MUON_LR="${MUON_LR:-0.004}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
DECAY_STEPS="${DECAY_STEPS:-}"
DECAY_FRAC="${DECAY_FRAC:-0.2}"
EMA_DECAY="${EMA_DECAY:-0.0}"

SAMPLE_PROMPT="${SAMPLE_PROMPT:-In a recent study, researchers found that}"
SAMPLE_MAX_TOKENS="${SAMPLE_MAX_TOKENS:-80}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-0.8}"

if [[ "$#" -gt 0 ]]; then
  VARIANTS=("$@")
else
  VARIANTS=(baseline residual recursive linear hybrid logos)
fi

# Sanity: looped depth and residual grouping must reconcile with EFFECTIVE_DEPTH.
looped_depth=$(( NUM_ENTRY_LAYERS + NUM_BODY_LAYERS * NUM_LOOPS + NUM_EXIT_LAYERS ))
if [[ "$looped_depth" -ne "$EFFECTIVE_DEPTH" ]]; then
  echo "ERROR: entry+body*loops+exit ($looped_depth) != EFFECTIVE_DEPTH ($EFFECTIVE_DEPTH)" >&2
  exit 2
fi
if (( EFFECTIVE_DEPTH % RESIDUAL_NUM_BLOCKS != 0 )); then
  echo "ERROR: EFFECTIVE_DEPTH ($EFFECTIVE_DEPTH) not divisible by RESIDUAL_NUM_BLOCKS ($RESIDUAL_NUM_BLOCKS)" >&2
  exit 2
fi

shared_args=(
  --streaming
  --dataset "$DATASET"
  --text-column "$TEXT_COLUMN"
  --val-docs "$VAL_DOCS"
  --tiktoken-encoding "$TOKENIZER"
  --total-tokens "$TOTAL_TOKENS"
  --batch-size "$BATCH_SIZE"
  --max-length "$MAX_LENGTH"
  --log-every "$LOG_EVERY"
  --eval-every "$EVAL_EVERY"
  --save-every "$SAVE_EVERY"
  --sample-every "$SAMPLE_EVERY"
  --keep-last-n "$KEEP_LAST_N"
  --num-workers "$NUM_WORKERS"
  --prefetch-factor "$PREFETCH_FACTOR"
  --bf16
  --diagnostic-every "$DIAGNOSTIC_EVERY"
  --d-model "$D_MODEL"
  --num-heads "$NUM_HEADS"
  --head-dim "$HEAD_DIM"
  --d-ff "$D_FF"
  --chunk-size "$CHUNK_SIZE"
  --conv-size "$CONV_SIZE"
  --swa-window "$SWA_WINDOW"
  --swa-every "$SWA_EVERY"
  --swa-offset "$SWA_OFFSET"
  --csa-compression "$CSA_COMPRESSION"
  --csa-top-k "$CSA_TOP_K"
  --csa-indexer-heads "$CSA_INDEXER_HEADS"
  --csa-indexer-dim "$CSA_INDEXER_DIM"
  --hca-compression "$HCA_COMPRESSION"
  --lm-head-chunk-size "$LM_HEAD_CHUNK_SIZE"
  --muon
  --muon-schedule-hyperparams
  --lr "$LR"
  --embed-lr "$EMBED_LR"
  --muon-lr "$MUON_LR"
  --weight-decay "$WEIGHT_DECAY"
  --grad-clip "$GRAD_CLIP"
  --scheduler wsd
  --warmup-steps "$WARMUP_STEPS"
  --decay-frac "$DECAY_FRAC"
  --opt-state-log-every "$OPT_STATE_LOG_EVERY"
  --ema-decay "$EMA_DECAY"
  --sample-prompt "$SAMPLE_PROMPT"
  --sample-max-tokens "$SAMPLE_MAX_TOKENS"
  --sample-temperature "$SAMPLE_TEMPERATURE"
  --resume "$RESUME"
  --wandb
  --wandb-project "$WANDB_PROJECT"
  --wandb-mode "$WANDB_MODE"
)

if [[ -n "$DATASET_CONFIG" ]]; then
  shared_args+=(--dataset-config "$DATASET_CONFIG")
fi

if [[ -n "$WANDB_ENTITY" ]]; then
  shared_args+=(--wandb-entity "$WANDB_ENTITY")
fi

if [[ "$COMPILE" == "1" ]]; then
  shared_args+=(--compile --compile-mode "$COMPILE_MODE")
fi

if [[ -n "$DECAY_STEPS" ]]; then
  shared_args+=(--decay-steps "$DECAY_STEPS")
fi

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

for variant in "${VARIANTS[@]}"; do
  run_name="${RUN_NAME_PREFIX}-${variant}-${TOTAL_TOKENS}"
  save_dir="${RUNS_ROOT}/${run_name}"

  case "$variant" in
    baseline)
      model_args=(--model baseline --num-layers "$EFFECTIVE_DEPTH")
      ;;
    residual)
      model_args=(
        --model residual
        --num-layers "$EFFECTIVE_DEPTH"
        --num-blocks "$RESIDUAL_NUM_BLOCKS"
      )
      ;;
    recursive)
      model_args=(
        --model recursive
        --num-entry-layers "$NUM_ENTRY_LAYERS"
        --num-body-layers "$NUM_BODY_LAYERS"
        --num-exit-layers "$NUM_EXIT_LAYERS"
        --num-loops "$NUM_LOOPS"
      )
      ;;
    linear)
      model_args=(--model linear --num-layers "$EFFECTIVE_DEPTH")
      ;;
    hybrid)
      model_args=(
        --model hybrid
        --num-layers "$EFFECTIVE_DEPTH"
        --attn-pattern "$HYBRID_ATTN_PATTERN"
      )
      ;;
    logos)
      model_args=(
        --model logos
        --num-entry-layers "$NUM_ENTRY_LAYERS"
        --num-body-layers "$NUM_BODY_LAYERS"
        --num-exit-layers "$NUM_EXIT_LAYERS"
        --num-loops "$NUM_LOOPS"
        --entry-attn-pattern "$ENTRY_ATTN_PATTERN"
        --body-attn-pattern "$BODY_ATTN_PATTERN"
        --exit-attn-pattern "$EXIT_ATTN_PATTERN"
      )
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      echo "Expected one of: baseline residual recursive linear hybrid logos" >&2
      exit 2
      ;;
  esac

  cmd=(
    uv run torchrun
    --standalone
    --nproc_per_node "$GPUS"
    scripts/train.py
    "${model_args[@]}"
    "${shared_args[@]}"
    --save-dir "$save_dir"
    --wandb-run-name "$run_name"
    --wandb-tags
    bakeoff
    isoflop
    "$variant"
    "$TOTAL_TOKENS"
    "gpus-$GPUS"
  )

  echo
  echo "================================================================"
  echo "Starting W&B bakeoff run: $run_name"
  echo "  variant:        $variant"
  echo "  effective depth:$EFFECTIVE_DEPTH blocks  (d_model=$D_MODEL d_ff=$D_FF dense)"
  echo "  save_dir:       $save_dir"
  echo "  token budget:   $TOTAL_TOKENS"
  echo "  per-GPU batch:  $BATCH_SIZE x $MAX_LENGTH  on $GPUS GPUs"
  echo "================================================================"

  if [[ "$DRY_RUN" == "1" ]]; then
    print_cmd "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
done
