#!/usr/bin/env bash
set -euo pipefail

# Serial W&B bakeoff for the Logos model family.
#
# Defaults are chosen for a 4xH100/NVLink node, but every important knob can be
# overridden from the environment:
#
#   BATCH_SIZE=4 WANDB_PROJECT=logos-bakeoff-25m ./scripts/run_wandb_bakeoff_1b.sh
#   DRY_RUN=1 ./scripts/run_wandb_bakeoff_1b.sh logos hybrid baseline

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPUS="${GPUS:-4}"
TOTAL_TOKENS="${TOTAL_TOKENS:-25M}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_LENGTH="${MAX_LENGTH:-4096}"

DATASET="${DATASET:-HuggingFaceFW/fineweb-edu}"
DATASET_CONFIG="${DATASET_CONFIG:-sample-100BT}"
TEXT_COLUMN="${TEXT_COLUMN:-text}"
TOKENIZER="${TOKENIZER:-cl100k_base}"
VAL_DOCS="${VAL_DOCS:-200}"

WANDB_PROJECT="${WANDB_PROJECT:-logos-bakeoff-25m}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-25m}"
RUNS_ROOT="${RUNS_ROOT:-runs/wandb-bakeoff-25m}"

NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-8}"
LOG_EVERY="${LOG_EVERY:-25}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
SAMPLE_EVERY="${SAMPLE_EVERY:-1000}"
KEEP_LAST_N="${KEEP_LAST_N:-2}"
DIAGNOSTIC_EVERY="${DIAGNOSTIC_EVERY:-100}"
MOE_LOG_EVERY="${MOE_LOG_EVERY:-500}"
OPT_STATE_LOG_EVERY="${OPT_STATE_LOG_EVERY:-500}"
RESUME="${RESUME:-auto}"
DRY_RUN="${DRY_RUN:-0}"
COMPILE="${COMPILE:-1}"
COMPILE_MODE="${COMPILE_MODE:-default}"

D_MODEL="${D_MODEL:-1024}"
NUM_HEADS="${NUM_HEADS:-16}"
HEAD_DIM="${HEAD_DIM:-64}"
D_FF="${D_FF:-2730}"
NONLOOP_LAYERS="${NONLOOP_LAYERS:-22}"
RESIDUAL_NUM_BLOCKS="${RESIDUAL_NUM_BLOCKS:-11}"

NUM_ENTRY_LAYERS="${NUM_ENTRY_LAYERS:-2}"
NUM_BODY_LAYERS="${NUM_BODY_LAYERS:-6}"
NUM_EXIT_LAYERS="${NUM_EXIT_LAYERS:-2}"
NUM_LOOPS="${NUM_LOOPS:-3}"

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
LM_HEAD_CHUNK_SIZE="${LM_HEAD_CHUNK_SIZE:-4096}"
CKPT_GRANULARITY="${CKPT_GRANULARITY:-per-block}"

ENTRY_ATTN_PATTERN="${ENTRY_ATTN_PATTERN:-hca,kda}"
BODY_ATTN_PATTERN="${BODY_ATTN_PATTERN:-hca,csa,kda,swa,csa,kda,csa,hca,kda,swa,csa,kda,csa,csa,kda,swa,hca,kda}"
EXIT_ATTN_PATTERN="${EXIT_ATTN_PATTERN:-csa,swa}"
HYBRID_ATTN_PATTERN="${HYBRID_ATTN_PATTERN:-hca,kda,hca,csa,kda,swa,csa,kda,csa,hca,kda,swa,csa,kda,csa,csa,kda,swa,hca,kda,csa,swa}"

NUM_SHARED_EXPERTS="${NUM_SHARED_EXPERTS:-2}"
NUM_SPARSE_EXPERTS="${NUM_SPARSE_EXPERTS:-32}"
TOP_K="${TOP_K:-6}"
ENTRY_TOP_K="${ENTRY_TOP_K:-12}"
EXIT_TOP_K="${EXIT_TOP_K:-12}"
EXPERT_D_FF="${EXPERT_D_FF:-832}"
CAPACITY_FACTOR="${CAPACITY_FACTOR:-2.0}"
MOE_DIVERSITY_FACTOR="${MOE_DIVERSITY_FACTOR:-0}"
BIAS_UPDATE_RATE="${BIAS_UPDATE_RATE:-0.02}"
ROUTER_BIAS_ERROR_CLIP="${ROUTER_BIAS_ERROR_CLIP:-1.0}"
ROUTER_BIAS_CLIP="${ROUTER_BIAS_CLIP:-1.0}"
ROUTER_LOGIT_NOISE_STD="${ROUTER_LOGIT_NOISE_STD:-0.08}"
ROUTER_LOGIT_NOISE_DECAY_STEPS="${ROUTER_LOGIT_NOISE_DECAY_STEPS:-1000}"
ROUTER_INIT_STD="${ROUTER_INIT_STD:-0.002}"
MOE_AUX_LOSS_WEIGHT="${MOE_AUX_LOSS_WEIGHT:-1e-3}"
MOE_AUX_LOSS_DECAY_STEPS="${MOE_AUX_LOSS_DECAY_STEPS:-1000}"

LR="${LR:-0.002}"
EMBED_LR="${EMBED_LR:-0.04}"
MUON_LR="${MUON_LR:-0.004}"
ROUTER_LR="${ROUTER_LR:-5e-4}"
ROUTER_WARMUP_STEPS="${ROUTER_WARMUP_STEPS:-500}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
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

shared_args=(
  --streaming
  --dataset "$DATASET"
  --dataset-config "$DATASET_CONFIG"
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
  --use-moe
  --num-shared-experts "$NUM_SHARED_EXPERTS"
  --num-sparse-experts "$NUM_SPARSE_EXPERTS"
  --top-k "$TOP_K"
  --expert-d-ff "$EXPERT_D_FF"
  --capacity-factor "$CAPACITY_FACTOR"
  --moe-diversity-factor "$MOE_DIVERSITY_FACTOR"
  --bias-update-rate "$BIAS_UPDATE_RATE"
  --router-bias-error-clip "$ROUTER_BIAS_ERROR_CLIP"
  --router-bias-clip "$ROUTER_BIAS_CLIP"
  --router-logit-noise-std "$ROUTER_LOGIT_NOISE_STD"
  --router-logit-noise-decay-steps "$ROUTER_LOGIT_NOISE_DECAY_STEPS"
  --router-init-std "$ROUTER_INIT_STD"
  --moe-aux-loss-weight "$MOE_AUX_LOSS_WEIGHT"
  --moe-aux-loss-decay-steps "$MOE_AUX_LOSS_DECAY_STEPS"
  --moe-log-every "$MOE_LOG_EVERY"
  --lm-head-chunk-size "$LM_HEAD_CHUNK_SIZE"
  --muon
  --muon-schedule-hyperparams
  --lr "$LR"
  --embed-lr "$EMBED_LR"
  --muon-lr "$MUON_LR"
  --router-lr "$ROUTER_LR"
  --router-warmup-steps "$ROUTER_WARMUP_STEPS"
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
      model_args=(--model baseline --num-layers "$NONLOOP_LAYERS")
      ;;
    residual)
      model_args=(
        --model residual
        --num-layers "$NONLOOP_LAYERS"
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
      model_args=(--model linear --num-layers "$NONLOOP_LAYERS")
      ;;
    hybrid)
      model_args=(
        --model hybrid
        --num-layers "$NONLOOP_LAYERS"
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
        --entry-top-k "$ENTRY_TOP_K"
        --exit-top-k "$EXIT_TOP_K"
        --gradient-checkpointing
        --ckpt-granularity "$CKPT_GRANULARITY"
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
    "$variant"
    "$TOTAL_TOKENS"
    "$DATASET"
    "$DATASET_CONFIG"
    "gpus-$GPUS"
  )

  echo
  echo "================================================================"
  echo "Starting W&B bakeoff run: $run_name"
  echo "  variant:      $variant"
  echo "  save_dir:     $save_dir"
  echo "  token budget: $TOTAL_TOKENS"
  echo "  per-GPU batch:$BATCH_SIZE x $MAX_LENGTH"
  echo "================================================================"

  if [[ "$DRY_RUN" == "1" ]]; then
    print_cmd "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
done
