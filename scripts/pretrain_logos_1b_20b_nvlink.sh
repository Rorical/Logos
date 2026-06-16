#!/usr/bin/env bash
set -euo pipefail

# Single-node NVLink DDP pretraining for the 1B Logos config.
# Usage: ./scripts/pretrain_logos_1b_20b_nvlink.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if command -v uv >/dev/null 2>&1; then
  UV=(uv)
else
  echo "uv not found; installing uv with ${PYTHON_BIN} -m pip ..."
  "${PYTHON_BIN}" -m pip install --user -q -U uv
  UV=("${PYTHON_BIN}" -m uv)
fi

if [[ "${SKIP_UV_SYNC:-0}" != "1" ]]; then
  echo "Syncing project dependencies, including W&B support ..."
  "${UV[@]}" sync --extra wandb
fi

detect_visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local csv="${CUDA_VISIBLE_DEVICES}"
    csv="${csv// /}"
    if [[ -z "$csv" || "$csv" == "-1" ]]; then
      printf '0\n'
      return
    fi
    local old_ifs="$IFS"
    IFS=',' read -r -a visible <<< "$csv"
    IFS="$old_ifs"
    printf '%s\n' "${#visible[@]}"
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d '[:space:]'
    printf '\n'
    return
  fi

  printf '0\n'
}

DETECTED_GPUS="$(detect_visible_gpus)"
GPUS="${GPUS:-4}"
if ! [[ "$GPUS" =~ ^[0-9]+$ ]]; then
  echo "GPUS must be an integer, got: $GPUS" >&2
  exit 2
fi
if [[ "$GPUS" == "0" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    GPUS=8
  else
    echo "No visible NVIDIA GPUs detected. Set CUDA_VISIBLE_DEVICES or run on a GPU node." >&2
    exit 1
  fi
fi
if [[ "$DRY_RUN" != "1" && "$GPUS" -lt 2 ]]; then
  echo "This script is for multi-GPU DDP; detected GPUS=$GPUS. Set GPUS>=2 or use a single-GPU script." >&2
  exit 1
fi
if [[ "$DRY_RUN" != "1" && "$DETECTED_GPUS" =~ ^[0-9]+$ && "$DETECTED_GPUS" -gt 0 && "$GPUS" -gt "$DETECTED_GPUS" ]]; then
  echo "Requested GPUS=$GPUS but only $DETECTED_GPUS GPUs are visible." >&2
  exit 1
fi

if [[ "$DRY_RUN" != "1" && "${REQUIRE_NVLINK:-1}" == "1" ]]; then
  topo="$(nvidia-smi topo -m 2>/dev/null || true)"
  if [[ -n "$topo" && "$topo" != *NV* ]]; then
    echo "No NVLink connection was reported by nvidia-smi topo -m." >&2
    echo "Set REQUIRE_NVLINK=0 to run anyway." >&2
    exit 1
  fi
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/.torchinductor-cache}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"

TOTAL_TOKENS="${TOTAL_TOKENS:-20B}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
TST_BAG_SIZE="${TST_BAG_SIZE:-4}"
TST_RATIO="${TST_RATIO:-0.3}"

DATASET="${DATASET:-HuggingFaceFW/fineweb-edu}"
DATASET_CONFIG="${DATASET_CONFIG:-sample-100BT}"
TEXT_COLUMN="${TEXT_COLUMN:-text}"
TOKENIZER="${TOKENIZER:-cl100k_base}"
VAL_DOCS="${VAL_DOCS:-200}"
STREAM_SHUFFLE_BUFFER="${STREAM_SHUFFLE_BUFFER:-10000}"

RUN_NAME="${RUN_NAME:-logos-1b-20b-tst-nvlink-${GPUS}gpu}"
SAVE_DIR="${SAVE_DIR:-runs/logos-pretrain/${RUN_NAME}}"
RESUME="${RESUME:-auto}"
KEEP_LAST_N="${KEEP_LAST_N:-5}"

WANDB_PROJECT="${WANDB_PROJECT:-logos-pretrain}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_MODE

NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-8}"
LOG_EVERY="${LOG_EVERY:-50}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
SAMPLE_EVERY="${SAMPLE_EVERY:-20000}"
DIAGNOSTIC_EVERY="${DIAGNOSTIC_EVERY:-100}"
MOE_LOG_EVERY="${MOE_LOG_EVERY:-1000}"
OPT_STATE_LOG_EVERY="${OPT_STATE_LOG_EVERY:-1000}"

D_MODEL="${D_MODEL:-1024}"
NUM_HEADS="${NUM_HEADS:-16}"
HEAD_DIM="${HEAD_DIM:-64}"
D_FF="${D_FF:-2730}"
NUM_ENTRY_LAYERS="${NUM_ENTRY_LAYERS:-2}"
NUM_BODY_LAYERS="${NUM_BODY_LAYERS:-6}"
NUM_EXIT_LAYERS="${NUM_EXIT_LAYERS:-2}"
NUM_LOOPS="${NUM_LOOPS:-3}"

CHUNK_SIZE="${CHUNK_SIZE:-128}"
CONV_SIZE="${CONV_SIZE:-4}"
ENTRY_ATTN_PATTERN="${ENTRY_ATTN_PATTERN:-hca,kda}"
BODY_ATTN_PATTERN="${BODY_ATTN_PATTERN:-hca,csa,kda,swa,csa,kda,csa,hca,kda,swa,csa,kda,csa,csa,kda,swa,hca,kda}"
EXIT_ATTN_PATTERN="${EXIT_ATTN_PATTERN:-csa,swa}"
CSA_COMPRESSION="${CSA_COMPRESSION:-4}"
CSA_TOP_K="${CSA_TOP_K:-64}"
CSA_INDEXER_HEADS="${CSA_INDEXER_HEADS:-4}"
CSA_INDEXER_DIM="${CSA_INDEXER_DIM:-32}"
HCA_COMPRESSION="${HCA_COMPRESSION:-128}"
SWA_WINDOW="${SWA_WINDOW:-256}"
SWA_EVERY="${SWA_EVERY:-4}"
SWA_OFFSET="${SWA_OFFSET:-3}"

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
ROUTER_LOGIT_NOISE_DECAY_STEPS="${ROUTER_LOGIT_NOISE_DECAY_STEPS:-8000}"
ROUTER_INIT_STD="${ROUTER_INIT_STD:-0.002}"
MOE_AUX_LOSS_WEIGHT="${MOE_AUX_LOSS_WEIGHT:-1e-3}"
MOE_AUX_LOSS_DECAY_STEPS="${MOE_AUX_LOSS_DECAY_STEPS:-8000}"

LM_HEAD_CHUNK_SIZE="${LM_HEAD_CHUNK_SIZE:-4096}"
CKPT_GRANULARITY="${CKPT_GRANULARITY:-per-block}"
COMPILE_MODE="${COMPILE_MODE:-default}"
DDP_BUCKET_CAP_MB="${DDP_BUCKET_CAP_MB:-128}"

LR="${LR:-0.002}"
EMBED_LR="${EMBED_LR:-0.04}"
MUON_LR="${MUON_LR:-0.004}"
ROUTER_LR="${ROUTER_LR:-5e-4}"
ROUTER_WARMUP_STEPS="${ROUTER_WARMUP_STEPS:-8000}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
WARMUP_STEPS="${WARMUP_STEPS:-8000}"
DECAY_STEPS="${DECAY_STEPS:-70000}"
DECAY_FRAC="${DECAY_FRAC:-0.2}"
EMA_DECAY="${EMA_DECAY:-0.0}"
SEED="${SEED:-42}"

SAMPLE_PROMPT="${SAMPLE_PROMPT:-In a recent study, researchers found that}"
SAMPLE_MAX_TOKENS="${SAMPLE_MAX_TOKENS:-80}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-0.8}"

train_args=(
  --model logos
  --streaming
  --dataset "$DATASET"
  --dataset-config "$DATASET_CONFIG"
  --text-column "$TEXT_COLUMN"
  --val-docs "$VAL_DOCS"
  --stream-shuffle-buffer "$STREAM_SHUFFLE_BUFFER"
  --tiktoken-encoding "$TOKENIZER"
  --total-tokens "$TOTAL_TOKENS"
  --tst-bag-size "$TST_BAG_SIZE"
  --tst-ratio "$TST_RATIO"
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
  --compile
  --compile-mode "$COMPILE_MODE"
  --ddp-static-graph
  --ddp-gradient-as-bucket-view
  --ddp-broadcast-buffers
  --ddp-bucket-cap-mb "$DDP_BUCKET_CAP_MB"
  --gradient-checkpointing
  --ckpt-granularity "$CKPT_GRANULARITY"
  --diagnostic-every "$DIAGNOSTIC_EVERY"
  --d-model "$D_MODEL"
  --num-heads "$NUM_HEADS"
  --head-dim "$HEAD_DIM"
  --d-ff "$D_FF"
  --num-entry-layers "$NUM_ENTRY_LAYERS"
  --num-body-layers "$NUM_BODY_LAYERS"
  --num-exit-layers "$NUM_EXIT_LAYERS"
  --num-loops "$NUM_LOOPS"
  --chunk-size "$CHUNK_SIZE"
  --conv-size "$CONV_SIZE"
  --entry-attn-pattern "$ENTRY_ATTN_PATTERN"
  --body-attn-pattern "$BODY_ATTN_PATTERN"
  --exit-attn-pattern "$EXIT_ATTN_PATTERN"
  --csa-compression "$CSA_COMPRESSION"
  --csa-top-k "$CSA_TOP_K"
  --csa-indexer-heads "$CSA_INDEXER_HEADS"
  --csa-indexer-dim "$CSA_INDEXER_DIM"
  --hca-compression "$HCA_COMPRESSION"
  --swa-window "$SWA_WINDOW"
  --swa-every "$SWA_EVERY"
  --swa-offset "$SWA_OFFSET"
  --use-moe
  --num-shared-experts "$NUM_SHARED_EXPERTS"
  --num-sparse-experts "$NUM_SPARSE_EXPERTS"
  --top-k "$TOP_K"
  --entry-top-k "$ENTRY_TOP_K"
  --exit-top-k "$EXIT_TOP_K"
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
  --decay-steps "$DECAY_STEPS"
  --decay-frac "$DECAY_FRAC"
  --opt-state-log-every "$OPT_STATE_LOG_EVERY"
  --ema-decay "$EMA_DECAY"
  --sample-prompt "$SAMPLE_PROMPT"
  --sample-max-tokens "$SAMPLE_MAX_TOKENS"
  --sample-temperature "$SAMPLE_TEMPERATURE"
  --save-dir "$SAVE_DIR"
  --resume "$RESUME"
  --seed "$SEED"
  --wandb
  --wandb-project "$WANDB_PROJECT"
  --wandb-mode "$WANDB_MODE"
  --wandb-run-name "$RUN_NAME"
)

if [[ -n "$WANDB_ENTITY" ]]; then
  train_args+=(--wandb-entity "$WANDB_ENTITY")
fi

train_args+=(
  --wandb-tags
  logos
  pretrain
  1b
  20b-tokens
  tst
  "tst-bag-$TST_BAG_SIZE"
  ddp
  nvlink
  "gpus-$GPUS"
)

cmd=(
  "${UV[@]}"
  run
  torchrun
  --standalone
  --nproc_per_node "$GPUS"
  scripts/train.py
  "${train_args[@]}"
)

echo
echo "================================================================"
echo "Logos 1B 20B-token NVLink DDP pretraining"
echo "  GPUs:              $GPUS"
echo "  per-GPU batch:     $BATCH_SIZE x $MAX_LENGTH"
echo "  total tokens:      $TOTAL_TOKENS"
echo "  TST:               bag_size=$TST_BAG_SIZE ratio=$TST_RATIO"
echo "  compile mode:      $COMPILE_MODE"
echo "  DDP bucket MB:     $DDP_BUCKET_CAP_MB"
echo "  save dir:          $SAVE_DIR"
echo "  wandb:             $WANDB_PROJECT / $RUN_NAME ($WANDB_MODE)"
echo "================================================================"

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

if [[ "$WANDB_MODE" == "online" && -n "${WANDB_API_KEY:-}" && "${SKIP_WANDB_LOGIN:-0}" != "1" ]]; then
  "${UV[@]}" run wandb login --relogin "$WANDB_API_KEY"
elif [[ "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set; relying on an existing wandb login." >&2
fi

nvidia-smi
nvidia-smi topo -m || true

"${cmd[@]}"
