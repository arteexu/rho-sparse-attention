#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/aws_tinyllama_rho_1b}"
STRATEGY="${STRATEGY:-slm}"
MODEL_NAME="${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T}"
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME:-TinyLlama/TinyLlama_v1.1_math_code}"
DATASET_NAME="${DATASET_NAME:-open-web-math/open-web-math}"
MAX_STEPS="${MAX_STEPS:-2000}"
SEQ_LEN="${SEQ_LEN:-2048}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LR="${LR:-2e-5}"
SAVE_EVERY="${SAVE_EVERY:-250}"
EVAL_EVERY="${EVAL_EVERY:-100}"
EVAL_BATCHES="${EVAL_BATCHES:-8}"
SELECT_RATIO_SCHEDULE="${SELECT_RATIO_SCHEDULE:-0:0.6,800:0.8,1400:1.0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"

mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
  EXTRA_ARGS+=(--attn-implementation "$ATTN_IMPLEMENTATION")
fi

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" rho_hf_pretrain.py \
  --model-name "$MODEL_NAME" \
  --reference-model-name "$REFERENCE_MODEL_NAME" \
  --dataset-name "$DATASET_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --strategy "$STRATEGY" \
  --seq-len "$SEQ_LEN" \
  --micro-batch-size "$MICRO_BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --max-steps "$MAX_STEPS" \
  --warmup-steps 100 \
  --slm-warmup-steps 20 \
  --lr "$LR" \
  --dtype bf16 \
  --gradient-checkpointing \
  --save-every "$SAVE_EVERY" \
  --eval-every "$EVAL_EVERY" \
  --eval-batches "$EVAL_BATCHES" \
  --select-ratio-schedule "$SELECT_RATIO_SCHEDULE" \
  --resume \
  "${EXTRA_ARGS[@]}"
