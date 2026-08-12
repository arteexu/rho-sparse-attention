#!/usr/bin/env bash
set -euo pipefail

# This script is meant to run inside the edu-llm capacity-block container.
# It is launched by torchrun, either by .edullm/run.yaml on the single-node
# workflow or by block-run-distributed.yml on one or more nodes.

RUN_ID="${EDULLM_RUN_ID:-rho-block-12h}"
LOCAL_RANK_ID="${LOCAL_RANK:-0}"
GLOBAL_RANK_ID="${RANK:-0}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export HF_HOME="${HF_HOME:-/work/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

DEPS_SENTINEL="/tmp/rho_deps_ready_${RUN_ID}"
if [[ "${LOCAL_RANK_ID}" == "0" ]]; then
  python -m pip install --upgrade pip
  python -m pip install -e .
  touch "${DEPS_SENTINEL}"
else
  for _attempt in $(seq 1 1800); do
    [[ -f "${DEPS_SENTINEL}" ]] && break
    sleep 2
  done
  [[ -f "${DEPS_SENTINEL}" ]] || {
    echo "Timed out waiting for dependency install on local rank 0" >&2
    exit 1
  }
fi

MODEL_NAME="${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T}"
REFERENCE_MODEL_NAME="${REFERENCE_MODEL_NAME:-TinyLlama/TinyLlama_v1.1_math_code}"
DATASET_NAME="${DATASET_NAME:-open-web-math/open-web-math}"
DATASET_CONFIG="${DATASET_CONFIG:-}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
TEXT_FIELD="${TEXT_FIELD:-text}"
SEQ_LEN="${SEQ_LEN:-2048}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_STEPS="${MAX_STEPS:-900}"
LR="${LR:-2e-5}"
SAVE_EVERY="${SAVE_EVERY:-150}"
EVAL_EVERY="${EVAL_EVERY:-75}"
EVAL_BATCHES="${EVAL_BATCHES:-8}"
LOG_EVERY="${LOG_EVERY:-10}"
WARMUP_STEPS="${WARMUP_STEPS:-75}"
SLM_WARMUP_STEPS="${SLM_WARMUP_STEPS:-20}"
SELECT_RATIO_SCHEDULE="${SELECT_RATIO_SCHEDULE:-0:0.6,300:0.8,650:1.0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
EXPLICIT_STRATEGIES="${STRATEGIES:-}"
if [[ -n "${STRATEGY:-}" && -z "${EXPLICIT_STRATEGIES}" ]]; then
  STRATEGIES="${STRATEGY}"
else
  STRATEGIES="${EXPLICIT_STRATEGIES:-slm}"
fi
MAX_RUNTIME_HOURS_PER_STRATEGY="${MAX_RUNTIME_HOURS_PER_STRATEGY:-3.5}"

RUN_ROOT="${RUN_ROOT:-/work/runs/${RUN_ID}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/work/log/artifacts}"
mkdir -p "${RUN_ROOT}" "${ARTIFACT_ROOT}"

IFS=',' read -r -a RAW_STRATEGY_LIST <<< "${STRATEGIES}"
STRATEGY_LIST=()
for raw_strategy in "${RAW_STRATEGY_LIST[@]}"; do
  strategy="$(echo "${raw_strategy}" | tr -d '[:space:]')"
  [[ -n "${strategy}" ]] && STRATEGY_LIST+=("${strategy}")
done

if [[ "${#STRATEGY_LIST[@]}" == "0" ]]; then
  echo "No strategy requested. Set STRATEGY=slm or STRATEGIES=slm." >&2
  exit 64
fi

if [[ "${#STRATEGY_LIST[@]}" != "1" && "${ALLOW_MULTI_STRATEGY_UNDER_TORCHRUN:-false}" != "true" ]]; then
  if [[ -n "${RANK:-}" || -n "${LOCAL_RANK:-}" || -n "${WORLD_SIZE:-}" ]]; then
    echo "Multiple strategies inside one distributed torchrun can fail during DDP/NCCL reinitialization." >&2
    echo "Launch one Block run per strategy, for example STRATEGY=clm, STRATEGY=random, and STRATEGY=slm." >&2
    echo "Set ALLOW_MULTI_STRATEGY_UNDER_TORCHRUN=true only if you intentionally want the old suite behavior." >&2
    exit 64
  fi
fi

for strategy in "${STRATEGY_LIST[@]}"; do
  output_dir="${RUN_ROOT}/${strategy}"
  mirror_dir="${ARTIFACT_ROOT}/${strategy}"
  mkdir -p "${output_dir}" "${mirror_dir}"

  if [[ "${GLOBAL_RANK_ID}" == "0" ]]; then
    echo "=== RHO block run: ${strategy} ==="
    echo "run_id=${RUN_ID}"
    echo "model=${MODEL_NAME}"
    echo "reference=${REFERENCE_MODEL_NAME}"
    echo "dataset=${DATASET_NAME}/${DATASET_SPLIT}"
    echo "output_dir=${output_dir}"
    echo "mirror_dir=${mirror_dir}"
  fi

  extra_args=()
  if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
    extra_args+=(--attn-implementation "${ATTN_IMPLEMENTATION}")
  fi

  python rho_hf_pretrain.py \
    --model-name "${MODEL_NAME}" \
    --reference-model-name "${REFERENCE_MODEL_NAME}" \
    --dataset-name "${DATASET_NAME}" \
    --dataset-config "${DATASET_CONFIG}" \
    --dataset-split "${DATASET_SPLIT}" \
    --text-field "${TEXT_FIELD}" \
    --output-dir "${output_dir}" \
    --strategy "${strategy}" \
    --seq-len "${SEQ_LEN}" \
    --micro-batch-size "${MICRO_BATCH_SIZE}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --max-steps "${MAX_STEPS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --slm-warmup-steps "${SLM_WARMUP_STEPS}" \
    --lr "${LR}" \
    --dtype bf16 \
    --gradient-checkpointing \
    --save-every "${SAVE_EVERY}" \
    --eval-every "${EVAL_EVERY}" \
    --eval-batches "${EVAL_BATCHES}" \
    --log-every "${LOG_EVERY}" \
    --select-ratio-schedule "${SELECT_RATIO_SCHEDULE}" \
    --max-runtime-hours "${MAX_RUNTIME_HOURS_PER_STRATEGY}" \
    --mirror-dir "${mirror_dir}" \
    --resume \
    "${extra_args[@]}"
done
