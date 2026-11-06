#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Activate the matching Python, CANN and ATB environment before running.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash examples/spec_decode/run_dflash_asd.sh [--dry-run] TARGET_MODEL DRAFT_MODEL

Starts a DFlash + ASD HTTP server. Requests must use temperature=0.
--dry-run prints the configuration and command without loading models.

Environment overrides:
  ASD_PYTHON                              Python executable (default: python)
  ASCEND_RT_VISIBLE_DEVICES                Logical devices (default: 0,1)
  VLLM_USE_V2_MODEL_RUNNER                  Runner selection (default: 1)
  VLLM_ASCEND_ENABLE_ASD                   ASD switch (default: 1; 0 disables)
  ASD_K                                   Draft candidates (default: 15; 1..15)
  ASD_TP                                  Target TP (default: 2)
  ASD_DRAFT_TP                            Draft TP (default: ASD_TP)
  VLLM_ASCEND_ASD_RISK_BUDGET               B (default: 8)
  VLLM_ASCEND_ASD_MAX_REGRET_PER_VALUE      g (default: 0.25)
  VLLM_ASCEND_ASD_MAX_RELAXED_TOKENS_PER_BLOCK  M (default: 2)
  ASD_HOST / ASD_PORT                      HTTP bind (default: 127.0.0.1 / 8815)
  ASD_SERVED_MODEL_NAME                    API name (default: qwen3-32b-dflash-asd)
  ASD_MAX_MODEL_LEN                        Context limit (default: 4096)
  ASD_MAX_NUM_SEQS                         Scheduler capacity (default: 16)
  ASD_MAX_BATCHED_TOKENS                   Batch token limit (default: 528)
  ASD_MEMORY_UTILIZATION                  Device memory fraction (default: 0.9)
  ASD_SEED                                Server seed (default: 0)
EOF
}

asd_dry_run=0
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if [[ ${1:-} == --dry-run ]]; then
    asd_dry_run=1
    shift
fi
if [[ $# -ne 2 ]]; then
    usage >&2
    exit 2
fi

asd_target_model=$1
asd_draft_model=$2
asd_python=${ASD_PYTHON:-python}
asd_k=${ASD_K:-15}
asd_tp=${ASD_TP:-2}
asd_draft_tp=${ASD_DRAFT_TP:-$asd_tp}
if [[ ! $asd_k =~ ^[1-9][0-9]*$ ]] || (( asd_k > 15 )); then
    echo "ASD_K must be an integer from 1 to 15 for this DFlash attention path." >&2
    exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}
export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export VLLM_ASCEND_ENABLE_ASD=${VLLM_ASCEND_ENABLE_ASD:-1}
export VLLM_ASCEND_ASD_RISK_BUDGET=${VLLM_ASCEND_ASD_RISK_BUDGET:-8}
export VLLM_ASCEND_ASD_MAX_REGRET_PER_VALUE=${VLLM_ASCEND_ASD_MAX_REGRET_PER_VALUE:-0.25}
export VLLM_ASCEND_ASD_MAX_RELAXED_TOKENS_PER_BLOCK=${VLLM_ASCEND_ASD_MAX_RELAXED_TOKENS_PER_BLOCK:-2}

# JSON encoding preserves model paths containing spaces, quotes or backslashes.
asd_spec_config=$("$asd_python" - "$asd_draft_model" "$asd_k" "$asd_draft_tp" <<'PY'
import json
import sys

print(json.dumps({
    "method": "dflash",
    "model": sys.argv[1],
    "num_speculative_tokens": int(sys.argv[2]),
    "draft_tensor_parallel_size": int(sys.argv[3]),
}))
PY
)

asd_command=(
    "$asd_python" -m vllm.entrypoints.cli.main serve "$asd_target_model"
    --trust-remote-code
    --generation-config vllm
    --dtype bfloat16
    --tensor-parallel-size "$asd_tp"
    --distributed-executor-backend mp
    --gpu-memory-utilization "${ASD_MEMORY_UTILIZATION:-0.9}"
    --max-model-len "${ASD_MAX_MODEL_LEN:-4096}"
    --max-num-batched-tokens "${ASD_MAX_BATCHED_TOKENS:-528}"
    --max-num-seqs "${ASD_MAX_NUM_SEQS:-16}"
    --seed "${ASD_SEED:-0}"
    --no-enforce-eager
    --enable-prefix-caching
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --speculative-config "$asd_spec_config"
    --served-model-name "${ASD_SERVED_MODEL_NAME:-qwen3-32b-dflash-asd}"
    --host "${ASD_HOST:-127.0.0.1}"
    --port "${ASD_PORT:-8815}"
)

printf 'DFlash: ASD=%s K=%s TP=%s draft_TP=%s B=%s g=%s M=%s runner_v2=%s devices=%s\n' \
    "$VLLM_ASCEND_ENABLE_ASD" "$asd_k" "$asd_tp" "$asd_draft_tp" "$VLLM_ASCEND_ASD_RISK_BUDGET" \
    "$VLLM_ASCEND_ASD_MAX_REGRET_PER_VALUE" "$VLLM_ASCEND_ASD_MAX_RELAXED_TOKENS_PER_BLOCK" \
    "$VLLM_USE_V2_MODEL_RUNNER" "$ASCEND_RT_VISIBLE_DEVICES"
if (( asd_dry_run )); then
    asd_env_names=(
        ASCEND_RT_VISIBLE_DEVICES VLLM_USE_V1 VLLM_USE_V2_MODEL_RUNNER
        VLLM_WORKER_MULTIPROC_METHOD HCCL_OP_EXPANSION_MODE VLLM_ASCEND_ENABLE_ASD
        VLLM_ASCEND_ASD_RISK_BUDGET VLLM_ASCEND_ASD_MAX_REGRET_PER_VALUE
        VLLM_ASCEND_ASD_MAX_RELAXED_TOKENS_PER_BLOCK
    )
    printf 'env '
    for asd_env_name in "${asd_env_names[@]}"; do
        printf '%q ' "$asd_env_name=${!asd_env_name}"
    done
    printf '%q ' "${asd_command[@]}"
    printf '\n'
    exit 0
fi

exec "${asd_command[@]}"
