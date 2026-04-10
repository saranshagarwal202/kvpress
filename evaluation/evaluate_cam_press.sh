#!/bin/bash
# Evaluate CAMPress vs DecodingPress

set -euo pipefail

# ===== Configuration =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results"

DATASET="ruler"
DATA_DIR="4096"
MODEL="Qwen/Qwen3-8B"
LOG_LEVEL="DEBUG"
FRACTION="0.5"
COMPRESSION_INTERVAL="4"
MODEL_KWARGS="{'attn_implementation': 'eager'}"

PRESSES=("cam_knorm")

# Create results directory if needed
mkdir -p "${RESULTS_DIR}"

# ===== Helper: run a single evaluation and capture all output =====
run_eval() {
    local press_name="$1"
    local output_file="$2"

    echo "========================================"
    echo "Running: press=${press_name} interval=${COMPRESSION_INTERVAL}"
    echo "Output:  ${output_file}"
    echo "========================================"

    local start_time=$SECONDS

    script -q -e -c "python ${SCRIPT_DIR}/evaluate.py \
        --dataset ${DATASET} \
        --data_dir ${DATA_DIR} \
        --model ${MODEL} \
        --press_name ${press_name} \
        --compression_interval ${COMPRESSION_INTERVAL} \
        --fraction ${FRACTION} \
        --log_level ${LOG_LEVEL} \
        --model_kwargs \"${MODEL_KWARGS}\"" \
        "${output_file}"

    local elapsed=$(( SECONDS - start_time ))
    echo "Wall time: ${elapsed}s for press=${press_name}" | tee -a "${output_file}"
}

# ===== Run evaluation for all presses =====
echo "===== Evaluating CAMPress and DecodingPress ====="
for press in "${PRESSES[@]}"; do
    OUTPUT_FILE="${RESULTS_DIR}/output_${press}_interval_${COMPRESSION_INTERVAL}.txt"
    run_eval "${press}" "${OUTPUT_FILE}"
done

echo "===== All evaluations completed. ====="
