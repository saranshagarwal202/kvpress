#!/bin/bash
# Evaluate CAMPress vs DecodingPress, and benchmark Triton vs Torch merge kernels.

set -euo pipefail

# ===== Configuration =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAM_PRESS_FILE="${SCRIPT_DIR}/../kvpress/presses/cam_press.py"
RESULTS_DIR="${SCRIPT_DIR}/../results"

DATASET="ruler"
DATA_DIR="4096"
MODEL="Qwen/Qwen3-8B"
LOG_LEVEL="DEBUG"
FRACTION="0.5"
MODEL_KWARGS="{'attn_implementation': 'eager'}"

# Press names for Part 1
PART1_PRESSES=("cam_knorm" "cam_streaming_llm" "decoding_knorm" "decoding_streaming_llm")
PART1_INTERVAL="4"

# CAM press names for Part 2 (triton vs torch benchmark)
CAM_PRESSES=("cam_knorm")
INTERVALS=(2 4 8 16 32 64)

# ===== Cleanup trap: always restore use_triton to True =====
restore_use_triton() {
    echo "Restoring use_triton to True in ${CAM_PRESS_FILE}..."
    sed -i 's/use_triton: bool = .*/use_triton: bool = True/' "${CAM_PRESS_FILE}"
    echo "Restored."
}
trap restore_use_triton EXIT

# Create results directory if needed
mkdir -p "${RESULTS_DIR}"

# ===== Helper: run a single evaluation and capture all output =====
run_eval() {
    local press_name="$1"
    local interval="$2"
    local output_file="$3"

    echo "========================================"
    echo "Running: press=${press_name} interval=${interval}"
    echo "Output:  ${output_file}"
    echo "========================================"

    local start_time=$SECONDS

    script -q -e -c "python ${SCRIPT_DIR}/evaluate.py \
        --dataset ${DATASET} \
        --data_dir ${DATA_DIR} \
        --model ${MODEL} \
        --press_name ${press_name} \
        --compression_interval ${interval} \
        --fraction ${FRACTION} \
        --log_level ${LOG_LEVEL} \
        --model_kwargs \"${MODEL_KWARGS}\"" \
        "${output_file}"

    local elapsed=$(( SECONDS - start_time ))
    echo "Wall time: ${elapsed}s for press=${press_name} interval=${interval}" | tee -a "${output_file}"
}

# ===== Part 1: Basic evaluation with 4 press names =====
echo "===== Part 1: Basic evaluation ====="
for press in "${PART1_PRESSES[@]}"; do
    OUTPUT_FILE="${RESULTS_DIR}/output_${press}_interval_${PART1_INTERVAL}.txt"
    run_eval "${press}" "${PART1_INTERVAL}" "${OUTPUT_FILE}"
done

# ===== Part 2: Triton vs Torch benchmark =====
echo "===== Part 2: Triton vs Torch benchmark ====="

for use_triton_val in "True" "False"; do
    echo "Setting use_triton to ${use_triton_val}..."
    sed -i "s/use_triton: bool = .*/use_triton: bool = ${use_triton_val}/" "${CAM_PRESS_FILE}"

    if [ "${use_triton_val}" = "True" ]; then
        triton_label="triton_true"
    else
        triton_label="triton_false"
    fi

    for press in "${CAM_PRESSES[@]}"; do
        for interval in "${INTERVALS[@]}"; do
            OUTPUT_FILE="${RESULTS_DIR}/output_${press}_${triton_label}_interval_${interval}.txt"
            run_eval "${press}" "${interval}" "${OUTPUT_FILE}"
        done
    done
done

# Trap will restore use_triton=True on exit
echo "===== All evaluations completed. ====="
