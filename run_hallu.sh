#!/bin/bash
set -euo pipefail

# -------- conda setup --------
# Initialize conda so `conda activate` works inside the script
eval "$(conda shell.bash hook)"
conda activate hallu

# -------- configuration --------
# Evaluation models to run (your existing ones)
EVAL_MODELS=("SelfCheckBaseline" "PTrueOriginalBaseline")

# Base models (by KEY from your dictionary) under ~20B params
BASE_MODELS=(
  "qwen3-4b"
  "qwen3-8b"
  "qwen3-14b"
)

# Optional: where to store logs
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# -------- run all combos --------
for eval_model in "${EVAL_MODELS[@]}"; do
  for base_key in "${BASE_MODELS[@]}"; do
    log_file="${LOG_DIR}/${eval_model}__${base_key}.log"
    echo ">>> Running: --model=${eval_model} --base_model=${base_key}"
    # Use `tee` so you can see output live and keep a log.
    # Remove `|| true` if you want failures to stop the script.
    python hallucination.py --model="${eval_model}" --base_model="${base_key}" 2>&1 | tee "${log_file}" || true
  done
done

echo "All runs attempted. Logs in: ${LOG_DIR}/"
