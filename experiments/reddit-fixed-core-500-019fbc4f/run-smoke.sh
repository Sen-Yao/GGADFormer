#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 <gpu-index> <run-id> <expected-code-sha>" >&2
  exit 2
fi

gpu_index="$1"
run_id="$2"
expected_code_sha="$3"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
runtime_bin="$HOME/.conda/envs/DualRefGAD/bin"

export CUDA_VISIBLE_DEVICES="$gpu_index"
export EXPECTED_CODE_SHA="$expected_code_sha"
export EXECUTION_HOST="HCCS-85"
export WANDB_RUN_ID="$run_id"
export WANDB_NAME="reddit-fixed-core-smoke-$run_id"
export WANDB_RUN_GROUP="reddit-fixed-core-500-019fbc4f"
export WANDB_JOB_TYPE="smoke"
export WANDB_DIR="$HOME/wandb/reddit-fixed-core-500-019fbc4f"
export PATH="$runtime_bin:$PATH"
mkdir -p "$WANDB_DIR"

exec python "$repo_root/experiments/reddit-fixed-core-500-019fbc4f/run-trial.py" \
  --phase smoke --config-id smoke-control --seed 0
