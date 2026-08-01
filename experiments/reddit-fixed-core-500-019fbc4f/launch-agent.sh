#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 <gpu-index> <sweep-id> <count> <expected-code-sha>" >&2
  exit 2
fi

gpu_index="$1"
sweep_id="$2"
agent_count="$3"
expected_code_sha="$4"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
task_root="$repo_root/experiments/reddit-fixed-core-500-019fbc4f"
runtime_bin="$HOME/.conda/envs/DualRefGAD/bin"

if [[ "$(git -C "$repo_root" rev-parse HEAD)" != "$expected_code_sha" ]]; then
  echo "execution SHA mismatch" >&2
  exit 3
fi
if [[ -n "$(git -C "$repo_root" status --porcelain --untracked-files=no)" ]]; then
  echo "tracked execution worktree is dirty" >&2
  exit 4
fi
if [[ ! "$gpu_index" =~ ^[0-7]$ || ! "$agent_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid GPU index or count" >&2
  exit 5
fi

export CUDA_VISIBLE_DEVICES="$gpu_index"
export EXPECTED_CODE_SHA="$expected_code_sha"
export EXECUTION_HOST="HCCS-85"
export WANDB_ENTITY="HCCS"
export WANDB_PROJECT="GGADFormer"
export WANDB_DISABLE_CODE="true"
export WANDB_CONSOLE="off"
export WANDB_LOG_MODEL="false"
export WANDB_DIR="$HOME/wandb/reddit-fixed-core-500-019fbc4f"
export PATH="$runtime_bin:$PATH"
mkdir -p "$WANDB_DIR"

exec wandb agent "HCCS/GGADFormer/$sweep_id" --count "$agent_count"
