#!/usr/bin/env bash
set -u

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <gpu-index> <sweep-id>" >&2
  exit 64
fi

gpu_index="$1"
sweep_id="$2"

case "$gpu_index" in
  0|1|2|3|4) ;;
  *)
    echo "unsupported GPU index: $gpu_index" >&2
    exit 64
    ;;
esac

if [[ "$sweep_id" != "n30dxpp2" ]]; then
  echo "unexpected sweep ID: $sweep_id" >&2
  exit 64
fi

task_root="/root/gpufree-data/linziyao/VecGAD-tfinance-rmin03-iqxjqsdl-019fabbd"
worktree="$task_root/worktree-e071ae66"
environment="/root/gpufree-data/linziyao/.conda/envs/VecGAD-28bce1a8"
log_path="$task_root/logs/agent-gpu${gpu_index}.log"
start_path="$task_root/logs/agent-gpu${gpu_index}.start-utc"
finish_path="$task_root/logs/agent-gpu${gpu_index}.finish-utc"
exit_path="$task_root/logs/agent-gpu${gpu_index}.exitcode"

export PATH="$environment/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CODEX_THREAD_ID="019fabbd-35b1-7321-951c-2106f73e2e6b"
export CUDA_VISIBLE_DEVICES="$gpu_index"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_DISABLE_CODE=true
export WANDB_CONSOLE=off
export WANDB_DIR="$task_root/wandb"
export WANDB_CACHE_DIR="$task_root/wandb-cache"
export WANDB_DATA_DIR="$task_root/wandb-data"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$task_root/mplconfig"

mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$MPLCONFIGDIR"
cd "$worktree" || exit 72

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "$started_at" > "$start_path"
printf 'agent_started_utc=%s gpu=%s sweep=%s code_sha=%s\n' \
  "$started_at" "$gpu_index" "$sweep_id" "$(git rev-parse HEAD)" | tee "$log_path"

set -o pipefail
wandb agent "HCCS/GGADFormer/$sweep_id" --count 1 2>&1 | tee -a "$log_path"
agent_exit_code="${PIPESTATUS[0]}"

finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "$agent_exit_code" > "$exit_path"
printf '%s\n' "$finished_at" > "$finish_path"
printf 'agent_finished_utc=%s gpu=%s sweep=%s exit_code=%s\n' \
  "$finished_at" "$gpu_index" "$sweep_id" "$agent_exit_code" | tee -a "$log_path"

exit "$agent_exit_code"
