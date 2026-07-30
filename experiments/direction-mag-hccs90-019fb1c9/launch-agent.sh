#!/usr/bin/env bash
set -u

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <gpu-index> <dataset> <sweep-id> <count>" >&2
  exit 64
fi

gpu_index="$1"
dataset="$2"
sweep_id="$3"
count="$4"

case "$gpu_index" in
  0|1|2|3) ;;
  *)
    echo "unsupported GPU index for this protocol: $gpu_index" >&2
    exit 64
    ;;
esac

case "$dataset" in
  Amazon|elliptic|tolokers) ;;
  *)
    echo "unsupported dataset: $dataset" >&2
    exit 64
    ;;
esac

case "$count" in
  ''|*[!0-9]*)
    echo "count must be a positive integer" >&2
    exit 64
    ;;
esac

if [[ "$count" -le 0 ]]; then
  echo "count must be a positive integer" >&2
  exit 64
fi

task_root="/root/gpufree-data/linziyao/VecGAD-direction-mag-hccs90-019fb1c9"
worktree="$task_root/worktree-a5aaa858"
environment="/root/gpufree-data/linziyao/.conda/envs/VecGAD-28bce1a8"
log_dir="$task_root/logs/$dataset"
log_path="$log_dir/agent-${dataset}-gpu${gpu_index}.log"
start_path="$log_dir/agent-${dataset}-gpu${gpu_index}.start-utc"
finish_path="$log_dir/agent-${dataset}-gpu${gpu_index}.finish-utc"
exit_path="$log_dir/agent-${dataset}-gpu${gpu_index}.exitcode"

export PATH="$environment/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CODEX_THREAD_ID="019fb1c9-fd57-7962-ac54-d864531504ef"
export CUDA_VISIBLE_DEVICES="$gpu_index"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_DISABLE_CODE=true
export WANDB_CONSOLE=off
export WANDB_DIR="$task_root/wandb"
export WANDB_CACHE_DIR="$task_root/wandb-cache"
export WANDB_DATA_DIR="$task_root/wandb-data"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$task_root/mplconfig"

mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$MPLCONFIGDIR" "$log_dir"
cd "$worktree" || exit 72

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "$started_at" > "$start_path"
printf 'agent_started_utc=%s dataset=%s gpu=%s sweep=%s count=%s code_sha=%s\n' \
  "$started_at" "$dataset" "$gpu_index" "$sweep_id" "$count" "$(git rev-parse HEAD)" | tee "$log_path"

set -o pipefail
wandb agent "HCCS/GGADFormer/$sweep_id" --count "$count" 2>&1 | tee -a "$log_path"
agent_exit_code="${PIPESTATUS[0]}"

finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "$agent_exit_code" > "$exit_path"
printf '%s\n' "$finished_at" > "$finish_path"
printf 'agent_finished_utc=%s dataset=%s gpu=%s sweep=%s count=%s exit_code=%s\n' \
  "$finished_at" "$dataset" "$gpu_index" "$sweep_id" "$count" "$agent_exit_code" | tee -a "$log_path"

exit "$agent_exit_code"

