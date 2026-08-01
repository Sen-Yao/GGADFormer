#!/usr/bin/env bash
set -u

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <gpu-index> <sweep-id> <count>" >&2
  exit 64
fi

gpu_index="$1"
sweep_id="$2"
count="$3"

case "$gpu_index" in
  0|1|2|3|4|5|6|7) ;;
  *)
    echo "unsupported GPU index: $gpu_index" >&2
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

task_root="/root/gpufree-data/linziyao/VecGAD-tolokers-direction-mag-lrec01-019fbba0"
worktree="$task_root/worktree"
environment="/root/gpufree-data/linziyao/.conda/envs/DualRefGAD"
log_dir="$task_root/logs"
log_path="$log_dir/agent-gpu${gpu_index}.log"
start_path="$log_dir/agent-gpu${gpu_index}.start-utc"
finish_path="$log_dir/agent-gpu${gpu_index}.finish-utc"
exit_path="$log_dir/agent-gpu${gpu_index}.exitcode"

export PATH="$environment/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CODEX_THREAD_ID="019fbba0-51fe-7952-8bc7-6c68f4e18858"
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
printf 'agent_started_utc=%s gpu=%s sweep=%s count=%s execution_sha=%s code_sha=%s\n' \
  "$started_at" "$gpu_index" "$sweep_id" "$count" "$(git rev-parse HEAD)" \
  "fdb150b7927f26f2e8b5270365a324d844dc8b98" | tee "$log_path"

set -o pipefail
wandb agent "HCCS/GGADFormer/$sweep_id" --count "$count" 2>&1 | tee -a "$log_path"
agent_exit_code="${PIPESTATUS[0]}"

finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "$agent_exit_code" > "$exit_path"
printf '%s\n' "$finished_at" > "$finish_path"
printf 'agent_finished_utc=%s gpu=%s sweep=%s count=%s exit_code=%s\n' \
  "$finished_at" "$gpu_index" "$sweep_id" "$count" "$agent_exit_code" | tee -a "$log_path"

exit "$agent_exit_code"

