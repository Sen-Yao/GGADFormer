#!/usr/bin/env bash
set -u

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <gpu-index> <sweep-id> <trial-count>" >&2
  exit 64
fi

gpu_index="$1"
sweep_id="$2"
trial_count="$3"
expected_sweep_id="l6ubfjxt"

case "$gpu_index" in
  0|1) expected_trial_count=3 ;;
  2|7) expected_trial_count=2 ;;
  *)
    echo "unsupported GPU index: $gpu_index" >&2
    exit 64
    ;;
esac

if [[ "$trial_count" != "$expected_trial_count" ]]; then
  echo "unexpected trial count for GPU $gpu_index: $trial_count" >&2
  exit 64
fi

if [[ "$expected_sweep_id" != "PENDING_SWEEP_ID" && "$sweep_id" != "$expected_sweep_id" ]]; then
  echo "unexpected sweep ID: $sweep_id" >&2
  exit 64
fi

task_root="/root/gpufree-data/linziyao/VecGAD-elliptic-loss-unification-019fb305"
worktree="$task_root/worktree"
environment="/root/gpufree-data/linziyao/.conda/envs/VecGAD-28bce1a8"
log_path="$task_root/logs/agent-gpu${gpu_index}-$(date -u +%Y%m%dT%H%M%SZ).log"
start_path="${log_path%.log}.start-utc"
finish_path="${log_path%.log}.finish-utc"
exit_path="${log_path%.log}.exitcode"

export PATH="$environment/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CODEX_THREAD_ID="019fb305-c6f4-77e0-ac85-5dcaa2053a47"
export SOURCE_THREAD_ID="019fad7c-281f-7463-b248-dde17f1677ab"
export PROTOCOL_ID="elliptic-loss-unification-019fb305"
export EXECUTION_HOST="HCCS-90"
export CODE_SHA="$(git -C "$worktree" rev-parse HEAD 2>/dev/null || printf unknown)"
export GPU_INDEX="$gpu_index"
export CUDA_VISIBLE_DEVICES="$gpu_index"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_ENTITY="HCCS"
export WANDB_PROJECT="GGADFormer"
export WANDB_DISABLE_CODE=true
export WANDB_CONSOLE=off
export WANDB_DIR="$task_root/wandb"
export WANDB_CACHE_DIR="$task_root/wandb-cache"
export WANDB_DATA_DIR="$task_root/wandb-data"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$task_root/mplconfig"

mkdir -p "$task_root/logs" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$MPLCONFIGDIR"
cd "$worktree" || exit 72

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "$started_at" > "$start_path"
printf 'agent_started_utc=%s gpu=%s sweep=%s code_sha=%s protocol=%s\n' \
  "$started_at" "$gpu_index" "$sweep_id" "$(git rev-parse HEAD)" "$PROTOCOL_ID" | tee "$log_path"

set -o pipefail
wandb agent "HCCS/GGADFormer/$sweep_id" --count "$trial_count" 2>&1 | tee -a "$log_path"
agent_exit_code="${PIPESTATUS[0]}"

finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "$agent_exit_code" > "$exit_path"
printf '%s\n' "$finished_at" > "$finish_path"
printf 'agent_finished_utc=%s gpu=%s sweep=%s exit_code=%s\n' \
  "$finished_at" "$gpu_index" "$sweep_id" "$agent_exit_code" | tee -a "$log_path"

exit "$agent_exit_code"
