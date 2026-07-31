#!/usr/bin/env bash
set -u

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <gpu-index> <sweep-id>" >&2
  exit 64
fi

gpu_index="$1"
sweep_id="$2"
expected_sweep_id="mufbddb1"

case "$gpu_index" in
  0|1|2|3) ;;
  *)
    echo "unsupported GPU index: $gpu_index" >&2
    exit 64
    ;;
esac

if [[ "$sweep_id" != "$expected_sweep_id" ]]; then
  echo "unexpected sweep ID: $sweep_id" >&2
  exit 65
fi

task_root="/root/gpufree-data/linziyao/VecGAD-elliptic-loss-mechanism-019fad7c"
worktree="$task_root/worktree"
environment="/root/gpufree-data/linziyao/.conda/envs/DualRefGAD"
expected_code_sha="65014dd11bed01b761aa7c3889c7718b7950884d"
log_path="$task_root/logs/agent-gpu${gpu_index}-$(date -u +%Y%m%dT%H%M%SZ).log"
start_path="${log_path%.log}.start-utc"
finish_path="${log_path%.log}.finish-utc"
exit_path="${log_path%.log}.exitcode"

actual_code_sha="$(git -C "$worktree" rev-parse HEAD 2>/dev/null || printf unknown)"
if [[ "$actual_code_sha" != "$expected_code_sha" ]]; then
  echo "unexpected worktree SHA: $actual_code_sha" >&2
  exit 66
fi
if [[ -n "$(git -C "$worktree" status --porcelain)" ]]; then
  echo "execution worktree is dirty" >&2
  exit 66
fi

export PATH="$environment/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CODEX_THREAD_ID="019fb720-b85a-7ae3-8390-a0ab2b2345ad"
export SOURCE_THREAD_ID="019fad7c-281f-7463-b248-dde17f1677ab"
export PROTOCOL_ID="elliptic-loss-mechanism-019fad7c"
export EXECUTION_HOST="HCCS-85"
export CODE_SHA="$expected_code_sha"
export GPU_INDEX="$gpu_index"
export CUDA_VISIBLE_DEVICES="$gpu_index"
export PYTHONDONTWRITEBYTECODE=1
export DATASET_SHA256="2f502df4b87be8f8b5ed5ef8378876125c92b06afbc5b38ee58fe4b56b1b2023"
export OPTIMIZER_UPDATES_PER_EPOCH=2
export MECHANISM_DIAGNOSTICS_DIR="$task_root/diagnostics"
export WANDB_ENTITY="HCCS"
export WANDB_PROJECT="GGADFormer"
export WANDB_DISABLE_CODE=true
export WANDB_CONSOLE=off
export WANDB_DIR="$task_root/wandb"
export WANDB_CACHE_DIR="$task_root/wandb-cache"
export WANDB_DATA_DIR="$task_root/wandb-data"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$task_root/mplconfig"

mkdir -p "$task_root/logs" "$MECHANISM_DIAGNOSTICS_DIR" "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$MPLCONFIGDIR"
cd "$worktree" || exit 72

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "$started_at" > "$start_path"
printf 'agent_started_utc=%s gpu=%s sweep=%s code_sha=%s protocol=%s\n' \
  "$started_at" "$gpu_index" "$sweep_id" "$actual_code_sha" "$PROTOCOL_ID" | tee "$log_path"

set -o pipefail
wandb agent "HCCS/GGADFormer/$sweep_id" --count 1 2>&1 | tee -a "$log_path"
agent_exit_code="${PIPESTATUS[0]}"

finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "$agent_exit_code" > "$exit_path"
printf '%s\n' "$finished_at" > "$finish_path"
printf 'agent_finished_utc=%s gpu=%s sweep=%s exit_code=%s\n' \
  "$finished_at" "$gpu_index" "$sweep_id" "$agent_exit_code" | tee -a "$log_path"

exit "$agent_exit_code"
