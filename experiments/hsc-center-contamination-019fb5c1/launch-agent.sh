#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 <gpu-index> <trial-count> <sweep-id> <code-sha> <worktree> <task-root>" >&2
  exit 64
fi

gpu_index="$1"
trial_count="$2"
sweep_id="$3"
expected_code_sha="$4"
worktree="$5"
task_root="$6"

[[ "$gpu_index" =~ ^[0-7]$ ]] || { echo "invalid GPU index" >&2; exit 65; }
[[ "$trial_count" =~ ^[1-9][0-9]*$ ]] || { echo "invalid trial count" >&2; exit 66; }
[[ "$sweep_id" =~ ^[a-z0-9]+$ ]] || { echo "invalid sweep ID" >&2; exit 67; }
[[ "$expected_code_sha" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid code SHA" >&2; exit 68; }
[[ -d "$worktree/.git" || -f "$worktree/.git" ]] || { echo "missing worktree" >&2; exit 69; }

actual_code_sha="$(git -C "$worktree" rev-parse HEAD)"
[[ "$actual_code_sha" == "$expected_code_sha" ]] || { echo "execution SHA mismatch" >&2; exit 70; }
[[ -z "$(git -C "$worktree" status --short)" ]] || { echo "execution worktree is dirty" >&2; exit 71; }

runtime_bin="/root/gpufree-data/linziyao/.conda/envs/DualRefGAD/bin"
[[ -x "$runtime_bin/python" && -x "$runtime_bin/wandb" ]] || { echo "runtime is incomplete" >&2; exit 72; }

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_path="$task_root/logs/agent-gpu${gpu_index}-${timestamp}.log"
start_path="${log_path%.log}.start-utc"
finish_path="${log_path%.log}.finish-utc"
exit_path="${log_path%.log}.exitcode"

export PATH="$runtime_bin:$PATH"
export CUDA_VISIBLE_DEVICES="$gpu_index"
export GPU_INDEX="$gpu_index"
export CODE_SHA="$expected_code_sha"
export PROTOCOL_ID="hsc-center-contamination-019fb5c1-v1"
export EXECUTION_HOST="HCCS-85"
export WANDB_ENTITY="HCCS"
export WANDB_PROJECT="GGADFormer"
export WANDB_DISABLE_CODE="true"
export WANDB_CONSOLE="off"
export WANDB_DIR="$task_root/wandb"
export WANDB_CACHE_DIR="$task_root/wandb-cache"
export WANDB_DATA_DIR="$task_root/wandb-data"
export CHECKPOINT_DIR="$task_root/checkpoints"
export DIAGNOSTIC_DIR="$task_root/diagnostics"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED="0"
export MPLBACKEND="Agg"
export MPLCONFIGDIR="$task_root/mplconfig"

mkdir -p \
  "$task_root/logs" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$WANDB_DATA_DIR" \
  "$CHECKPOINT_DIR" \
  "$DIAGNOSTIC_DIR" \
  "$MPLCONFIGDIR"
cd "$worktree"

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "$started_at" > "$start_path"
printf 'agent_started_utc=%s gpu=%s sweep=%s code_sha=%s protocol=%s\n' \
  "$started_at" "$gpu_index" "$sweep_id" "$actual_code_sha" "$PROTOCOL_ID" | tee "$log_path"

set +e
set -o pipefail
wandb agent "HCCS/GGADFormer/$sweep_id" --count "$trial_count" 2>&1 | tee -a "$log_path"
agent_exit_code="${PIPESTATUS[0]}"
set -e

finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "$agent_exit_code" > "$exit_path"
printf '%s\n' "$finished_at" > "$finish_path"
printf 'agent_finished_utc=%s gpu=%s sweep=%s exit_code=%s\n' \
  "$finished_at" "$gpu_index" "$sweep_id" "$agent_exit_code" | tee -a "$log_path"

exit "$agent_exit_code"
