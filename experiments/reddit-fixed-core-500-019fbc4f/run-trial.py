#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import subprocess
import sys

from protocol import PHASE_SEEDS, PROTOCOL_ID, resolve_config, scientific_argv


def git_head(repo_root):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
    ).strip()


def tracked_worktree_changes(repo_root):
    return subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=str(repo_root),
        text=True,
    ).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=tuple(PHASE_SEEDS))
    parser.add_argument("--config-id", "--config_id", dest="config_id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    expected_sha = os.environ.get("EXPECTED_CODE_SHA", "")
    actual_sha = git_head(repo_root)
    if not expected_sha:
        raise SystemExit("EXPECTED_CODE_SHA is required")
    if actual_sha != expected_sha:
        raise SystemExit(
            "execution SHA mismatch: expected {}, observed {}".format(
                expected_sha, actual_sha
            )
        )
    changes = tracked_worktree_changes(repo_root)
    if changes:
        raise SystemExit("tracked execution worktree is dirty:\n{}".format(changes))

    config = resolve_config(args.phase, args.config_id, args.seed)
    os.environ.update({
        "PROTOCOL_ID": PROTOCOL_ID,
        "EXPERIMENT_PHASE": args.phase,
        "EXPERIMENT_CONFIG_ID": args.config_id,
        "FIXED_CORE_GUARD": "1",
        "PYTHONHASHSEED": str(args.seed),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "WANDB_ENTITY": "HCCS",
        "WANDB_PROJECT": "GGADFormer",
        "WANDB_DISABLE_CODE": "true",
        "WANDB_CONSOLE": "off",
        "WANDB_LOG_MODEL": "false",
    })
    command = [sys.executable, "run.py"] + scientific_argv(
        args.phase, args.config_id, args.seed
    )
    if args.print_command:
        print(" ".join(command))
        return
    if config["evaluation_protocol"] == "validation_only" and "frozen" in args.phase:
        raise SystemExit("validation-only phase identity is inconsistent")
    os.chdir(str(repo_root))
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
