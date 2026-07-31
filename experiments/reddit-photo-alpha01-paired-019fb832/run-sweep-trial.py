#!/usr/bin/env python
import argparse
import os
from pathlib import Path
import subprocess
import sys

from protocol import DATASETS, PROTOCOL_ID, SEEDS, VARIANTS, scientific_argv


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
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    expected_sha = os.environ.get("EXPECTED_CODE_SHA", "")
    actual_sha = git_head(repo_root)
    if expected_sha and actual_sha != expected_sha:
        raise SystemExit(
            "execution SHA mismatch: expected {}, observed {}".format(
                expected_sha, actual_sha
            )
        )
    changes = tracked_worktree_changes(repo_root)
    if changes:
        raise SystemExit("tracked execution worktree is dirty:\n{}".format(changes))

    os.environ["PROTOCOL_ID"] = PROTOCOL_ID
    os.environ["DECLARED_SWEEP_VARIANT"] = args.variant
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    command = [sys.executable, "run.py"] + scientific_argv(
        args.dataset, args.variant, args.seed
    )
    if args.print_command:
        print(" ".join(command))
        return
    os.chdir(str(repo_root))
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
