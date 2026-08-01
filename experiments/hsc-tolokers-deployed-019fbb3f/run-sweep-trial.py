#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


DATASET = "tolokers"
CONDITIONS = ("default", "q0", "q10", "q20", "q30", "q40")
SEEDS = tuple(range(5))
AUTHORITATIVE_ARGS = (
    "--ablation_mode=none",
    "--batch_size=1024",
    "--data_split_seed=42",
    "--dataset=tolokers",
    "--end_lr=0.00001",
    "--lambda_rec_emb=0.1",
    "--num_epoch=100",
    "--outlier_beta=0.3",
    "--peak_lr=0.0001",
    "--pp_k=10",
    "--progregate_alpha=0.9",
    "--rec_loss_weight=1",
    "--ring_R_max=1",
    "--ring_R_min=0.3",
    "--ring_loss_weight=1",
    "--sample_rate=0.15",
    "--train_rate=0.05",
    "--warmup_updates=5",
)


def build_run_argv(dataset, condition, seed, executable=sys.executable):
    if dataset != DATASET:
        raise ValueError(f"unsupported dataset: {dataset}")
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported HSC center condition: {condition}")
    if seed not in SEEDS:
        raise ValueError(f"unsupported seed: {seed}")
    return [
        executable,
        "run.py",
        *AUTHORITATIVE_ARGS,
        f"--hsc_center_condition={condition}",
        f"--seed={seed}",
    ]


def verify_execution_identity():
    expected_sha = os.environ.get("CODE_SHA")
    if not expected_sha:
        raise SystemExit("CODE_SHA must identify the committed execution worktree")
    actual_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_sha != expected_sha:
        raise SystemExit(f"unexpected execution SHA: {actual_sha}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=(DATASET,))
    parser.add_argument(
        "--hsc_center_condition", required=True, choices=CONDITIONS
    )
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    args = parser.parse_args()

    verify_execution_identity()
    argv = build_run_argv(
        args.dataset, args.hsc_center_condition, args.seed
    )
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
