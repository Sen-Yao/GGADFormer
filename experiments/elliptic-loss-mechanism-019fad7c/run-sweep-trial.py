#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


EXPECTED_CODE_SHA = "65014dd11bed01b761aa7c3889c7718b7950884d"
COMMON_ARGS = [
    "--batch_size=32768",
    "--dataset=elliptic",
    "--end_lr=0.0003",
    "--mechanism_diagnostics=true",
    "--num_epoch=150",
    "--outlier_beta=0.3",
    "--peak_lr=0.0005",
    "--pp_k=7",
    "--progregate_alpha=0.6",
    "--rec_loss_weight=1",
    "--ring_R_max=1",
    "--ring_R_min=0.3",
    "--train_rate=0.05",
    "--warmup_updates=50",
]
CELL_ARGS = {
    "control_2_20": ["--lambda_rec_emb=2", "--ring_loss_weight=20"],
    "emb_only_0p1_20": ["--lambda_rec_emb=0.1", "--ring_loss_weight=20"],
    "ring_only_2_1": ["--lambda_rec_emb=2", "--ring_loss_weight=1"],
    "unified_0p1_1": ["--lambda_rec_emb=0.1", "--ring_loss_weight=1"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism_cell", required=True, choices=sorted(CELL_ARGS))
    parser.add_argument("--seed", required=True, type=int, choices=[0])
    args = parser.parse_args()

    actual_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_sha != EXPECTED_CODE_SHA:
        raise SystemExit("unexpected code SHA: {}".format(actual_sha))

    os.environ["DECLARED_MECHANISM_CELL"] = args.mechanism_cell
    argv = [
        sys.executable,
        "run.py",
        *COMMON_ARGS,
        *CELL_ARGS[args.mechanism_cell],
        "--seed={}".format(args.seed),
    ]
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
