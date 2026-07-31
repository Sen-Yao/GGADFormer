#!/usr/bin/env python
import argparse
import os
import sys


COMMON_ARGS = [
    "--batch_size=8192",
    "--dataset=elliptic",
    "--end_lr=0.0001",
    "--num_epoch=200",
    "--outlier_beta=0.3",
    "--peak_lr=0.0005",
    "--pp_k=8",
    "--progregate_alpha=0.8",
    "--rec_loss_weight=1",
    "--ring_R_max=1",
    "--ring_R_min=0.3",
    "--train_rate=0.05",
    "--warmup_updates=50",
    "--hsc_diagnostics",
]

VARIANT_ARGS = {
    "control_2_20": ["--lambda_rec_emb=2", "--ring_loss_weight=20"],
    "unified_0p1_1": ["--lambda_rec_emb=0.1", "--ring_loss_weight=1"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(VARIANT_ARGS))
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    args = parser.parse_args()

    os.environ["DECLARED_SWEEP_VARIANT"] = args.variant
    argv = [
        sys.executable,
        "run.py",
        *COMMON_ARGS,
        *VARIANT_ARGS[args.variant],
        f"--seed={args.seed}",
    ]
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
