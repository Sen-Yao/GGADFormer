#!/usr/bin/env python
import argparse
import os
import sys


COMMON_ARGS = [
    "--dataset=elliptic",
    "--lambda_rec_emb=0.1",
    "--ring_loss_weight=1",
    "--outlier_beta=0.3",
    "--peak_lr=0.0005",
    "--rec_loss_weight=1",
    "--ring_R_max=1",
    "--ring_R_min=0.3",
    "--train_rate=0.05",
    "--warmup_updates=50",
    "--hsc_diagnostics",
]

OPTIMIZATION_ARGS = {
    "current": ["--batch_size=32768", "--end_lr=0.0003", "--num_epoch=150"],
    "mixed": ["--batch_size=8192", "--end_lr=0.0001", "--num_epoch=200"],
}

PROPAGATION_ARGS = {
    "current": ["--pp_k=7", "--progregate_alpha=0.6"],
    "mixed": ["--pp_k=8", "--progregate_alpha=0.8"],
}

OPTIMIZATION_METADATA = {
    "current": {"final_step": "150", "batches_per_epoch": "2"},
    "mixed": {"final_step": "200", "batches_per_epoch": "6"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimization_bundle", required=True, choices=sorted(OPTIMIZATION_ARGS))
    parser.add_argument("--propagation_bundle", required=True, choices=sorted(PROPAGATION_ARGS))
    parser.add_argument("--seed", required=True, type=int, choices=range(5))
    args = parser.parse_args()

    cell = f"opt_{args.optimization_bundle}__prop_{args.propagation_bundle}"
    metadata = OPTIMIZATION_METADATA[args.optimization_bundle]
    os.environ["OPTIMIZATION_BUNDLE"] = args.optimization_bundle
    os.environ["PROPAGATION_BUNDLE"] = args.propagation_bundle
    os.environ["FACTORIAL_CELL"] = cell
    os.environ["FINAL_HISTORY_STEP"] = metadata["final_step"]
    os.environ["OPTIMIZER_UPDATES_PER_EPOCH"] = metadata["batches_per_epoch"]

    argv = [
        sys.executable,
        "run.py",
        *COMMON_ARGS,
        *OPTIMIZATION_ARGS[args.optimization_bundle],
        *PROPAGATION_ARGS[args.propagation_bundle],
        f"--seed={args.seed}",
    ]
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()

