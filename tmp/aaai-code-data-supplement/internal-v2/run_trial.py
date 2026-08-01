"""Internal-only W&B wrapper for one anonymous-package trial."""

import json
import os
import subprocess
import sys
from pathlib import Path

import wandb


PACKAGE = Path(__file__).resolve().parents[0] / ".." / "package-v2"
RESULTS = Path(__file__).resolve().parent / "results-85"
COMMON = {
    "data_split_seed": 42,
    "train_rate": 0.05,
    "sample_rate": 0.15,
    "outlier_beta": 0.3,
    "ring_R_min": 0.3,
    "ring_R_max": 1.0,
    "lambda_rec_tok": 1.0,
    "bce_loss_weight": 1.0,
    "rec_loss_weight": 1.0,
    "control": "full",
    "embedding_dim": 256,
    "ffn_dim": 256,
    "num_heads": 2,
    "num_layers": 3,
    "dropout": 0.4,
    "attention_dropout": 0.4,
    "weight_decay": 0.0,
}
DATASET_CONFIGS = {
    "Amazon": dict(batch_size=1024, num_epoch=100, peak_lr=0.0003,
                   end_lr=0.0001, warmup_updates=50, pp_k=5,
                   progregate_alpha=0.4, lambda_rec_emb=0.1,
                   ring_loss_weight=1.0),
    "reddit": dict(batch_size=1024, num_epoch=200, peak_lr=0.0005,
                   end_lr=0.0001, warmup_updates=50, pp_k=10,
                   progregate_alpha=0.1, lambda_rec_emb=0.1,
                   ring_loss_weight=1.0),
    "photo": dict(batch_size=128, num_epoch=200, peak_lr=0.0005,
                  end_lr=0.0001, warmup_updates=50, pp_k=6,
                  progregate_alpha=0.1, lambda_rec_emb=0.1,
                  ring_loss_weight=1.0),
    "elliptic": dict(batch_size=32768, num_epoch=150, peak_lr=0.0005,
                     end_lr=0.0003, warmup_updates=50, pp_k=7,
                     progregate_alpha=0.6, lambda_rec_emb=2.0,
                     ring_loss_weight=20.0),
    "t_finance": dict(batch_size=8192, num_epoch=40, peak_lr=0.0005,
                      end_lr=0.0001, warmup_updates=50, pp_k=7,
                      progregate_alpha=0.3, lambda_rec_emb=0.1,
                      ring_loss_weight=1.0),
    "tolokers": dict(batch_size=1024, num_epoch=100, peak_lr=0.0001,
                     end_lr=0.00001, warmup_updates=5, pp_k=10,
                     progregate_alpha=0.9, lambda_rec_emb=0.1,
                     ring_loss_weight=1.0),
}


def main():
    run = wandb.init(entity="HCCS", project="DualRefGAD")
    assigned = dict(run.config)
    dataset = assigned["dataset"]
    seed = int(assigned["seed"])
    config = dict(COMMON)
    config.update(DATASET_CONFIGS[dataset])
    config.update({"dataset": dataset, "seed": seed})
    audit_config = dict(config)
    audit_config.update(
        {
            "execution_sha": os.environ["VECGAD_EXECUTION_SHA"],
            "protocol_id": "vecgad-package-v2-full-revalidation-019fbc58",
            "execution_host": "HCCS-85",
        }
    )
    run.config.update(audit_config, allow_val_change=True)
    command = [sys.executable, str(PACKAGE / "run.py")]
    for key, value in config.items():
        command.extend(["--{}".format(key), str(value)])
    command.extend(["--data_dir", "/root/gpufree-data/linziyao/DualRefGAD/dataset"])
    environment = os.environ.copy()
    environment.setdefault("WANDB_SILENT", "true")
    environment["PYTHONHASHSEED"] = str(seed)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    result = subprocess.run(command, check=False, text=True, capture_output=True, env=environment)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        run.summary["run_valid"] = False
        run.finish(exit_code=1)
        raise SystemExit(result.returncode)
    line = next(
        (line for line in result.stdout.splitlines() if line.startswith("final_epoch=")),
        None,
    )
    if line is None:
        run.summary["run_valid"] = False
        run.finish(exit_code=1)
        raise RuntimeError("missing fixed-endpoint result line")
    fields = dict(item.split("=", 1) for item in line.split())
    payload = {
        "dataset": dataset,
        "seed": seed,
        "final_epoch": int(fields["final_epoch"]),
        "AUROC": float(fields["AUROC"]),
        "AUPRC": float(fields["AUPRC"]),
        "runtime_seconds": float(fields["runtime_seconds"]),
    }
    RESULTS.mkdir(exist_ok=True)
    result_path = RESULTS / "{}-seed{}.json".format(dataset, seed)
    result_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    run.log({"AUROC": payload["AUROC"], "AUPRC": payload["AUPRC"]})
    run.summary.update(payload)
    run.summary["run_valid"] = True
    run.finish(exit_code=0)


if __name__ == "__main__":
    main()
