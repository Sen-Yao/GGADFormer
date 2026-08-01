#!/usr/bin/env python3
import argparse
from collections.abc import Mapping
import datetime
import json
import math
import statistics
import sys

import wandb


VARIANTS = ["none", "random_dir", "random_mag", "random_both", "constant_mag"]
SEEDS = list(range(5))
EXPECTED_CONFIG = {
    "batch_size": 1024,
    "data_split_seed": 42,
    "dataset": "tolokers",
    "end_lr": 0.00001,
    "lambda_rec_emb": 0.1,
    "model_type": "VecGAD",
    "num_epoch": 100,
    "outlier_beta": 0.3,
    "peak_lr": 0.0001,
    "pp_k": 10,
    "progregate_alpha": 0.9,
    "rec_loss_weight": 1,
    "ring_R_max": 1,
    "ring_R_min": 0.3,
    "ring_loss_weight": 1,
    "sample_rate": 0.15,
    "train_rate": 0.05,
    "warmup_updates": 5,
    "wandb_entity": "HCCS",
    "wandb_project": "GGADFormer",
}
EXPECTED_COMMIT = "fdb150b7927f26f2e8b5270365a324d844dc8b98"


def equal_value(actual, expected):
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def get_summary_value(summary, key):
    value = summary.get(key)
    if value is None:
        root, leaf = key.split(".", 1)
        value = summary.get(root)
        if isinstance(value, Mapping):
            value = value.get(leaf)
    if value is None or not math.isfinite(float(value)):
        raise ValueError("missing/non-finite summary value {}".format(key))
    return float(value)


def collect_run(run, sweep_id):
    config = dict(run.config)
    errors = []
    variant = config.get("ablation_mode")
    seed = config.get("seed")
    if variant not in VARIANTS:
        errors.append("unexpected variant {!r}".format(variant))
    if seed not in SEEDS:
        errors.append("unexpected seed {!r}".format(seed))
    for key, expected in EXPECTED_CONFIG.items():
        if not equal_value(config.get(key), expected):
            errors.append("config {}={!r}, expected {!r}".format(key, config.get(key), expected))
    if seed in SEEDS:
        direction_seed = seed * 1000003 + 1729
        magnitude_seed = seed * 1000003 + 7919
        if config.get("ablation_direction_seed") != direction_seed:
            errors.append("direction RNG seed mismatch")
        if config.get("ablation_magnitude_seed") != magnitude_seed:
            errors.append("magnitude RNG seed mismatch")
    else:
        direction_seed = None
        magnitude_seed = None
    if run.state != "finished":
        errors.append("state is {}".format(run.state))
    if getattr(run, "commit", None) != EXPECTED_COMMIT:
        errors.append("commit {!r}, expected {}".format(getattr(run, "commit", None), EXPECTED_COMMIT))

    history = []
    for row in run.scan_history(keys=["AUC", "AP", "_step"], page_size=1000):
        if row.get("AUC") is None or row.get("AP") is None or row.get("_step") is None:
            continue
        history.append({"step": int(row["_step"]), "AUC": float(row["AUC"]), "AP": float(row["AP"])})
    final_rows = [row for row in history if row["step"] == 100]
    if len(final_rows) != 1:
        errors.append("expected one final history row at step 100, got {}".format(len(final_rows)))
    summary = dict(run.summary._json_dict)
    try:
        auc_last = get_summary_value(summary, "AUC.last")
        ap_last = get_summary_value(summary, "AP.last")
    except ValueError as exc:
        errors.append(str(exc))
        auc_last = None
        ap_last = None
    if final_rows and auc_last is not None:
        if not math.isclose(auc_last, final_rows[0]["AUC"], rel_tol=1e-12, abs_tol=1e-12):
            errors.append("AUC.last does not match final history")
        if not math.isclose(ap_last, final_rows[0]["AP"], rel_tol=1e-12, abs_tol=1e-12):
            errors.append("AP.last does not match final history")

    return {
        "run_id": run.id,
        "name": run.name,
        "url": run.url,
        "sweep_id": sweep_id,
        "state": run.state,
        "program": run.program,
        "commit": getattr(run, "commit", None),
        "created_at": run.created_at,
        "runtime_seconds": summary.get("_runtime"),
        "dataset": config.get("dataset"),
        "ablation_mode": variant,
        "seed": seed,
        "ablation_direction_seed": config.get("ablation_direction_seed"),
        "ablation_magnitude_seed": config.get("ablation_magnitude_seed"),
        "AUC.last": auc_last,
        "AP.last": ap_last,
        "history_steps": [row["step"] for row in history],
        "final_step": max([row["step"] for row in history], default=None),
        "config_subset": {key: config.get(key) for key in sorted(EXPECTED_CONFIG)},
        "validation_errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    api = wandb.Api(timeout=30)
    sweep = api.sweep("HCCS/GGADFormer/{}".format(args.sweep_id))
    runs = [collect_run(run, args.sweep_id) for run in sweep.runs]
    identities = [(row["ablation_mode"], row["seed"]) for row in runs]
    expected_identities = [(variant, seed) for variant in VARIANTS for seed in SEEDS]
    duplicates = sorted({identity for identity in identities if identities.count(identity) > 1})
    missing = sorted(set(expected_identities) - set(identities))
    unexpected = sorted(set(identities) - set(expected_identities), key=str)
    aggregate = {}
    for variant in VARIANTS:
        rows = [row for row in runs if row["ablation_mode"] == variant and not row["validation_errors"]]
        aggregate[variant] = {}
        for metric in ["AUC.last", "AP.last"]:
            values = [row[metric] for row in rows]
            aggregate[variant][metric] = {
                "mean": statistics.mean(values) if values else None,
                "sample_std": statistics.stdev(values) if len(values) >= 2 else None,
                "values_by_seed": {str(row["seed"]): row[metric] for row in sorted(rows, key=lambda item: item["seed"])},
            }

    validation = {
        "sweep_state": sweep.state,
        "expected_trials": 25,
        "observed_runs": len(runs),
        "finished_trials": sum(row["state"] == "finished" for row in runs),
        "valid_trials": sum(not row["validation_errors"] for row in runs),
        "duplicates": [list(item) for item in duplicates],
        "missing": [list(item) for item in missing],
        "unexpected": [list(item) for item in unexpected],
        "all_valid": (
            len(runs) == 25
            and not duplicates
            and not missing
            and not unexpected
            and all(not row["validation_errors"] for row in runs)
        ),
    }
    payload = {
        "schema_version": 1,
        "collected_at_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "entity": "HCCS",
        "project": "GGADFormer",
        "sweep_id": args.sweep_id,
        "sweep_url": "https://wandb.ai/HCCS/GGADFormer/sweeps/{}".format(args.sweep_id),
        "execution_host": "HCCS-85",
        "code_sha": EXPECTED_COMMIT,
        "validation": validation,
        "aggregate": aggregate,
        "runs": sorted(runs, key=lambda row: (VARIANTS.index(row["ablation_mode"]) if row["ablation_mode"] in VARIANTS else 99, row["seed"] if row["seed"] in SEEDS else 99)),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if validation["all_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
