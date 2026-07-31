#!/usr/bin/env python3
import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

import wandb


DATASETS = ("Amazon", "tolokers")
CONDITIONS = ("default", "q0", "q10", "q20", "q30", "q40")
SEEDS = tuple(range(5))
Q_BY_CONDITION = {
    "default": None,
    "q0": 0.0,
    "q10": 0.1,
    "q20": 0.2,
    "q30": 0.3,
    "q40": 0.4,
}
EXPECTED_TRIALS = set(itertools.product(DATASETS, CONDITIONS, SEEDS))
BASE_CONFIG = {
    "Amazon": {
        "ablation_mode": "none",
        "batch_size": 1024,
        "data_split_seed": 42,
        "dataset": "Amazon",
        "end_lr": 0.0001,
        "lambda_rec_emb": 0.1,
        "num_epoch": 100,
        "outlier_beta": 0.3,
        "peak_lr": 0.0003,
        "pp_k": 5,
        "progregate_alpha": 0.4,
        "rec_loss_weight": 1.0,
        "ring_R_max": 1.0,
        "ring_R_min": 0.3,
        "ring_loss_weight": 1.0,
        "train_rate": 0.05,
        "warmup_updates": 50,
    },
    "tolokers": {
        "ablation_mode": "none",
        "batch_size": 1024,
        "data_split_seed": 42,
        "dataset": "tolokers",
        "end_lr": 0.0001,
        "lambda_rec_emb": 0.5,
        "lambda_rec_tok": 1.0,
        "num_epoch": 70,
        "outlier_beta": 0.3,
        "peak_lr": 0.0001,
        "pp_k": 3,
        "progregate_alpha": 0.3,
        "rec_loss_weight": 0.1,
        "ring_R_max": 0.5,
        "ring_R_min": 0.5,
        "ring_loss_weight": 20.0,
        "train_rate": 0.05,
        "warmup_updates": 50,
    },
}
SCALAR_METRICS = (
    "AUC.last",
    "AP.last",
    "HSC.ShellHit",
    "HSC.inner_violation",
    "HSC.outer_violation",
    "HSC.mean_loss",
    "HSC.center_shift_from_default",
    "HSC.center_shift_from_normal",
    "HSC.sampled_anomaly_fraction",
)
PAIRING_FIELDS = (
    "initial_model_sha256",
    "training_batch_trace_sha256",
    "pseudo_source_trace_sha256",
    "HSC.diagnostic_batch_trace_sha256",
    "HSC.diagnostic_source_trace_sha256",
)
T_CRITICAL_DF4_975 = 2.7764451051977987


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_equal(actual, expected, label):
    if isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")
    elif actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def summary_last(summary, name):
    value = summary.get(name)
    if hasattr(value, "get"):
        value = value.get("last")
    if value is None:
        raise AssertionError(f"missing summary value {name}")
    return float(value)


def direct_summary(summary, name):
    value = summary.get(name)
    if value is None:
        raise AssertionError(f"missing summary value {name}")
    return value


def aggregate_records(records):
    cells = {dataset: {condition: {} for condition in CONDITIONS} for dataset in DATASETS}
    for record in records:
        cells[record["dataset"]][record["hsc_center_condition"]][record["seed"]] = record

    aggregate = {}
    paired = {}
    for dataset in DATASETS:
        aggregate[dataset] = {}
        paired[dataset] = {}
        for condition in CONDITIONS:
            if sorted(cells[dataset][condition]) != list(SEEDS):
                raise AssertionError(f"seed coverage mismatch for {dataset}/{condition}")
            aggregate[dataset][condition] = {}
            for metric in SCALAR_METRICS:
                values = [cells[dataset][condition][seed][metric] for seed in SEEDS]
                aggregate[dataset][condition][metric] = {
                    "mean": statistics.mean(values),
                    "sample_std_ddof1": statistics.stdev(values),
                }

        for condition in CONDITIONS[1:]:
            paired[dataset][f"{condition}_minus_default"] = {}
            for metric in SCALAR_METRICS:
                rows = [
                    {
                        "seed": seed,
                        "delta": cells[dataset][condition][seed][metric]
                        - cells[dataset]["default"][seed][metric],
                    }
                    for seed in SEEDS
                ]
                values = [row["delta"] for row in rows]
                mean = statistics.mean(values)
                sample_std = statistics.stdev(values)
                half_width = T_CRITICAL_DF4_975 * sample_std / math.sqrt(len(SEEDS))
                paired[dataset][f"{condition}_minus_default"][metric] = {
                    "by_seed": rows,
                    "mean": mean,
                    "sample_std_ddof1": sample_std,
                    "paired_t_95ci": [mean - half_width, mean + half_width],
                }
    return aggregate, paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-id", required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--task-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or args.task_root / "evidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = args.task_root / "diagnostics"
    checkpoint_dir = args.task_root / "checkpoints"

    api = wandb.Api(timeout=60)
    sweep_path = f"HCCS/GGADFormer/{args.sweep_id}"
    sweep = api.sweep(sweep_path)
    if sweep.state != "FINISHED":
        raise AssertionError(f"sweep state is {sweep.state}, expected FINISHED")
    if len(sweep.runs) != len(EXPECTED_TRIALS):
        raise AssertionError(
            f"expected {len(EXPECTED_TRIALS)} runs, found {len(sweep.runs)}"
        )

    observed = set()
    records = []
    for run in sorted(
        sweep.runs,
        key=lambda item: (
            item.config.get("dataset", ""),
            item.config.get("hsc_center_condition", ""),
            int(item.config.get("seed", -1)),
        ),
    ):
        config = {key: value for key, value in run.config.items() if not key.startswith("_")}
        dataset = config.get("dataset")
        condition = config.get("hsc_center_condition")
        seed = int(config.get("seed", -1))
        trial = (dataset, condition, seed)
        if trial not in EXPECTED_TRIALS:
            raise AssertionError(f"unexpected trial {trial} in run {run.id}")
        if trial in observed:
            raise AssertionError(f"duplicate trial {trial}")
        observed.add(trial)
        if run.state != "finished":
            raise AssertionError(f"run {run.id} state is {run.state}")

        expected_config = BASE_CONFIG[dataset]
        for key, expected in expected_config.items():
            assert_equal(config.get(key), expected, f"run {run.id} config.{key}")
        assert_equal(config.get("hsc_oracle_q"), Q_BY_CONDITION[condition], f"run {run.id} q")
        assert_equal(config.get("code_sha"), args.code_sha, f"run {run.id} code_sha")
        assert_equal(config.get("execution_host"), "HCCS-85", f"run {run.id} host")
        assert_equal(
            config.get("protocol_identity"),
            "hsc-center-contamination-019fb5c1-v1",
            f"run {run.id} protocol",
        )
        assert_equal(
            config.get("pair_id"),
            f"{dataset}:seed={seed}:data_split_seed=42",
            f"run {run.id} pair_id",
        )

        expected_steps = list(range(0, int(config["num_epoch"]) + 1, 10))
        history = list(run.scan_history(keys=["_step", "AUC", "AP"], page_size=500))
        steps = [int(row["_step"]) for row in history]
        if steps != expected_steps:
            raise AssertionError(f"run {run.id}: history steps {steps}")
        final = history[-1]
        auc = float(final["AUC"])
        ap = float(final["AP"])
        assert_equal(summary_last(run.summary, "AUC"), auc, f"run {run.id} AUC.last")
        assert_equal(summary_last(run.summary, "AP"), ap, f"run {run.id} AP.last")
        assert_equal(
            direct_summary(run.summary, "fixed_endpoint_epoch"),
            int(config["num_epoch"]),
            f"run {run.id} fixed endpoint",
        )
        if direct_summary(run.summary, "run_valid") is not True:
            raise AssertionError(f"run {run.id} was not marked valid")
        if direct_summary(run.summary, "HSC.diagnostic_replay_repeat_verified") is not True:
            raise AssertionError(f"run {run.id} repeated diagnostic replay was not verified")

        diagnostic_path = diagnostic_dir / f"{run.id}.json"
        checkpoint_path = checkpoint_dir / f"{run.id}.pt"
        if not diagnostic_path.is_file() or not checkpoint_path.is_file():
            raise AssertionError(f"run {run.id} is missing checkpoint or diagnostic evidence")
        diagnostic_sha = sha256_file(diagnostic_path)
        checkpoint_sha = sha256_file(checkpoint_path)
        assert_equal(
            direct_summary(run.summary, "diagnostic_sha256"),
            diagnostic_sha,
            f"run {run.id} diagnostic hash",
        )
        assert_equal(
            direct_summary(run.summary, "checkpoint_sha256"),
            checkpoint_sha,
            f"run {run.id} checkpoint hash",
        )
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        identity = diagnostic["checkpoint_identity"]
        for key, expected in {
            "run_id": run.id,
            "dataset": dataset,
            "hsc_center_condition": condition,
            "seed": seed,
            "data_split_seed": 42,
            "code_sha": args.code_sha,
            "protocol_identity": "hsc-center-contamination-019fb5c1-v1",
            "final_training_epoch": int(config["num_epoch"]),
        }.items():
            assert_equal(identity.get(key), expected, f"run {run.id} identity.{key}")
        if diagnostic.get("diagnostic_replay_repeat_verified") is not True:
            raise AssertionError(f"run {run.id} diagnostic repeat flag is false")
        assert_equal(diagnostic["checkpoint_sha256"], checkpoint_sha, f"run {run.id} checkpoint")
        assert_equal(diagnostic["final_metrics"]["AUC.last"], auc, f"run {run.id} diagnostic AUC")
        assert_equal(diagnostic["final_metrics"]["AP.last"], ap, f"run {run.id} diagnostic AP")

        hsc = diagnostic["hsc_diagnostics"]
        metric_values = {
            "AUC.last": auc,
            "AP.last": ap,
            "HSC.ShellHit": float(hsc["ShellHit"]),
            "HSC.inner_violation": float(hsc["inner_violation"]),
            "HSC.outer_violation": float(hsc["outer_violation"]),
            "HSC.mean_loss": float(hsc["mean_hsc_loss"]),
            "HSC.center_shift_from_default": float(hsc["center_shift_from_default"]),
            "HSC.center_shift_from_normal": float(hsc["center_shift_from_normal"]),
            "HSC.sampled_anomaly_fraction": float(hsc["sampled_anomaly_fraction"]),
        }
        for metric, value in metric_values.items():
            if metric.startswith("HSC."):
                assert_equal(
                    direct_summary(run.summary, metric),
                    value,
                    f"run {run.id} summary {metric}",
                )

        pairing = {
            "initial_model_sha256": diagnostic["initial_model_sha256"],
            "training_batch_trace_sha256": diagnostic["training_batch_trace_sha256"],
            "pseudo_source_trace_sha256": diagnostic["pseudo_source_trace_sha256"],
            "HSC.diagnostic_batch_trace_sha256": hsc["batch_trace_sha256"],
            "HSC.diagnostic_source_trace_sha256": hsc["source_trace_sha256"],
        }
        for key, value in pairing.items():
            assert_equal(direct_summary(run.summary, key), value, f"run {run.id} {key}")

        records.append(
            {
                "dataset": dataset,
                "hsc_center_condition": condition,
                "hsc_oracle_q": Q_BY_CONDITION[condition],
                "seed": seed,
                "pair_id": config["pair_id"],
                "run_id": run.id,
                "url": run.url,
                "state": run.state,
                "gpu_index": str(config.get("gpu_index")),
                "runtime_seconds": float(run.summary.get("_runtime", 0.0)),
                "final_step": int(config["num_epoch"]),
                "code_sha": args.code_sha,
                "protocol_identity": config["protocol_identity"],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "diagnostic_path": str(diagnostic_path),
                "diagnostic_sha256": diagnostic_sha,
                "final_model_state_sha256": diagnostic["final_model_state_sha256"],
                "config": config,
                "config_sha256": sha256_bytes(canonical_bytes(config)),
                "history": history,
                "history_sha256": sha256_bytes(canonical_bytes(history)),
                **metric_values,
                **pairing,
            }
        )

    if observed != EXPECTED_TRIALS:
        raise AssertionError("expected trial coverage is incomplete")

    by_pair = {}
    for record in records:
        by_pair.setdefault(record["pair_id"], []).append(record)
    pairing_audit = {}
    for pair_id, pair_records in sorted(by_pair.items()):
        if len(pair_records) != len(CONDITIONS):
            raise AssertionError(f"{pair_id}: expected six paired conditions")
        if {record["hsc_center_condition"] for record in pair_records} != set(CONDITIONS):
            raise AssertionError(f"{pair_id}: condition coverage mismatch")
        pairing_audit[pair_id] = {}
        for field in PAIRING_FIELDS:
            values = {record[field] for record in pair_records}
            if len(values) != 1:
                raise AssertionError(f"{pair_id}: pairing mismatch in {field}")
            pairing_audit[pair_id][field] = next(iter(values))

    aggregate, paired = aggregate_records(records)
    collected_at = utc_now()
    authoritative = {
        "schema_version": 1,
        "collected_at_utc": collected_at,
        "source": f'wandb.Api().sweep("{sweep_path}") plus local checkpoint/diagnostic hashes',
        "entity": "HCCS",
        "project": "GGADFormer",
        "sweep_id": args.sweep_id,
        "sweep_url": f"https://wandb.ai/HCCS/GGADFormer/sweeps/{args.sweep_id}",
        "state": sweep.state,
        "code_sha": args.code_sha,
        "expected_trials": len(EXPECTED_TRIALS),
        "identity_validation": {
            "all_expected_trials_present_once": True,
            "all_runs_finished_and_valid": True,
            "all_final_history_steps_present": True,
            "summary_matches_final_history_and_diagnostics": True,
            "checkpoint_and_diagnostic_hashes_match": True,
            "pairing_hashes_match_within_dataset_seed": True,
            "repeated_diagnostic_replay_verified": True,
        },
        "pairing_audit": pairing_audit,
        "runs": records,
    }
    authoritative_path = output_dir / "authoritative-sweep.json"
    authoritative_path.write_text(
        json.dumps(authoritative, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )

    results = {
        "schema_version": 1,
        "collected_at_utc": collected_at,
        "entity": "HCCS",
        "project": "GGADFormer",
        "sweep_id": args.sweep_id,
        "sweep_state": sweep.state,
        "code_sha": args.code_sha,
        "expected_trials": len(EXPECTED_TRIALS),
        "observed_trials": len(records),
        "decision_threshold": None,
        "aggregate": aggregate,
        "paired_deltas_vs_default": paired,
        "source_hashes": {
            "authoritative-sweep.json": sha256_file(authoritative_path),
        },
    }
    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "authoritative_path": str(authoritative_path),
                "authoritative_sha256": sha256_file(authoritative_path),
                "results_path": str(results_path),
                "results_sha256": sha256_file(results_path),
                "run_count": len(records),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
