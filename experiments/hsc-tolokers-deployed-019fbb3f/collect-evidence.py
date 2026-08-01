#!/usr/bin/env python3
import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import re
import statistics
from collections.abc import Mapping
from pathlib import Path

import wandb


DATASETS = ("tolokers",)
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
    "tolokers": {
        "ablation_mode": "none",
        "batch_size": 1024,
        "data_split_seed": 42,
        "dataset": "tolokers",
        "end_lr": 0.00001,
        "lambda_rec_emb": 0.1,
        "lambda_rec_tok": 1.0,
        "num_epoch": 100,
        "outlier_beta": 0.3,
        "peak_lr": 0.0001,
        "pp_k": 10,
        "progregate_alpha": 0.9,
        "rec_loss_weight": 1.0,
        "ring_R_max": 1.0,
        "ring_R_min": 0.3,
        "ring_loss_weight": 1.0,
        "sample_rate": 0.15,
        "train_rate": 0.05,
        "warmup_updates": 5,
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
AUDIT_CONFIG = {
    "hsc_label_usage_scope": "oracle center construction only; excluded from all other losses and scoring",
    "fixed_final_epoch_metric_policy": "AUC.last/AP.last at fixed training endpoint",
    "hsc_diagnostic_policy": "final checkpoint; fixed weighted-sampler replay; sample-weighted shell metrics",
    "wandb_entity": "HCCS",
    "wandb_project": "GGADFormer",
}
T_CRITICAL_DF4_975 = 2.7764451051977987
PROVIDER_HISTORY_ARTIFACT_TYPE = "wandb-history"


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


def collect_agent_evidence(task_root, expected_agent_count):
    logs_dir = task_root / "logs"
    exit_paths = sorted(logs_dir.glob("agent-gpu*.exitcode"))
    if len(exit_paths) != expected_agent_count:
        raise AssertionError(
            f"expected {expected_agent_count} agent exit records, found {len(exit_paths)}"
        )

    records = []
    observed_gpus = set()
    for exit_path in exit_paths:
        match = re.fullmatch(r"agent-gpu([0-7])-.+\.exitcode", exit_path.name)
        if match is None:
            raise AssertionError(f"unrecognized agent exit record: {exit_path.name}")
        gpu_index = int(match.group(1))
        if gpu_index in observed_gpus:
            raise AssertionError(f"multiple terminal agent records for GPU {gpu_index}")
        observed_gpus.add(gpu_index)

        log_path = exit_path.with_suffix(".log")
        start_path = exit_path.with_suffix(".start-utc")
        finish_path = exit_path.with_suffix(".finish-utc")
        for path in (log_path, start_path, finish_path):
            if not path.is_file():
                raise AssertionError(f"missing agent evidence file: {path}")
        exit_code = int(exit_path.read_text(encoding="utf-8").strip())
        if exit_code != 0:
            raise AssertionError(f"agent on GPU {gpu_index} exited with {exit_code}")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "agent_started_utc=" not in log_text or "agent_finished_utc=" not in log_text:
            raise AssertionError(f"agent log lacks start/finish markers: {log_path}")
        if "exit_code=0" not in log_text:
            raise AssertionError(f"agent log lacks successful terminal marker: {log_path}")
        records.append(
            {
                "gpu_index": gpu_index,
                "log_path": str(log_path),
                "log_sha256": sha256_file(log_path),
                "exit_path": str(exit_path),
                "exit_sha256": sha256_file(exit_path),
                "exit_code": exit_code,
                "started_at_utc": start_path.read_text(encoding="utf-8").strip(),
                "finished_at_utc": finish_path.read_text(encoding="utf-8").strip(),
            }
        )
    if observed_gpus != set(range(expected_agent_count)):
        raise AssertionError(f"agent GPU coverage mismatch: {sorted(observed_gpus)}")
    return sorted(records, key=lambda row: row["gpu_index"])


def assert_equal(actual, expected, label):
    if isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")
    elif actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def summary_last(summary, name):
    payload = getattr(summary, "_json_dict", summary)
    value = payload.get(f"{name}.last")
    if value is None:
        value = payload.get(name)
    if isinstance(value, Mapping):
        value = value.get("last")
    if value is None:
        raise AssertionError(f"missing summary value {name}")
    return float(value)


def direct_summary(summary, name):
    payload = getattr(summary, "_json_dict", summary)
    value = payload.get(name)
    if value is None:
        raise AssertionError(f"missing summary value {name}")
    return value


def audit_wandb_artifacts(run):
    """Allow only W&B's provider-generated run-history artifact.

    W&B materializes this artifact asynchronously after a run is queried. It
    contains the metric history backing the API and is distinct from user
    uploaded checkpoints, source, data, or declared artifacts.
    """
    logged = list(run.logged_artifacts())
    used = list(run.used_artifacts())
    if used:
        raise AssertionError(f"run {run.id} has used W&B artifacts: {len(used)}")
    if len(logged) != 1:
        raise AssertionError(
            f"run {run.id} expected one provider history artifact, found {len(logged)}"
        )
    artifact = logged[0]
    expected_name = f"run-{run.id}-history:v0"
    if getattr(artifact, "name", None) != expected_name:
        raise AssertionError(f"run {run.id} has unexpected artifact name")
    for attribute, expected in (
        ("type", PROVIDER_HISTORY_ARTIFACT_TYPE),
        ("state", "COMMITTED"),
        ("entity", "HCCS"),
        ("project", "GGADFormer"),
        ("description", f"Weights & Biases Run History Data for {run.id}"),
    ):
        if getattr(artifact, attribute, None) != expected:
            raise AssertionError(f"run {run.id} provider artifact {attribute} mismatch")
    if dict(getattr(artifact, "metadata", {}) or {}):
        raise AssertionError(f"run {run.id} provider artifact has metadata")
    aliases = sorted(getattr(artifact, "aliases", []) or [])
    if aliases != ["latest"]:
        raise AssertionError(f"run {run.id} provider artifact aliases mismatch")
    files = list(artifact.files())
    if len(files) != 1 or getattr(files[0], "name", None) != "0000.parquet":
        raise AssertionError(f"run {run.id} provider artifact file manifest mismatch")
    file_entry = files[0]
    return {
        "only_provider_generated_history_artifact": True,
        "used_artifacts": [],
        "logged_artifacts": [
            {
                "name": artifact.name,
                "type": artifact.type,
                "state": artifact.state,
                "entity": artifact.entity,
                "project": artifact.project,
                "description": artifact.description,
                "aliases": aliases,
                "size": int(getattr(artifact, "size", 0)),
                "digest": getattr(artifact, "digest", None),
                "files": [
                    {
                        "name": file_entry.name,
                        "size": int(getattr(file_entry, "size", 0)),
                        "digest": getattr(file_entry, "digest", None),
                    }
                ],
            }
        ],
    }


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
    parser.add_argument("--expected-agent-count", type=int, default=8)
    args = parser.parse_args()

    output_dir = args.output_dir or args.task_root / "evidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = args.task_root / "diagnostics"
    checkpoint_dir = args.task_root / "checkpoints"
    agent_evidence = collect_agent_evidence(args.task_root, args.expected_agent_count)

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
        wandb_artifact_audit = audit_wandb_artifacts(run)

        expected_config = BASE_CONFIG[dataset]
        for key, expected in expected_config.items():
            assert_equal(config.get(key), expected, f"run {run.id} config.{key}")
        for key, expected in AUDIT_CONFIG.items():
            assert_equal(config.get(key), expected, f"run {run.id} audit.{key}")
        assert_equal(config.get("hsc_oracle_q"), Q_BY_CONDITION[condition], f"run {run.id} q")
        assert_equal(config.get("code_sha"), args.code_sha, f"run {run.id} code_sha")
        assert_equal(config.get("execution_host"), "HCCS-85", f"run {run.id} host")
        assert_equal(
            config.get("protocol_identity"),
            "hsc-tolokers-deployed-019fbb3f-v1",
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
        if direct_summary(run.summary, "_step") != int(config["num_epoch"]):
            raise AssertionError(f"run {run.id} summary endpoint step mismatch")

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
            "protocol_identity": "hsc-tolokers-deployed-019fbb3f-v1",
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
                "wandb_artifact_audit": wandb_artifact_audit,
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
            "audit_config_valid": True,
            "only_provider_generated_history_artifacts": True,
            "no_user_uploaded_or_used_wandb_artifacts": True,
        },
        "pairing_audit": pairing_audit,
        "agent_evidence": agent_evidence,
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
