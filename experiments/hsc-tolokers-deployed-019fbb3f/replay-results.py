#!/usr/bin/env python3
"""Independent W&B replay for the formal Tolokers HSC sweep."""

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import statistics

import wandb


CONDITIONS = ("default", "q0", "q10", "q20", "q30", "q40")
SEEDS = tuple(range(5))
EXPECTED = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
EXPECTED_CONFIG = {
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
}
EXPECTED_AUDIT = {
    "hsc_label_usage_scope": "oracle center construction only; excluded from all other losses and scoring",
    "fixed_final_epoch_metric_policy": "AUC.last/AP.last at fixed training endpoint",
    "hsc_diagnostic_policy": "final checkpoint; fixed weighted-sampler replay; sample-weighted shell metrics",
    "wandb_entity": "HCCS",
    "wandb_project": "GGADFormer",
}
PROTOCOL_ID = "hsc-tolokers-deployed-019fbb3f-v1"
PROVIDER_HISTORY_ARTIFACT_TYPE = "wandb-history"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def equal(actual, expected):
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def summary_value(summary, key):
    payload = getattr(summary, "_json_dict", summary)
    value = payload.get(key)
    if value is None and "." in key:
        root, leaf = key.split(".", 1)
        value = payload.get(root)
        if isinstance(value, Mapping):
            value = value.get(leaf)
    if value is None:
        raise AssertionError(f"missing summary value {key}")
    return value


def audit_wandb_artifacts(run):
    """Independently validate W&B's provider-generated history artifact only."""
    logged = list(run.logged_artifacts())
    used = list(run.used_artifacts())
    if used:
        raise AssertionError(f"run {run.id} has used W&B artifacts: {len(used)}")
    if len(logged) != 1:
        raise AssertionError(
            f"run {run.id} expected one provider history artifact, found {len(logged)}"
        )
    artifact = logged[0]
    if getattr(artifact, "name", None) != f"run-{run.id}-history:v0":
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
    if sorted(getattr(artifact, "aliases", []) or []) != ["latest"]:
        raise AssertionError(f"run {run.id} provider artifact aliases mismatch")
    files = list(artifact.files())
    if len(files) != 1 or getattr(files[0], "name", None) != "0000.parquet":
        raise AssertionError(f"run {run.id} provider artifact file manifest mismatch")
    return {
        "only_provider_generated_history_artifact": True,
        "used_artifacts": [],
        "logged_artifact": {
            "name": artifact.name,
            "type": artifact.type,
            "state": artifact.state,
            "entity": artifact.entity,
            "project": artifact.project,
            "description": artifact.description,
            "aliases": sorted(artifact.aliases),
            "size": int(getattr(artifact, "size", 0)),
            "digest": getattr(artifact, "digest", None),
            "file": {
                "name": files[0].name,
                "size": int(getattr(files[0], "size", 0)),
                "digest": getattr(files[0], "digest", None),
            },
        },
    }


def close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-15):
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def compute_aggregate(rows):
    aggregate = {}
    paired = {}
    by_identity = {
        (row["hsc_center_condition"], row["seed"]): row for row in rows
    }
    for condition in CONDITIONS:
        aggregate[condition] = {}
        for metric in ("AUC.last", "AP.last"):
            values = [by_identity[(condition, seed)][metric] for seed in SEEDS]
            aggregate[condition][metric] = {
                "mean": statistics.mean(values),
                "sample_std_ddof1": statistics.stdev(values),
                "values_by_seed": {
                    str(seed): by_identity[(condition, seed)][metric]
                    for seed in SEEDS
                },
            }
    for condition in CONDITIONS[1:]:
        paired[condition] = {}
        for metric in ("AUC.last", "AP.last"):
            deltas = {
                str(seed): by_identity[(condition, seed)][metric]
                - by_identity[("default", seed)][metric]
                for seed in SEEDS
            }
            values = list(deltas.values())
            paired[condition][metric] = {
                "by_seed": deltas,
                "all_positive": all(value > 0 for value in values),
                "all_negative": all(value < 0 for value in values),
                "mean": statistics.mean(values),
                "sample_std_ddof1": statistics.stdev(values),
            }
    return aggregate, paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-id", required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--task-root", required=True, type=Path)
    parser.add_argument("--authoritative", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    authoritative = json.loads(args.authoritative.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    api = wandb.Api(timeout=60)
    sweep = api.sweep(f"HCCS/GGADFormer/{args.sweep_id}")
    if sweep.state != "FINISHED":
        raise AssertionError(f"independent replay saw sweep state {sweep.state}")

    observed = set()
    rows = []
    diagnostic_dir = args.task_root / "diagnostics"
    for run in sweep.runs:
        config = dict(run.config)
        condition = config.get("hsc_center_condition")
        seed = config.get("seed")
        identity = (condition, seed)
        if identity not in EXPECTED:
            raise AssertionError(f"unexpected trial {identity} in run {run.id}")
        if identity in observed:
            raise AssertionError(f"duplicate trial {identity}")
        observed.add(identity)
        if run.state != "finished":
            raise AssertionError(f"run {run.id} state is {run.state}")
        for key, expected in EXPECTED_CONFIG.items():
            if not equal(config.get(key), expected):
                raise AssertionError(
                    f"run {run.id} config.{key}: expected {expected!r}, got {config.get(key)!r}"
                )
        for key, expected in EXPECTED_AUDIT.items():
            if config.get(key) != expected:
                raise AssertionError(f"run {run.id} config.{key} mismatch")
        for key, expected in {
            "code_sha": args.code_sha,
            "execution_host": "HCCS-85",
            "protocol_identity": PROTOCOL_ID,
            "pair_id": f"tolokers:seed={seed}:data_split_seed=42",
        }.items():
            if config.get(key) != expected:
                raise AssertionError(f"run {run.id} config.{key} mismatch")

        history = []
        for item in run.scan_history(keys=["_step", "AUC", "AP"], page_size=500):
            if all(item.get(key) is not None for key in ("_step", "AUC", "AP")):
                history.append(
                    {
                        "step": int(item["_step"]),
                        "AUC": float(item["AUC"]),
                        "AP": float(item["AP"]),
                    }
                )
        expected_steps = list(range(0, 101, 10))
        if [item["step"] for item in history] != expected_steps:
            raise AssertionError(f"run {run.id} final-history schedule mismatch")
        auc = float(history[-1]["AUC"])
        ap = float(history[-1]["AP"])
        close(summary_value(run.summary, "AUC.last"), auc, f"run {run.id} AUC.last")
        close(summary_value(run.summary, "AP.last"), ap, f"run {run.id} AP.last")
        if summary_value(run.summary, "fixed_endpoint_epoch") != 100:
            raise AssertionError(f"run {run.id} fixed endpoint mismatch")
        if summary_value(run.summary, "run_valid") is not True:
            raise AssertionError(f"run {run.id} is not valid")
        wandb_artifact_audit = audit_wandb_artifacts(run)

        diagnostic_path = diagnostic_dir / f"{run.id}.json"
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        if sha256_file(diagnostic_path) != summary_value(run.summary, "diagnostic_sha256"):
            raise AssertionError(f"run {run.id} diagnostic hash mismatch")
        checkpoint_path = Path(diagnostic["checkpoint_path"])
        if not checkpoint_path.is_file():
            raise AssertionError(f"run {run.id} checkpoint is missing")
        if sha256_file(checkpoint_path) != summary_value(run.summary, "checkpoint_sha256"):
            raise AssertionError(f"run {run.id} checkpoint hash mismatch")
        checkpoint_identity = diagnostic["checkpoint_identity"]
        expected_identity = {
            "run_id": run.id,
            "dataset": "tolokers",
            "hsc_center_condition": condition,
            "seed": seed,
            "data_split_seed": 42,
            "code_sha": args.code_sha,
            "protocol_identity": PROTOCOL_ID,
            "final_training_epoch": 100,
        }
        if checkpoint_identity != expected_identity:
            raise AssertionError(f"run {run.id} checkpoint identity mismatch")
        close(diagnostic["final_metrics"]["AUC.last"], auc, f"run {run.id} diagnostic AUC")
        close(diagnostic["final_metrics"]["AP.last"], ap, f"run {run.id} diagnostic AP")
        hsc = diagnostic["hsc_diagnostics"]
        for key, summary_key in (
            ("ShellHit", "HSC.ShellHit"),
            ("inner_violation", "HSC.inner_violation"),
            ("outer_violation", "HSC.outer_violation"),
            ("mean_hsc_loss", "HSC.mean_loss"),
            ("center_shift_from_default", "HSC.center_shift_from_default"),
            ("center_shift_from_normal", "HSC.center_shift_from_normal"),
            ("sampled_anomaly_fraction", "HSC.sampled_anomaly_fraction"),
        ):
            close(hsc[key], summary_value(run.summary, summary_key), f"run {run.id} {summary_key}")
        for key in (
            "initial_model_sha256", "training_batch_trace_sha256",
            "pseudo_source_trace_sha256",
        ):
            if diagnostic[key] != summary_value(run.summary, key):
                raise AssertionError(f"run {run.id} {key} mismatch")
        for key, summary_key in (
            ("batch_trace_sha256", "HSC.diagnostic_batch_trace_sha256"),
            ("source_trace_sha256", "HSC.diagnostic_source_trace_sha256"),
        ):
            if hsc[key] != summary_value(run.summary, summary_key):
                raise AssertionError(f"run {run.id} {summary_key} mismatch")
        rows.append(
            {
                "run_id": run.id,
                "hsc_center_condition": condition,
                "seed": seed,
                "AUC.last": auc,
                "AP.last": ap,
                "history": history,
                "diagnostic_sha256": sha256_file(diagnostic_path),
                "initial_model_sha256": diagnostic["initial_model_sha256"],
                "training_batch_trace_sha256": diagnostic["training_batch_trace_sha256"],
                "pseudo_source_trace_sha256": diagnostic["pseudo_source_trace_sha256"],
                "diagnostic_batch_trace_sha256": hsc["batch_trace_sha256"],
                "diagnostic_source_trace_sha256": hsc["source_trace_sha256"],
                "hsc": {
                    key: float(hsc[key]) for key in (
                        "ShellHit", "inner_violation", "outer_violation", "mean_hsc_loss",
                        "center_shift_from_default", "center_shift_from_normal",
                        "sampled_anomaly_fraction",
                    )
                },
                "wandb_artifact_audit": wandb_artifact_audit,
            }
        )

    if observed != EXPECTED or len(rows) != 30:
        raise AssertionError("independent replay trial coverage mismatch")

    for seed in SEEDS:
        paired_rows = [row for row in rows if row["seed"] == seed]
        if len(paired_rows) != len(CONDITIONS):
            raise AssertionError(f"seed {seed}: pairing coverage mismatch")
        for field in (
            "initial_model_sha256", "training_batch_trace_sha256",
            "pseudo_source_trace_sha256", "diagnostic_batch_trace_sha256",
            "diagnostic_source_trace_sha256",
        ):
            if len({row[field] for row in paired_rows}) != 1:
                raise AssertionError(f"seed {seed}: pairing mismatch in {field}")

    aggregate, paired = compute_aggregate(rows)
    authoritative_runs = {
        (row["hsc_center_condition"], row["seed"]): row
        for row in authoritative["runs"]
    }
    for row in rows:
        key = (row["hsc_center_condition"], row["seed"])
        candidate = authoritative_runs[key]
        if candidate["run_id"] != row["run_id"]:
            raise AssertionError(f"{key}: run ID differs from collector")
        for metric in ("AUC.last", "AP.last"):
            close(candidate[metric], row[metric], f"{key} collector {metric}")
    for condition in CONDITIONS:
        for metric in ("AUC.last", "AP.last"):
            recorded = results["aggregate"]["tolokers"][condition][metric]
            close(recorded["mean"], aggregate[condition][metric]["mean"], f"{condition} {metric} mean")
            close(
                recorded["sample_std_ddof1"],
                aggregate[condition][metric]["sample_std_ddof1"],
                f"{condition} {metric} std",
            )
    for condition in CONDITIONS[1:]:
        recorded_condition = results["paired_deltas_vs_default"]["tolokers"][
            f"{condition}_minus_default"
        ]
        for metric in ("AUC.last", "AP.last"):
            recorded = recorded_condition[metric]
            close(recorded["mean"], paired[condition][metric]["mean"], f"{condition} {metric} paired mean")
            close(
                recorded["sample_std_ddof1"],
                paired[condition][metric]["sample_std_ddof1"],
                f"{condition} {metric} paired std",
            )
            by_seed = {str(row["seed"]): row["delta"] for row in recorded["by_seed"]}
            if set(by_seed) != set(paired[condition][metric]["by_seed"]):
                raise AssertionError(f"{condition} {metric} paired seed set mismatch")
            for seed, delta in paired[condition][metric]["by_seed"].items():
                close(by_seed[seed], delta, f"{condition} {metric} seed {seed} delta")

    payload = {
        "schema_version": 1,
        "status": "passed",
        "source": f'direct independent wandb.Api().sweep("HCCS/GGADFormer/{args.sweep_id}") replay plus local diagnostics',
        "sweep_id": args.sweep_id,
        "code_sha": args.code_sha,
        "run_count": len(rows),
        "coverage": {
            "missing": [],
            "duplicate": [],
            "unexpected": [],
            "all_finished": True,
            "all_runs_valid": True,
            "only_provider_generated_history_artifacts": True,
            "no_user_uploaded_or_used_wandb_artifacts": True,
            "pairing_hashes_match_within_seed": True,
        },
        "source_hashes": {
            "authoritative": sha256_file(args.authoritative),
            "results": sha256_file(args.results),
        },
        "aggregate": aggregate,
        "paired_deltas_vs_default": paired,
        "runs": sorted(rows, key=lambda row: (CONDITIONS.index(row["hsc_center_condition"]), row["seed"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "passed",
                "run_count": len(rows),
                "output": str(args.output),
                "output_sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
