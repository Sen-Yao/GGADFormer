#!/usr/bin/env python3
import json
import math
import statistics
from datetime import datetime, timezone

import wandb

from protocol import (
    CONFIRMATION_CONFIG_IDS,
    FIXED_CORE,
    PHASE_BUDGET,
    PHASE_SEEDS,
    PROTOCOL_ID,
    SEARCH_SPACE,
    canonical_sha256,
    confirmation_trial_identities,
    resolve_config,
    screening_registry,
)


SWEEP_PATH = "HCCS/GGADFormer/219k2jj2"
EXPECTED_CODE_SHA = "73ae066e79ab20d09c69e3fc73b3ffb5e870fb8f"
EXPECTED_HOST = "HCCS-90"
PDF_REFERENCE = {"test_ap": 0.0441, "test_auc": 0.5782}


def summary_last(run, name):
    metric = run.summary.get(name)
    if metric is None:
        raise AssertionError("{} lacks summary metric {}".format(run.id, name))
    value = dict(metric).get("last")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AssertionError("{} has invalid {}.last".format(run.id, name))
    return float(value)


def assert_config_identity(run, config_id, seed):
    config = dict(run.config)
    expected = resolve_config("confirmation", config_id, seed)
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise AssertionError(
                "{} config mismatch for {}: expected {!r}, observed {!r}".format(
                    run.id, key, expected_value, config.get(key)
                )
            )
    metadata = {
        "code_sha": EXPECTED_CODE_SHA,
        "config_id": config_id,
        "execution_host": EXPECTED_HOST,
        "fixed_core_guard": True,
        "phase": "confirmation",
        "protocol_id": PROTOCOL_ID,
    }
    for key, expected_value in metadata.items():
        if config.get(key) != expected_value:
            raise AssertionError(
                "{} metadata mismatch for {}: expected {!r}, observed {!r}".format(
                    run.id, key, expected_value, config.get(key)
                )
            )
    if str(config.get("gpu_index")) not in {str(index) for index in range(8)}:
        raise AssertionError("{} has invalid GPU identity".format(run.id))
    return config


def collect_run(run):
    if run.state != "finished":
        raise AssertionError("{} is not finished: {}".format(run.id, run.state))
    config_id = run.config.get("config_id")
    seed = run.config.get("seed")
    if config_id not in CONFIRMATION_CONFIG_IDS:
        raise AssertionError("{} has unexpected config identity".format(run.id))
    if seed not in PHASE_SEEDS["confirmation"]:
        raise AssertionError("{} has unexpected seed identity".format(run.id))
    config = assert_config_identity(run, config_id, seed)

    val_auc = summary_last(run, "Val/AUC")
    val_ap = summary_last(run, "Val/AP")
    test_auc = summary_last(run, "Test/AUC")
    test_ap = summary_last(run, "Test/AP")

    summary_step = run.summary.get("_step")
    if summary_step != config["num_epoch"]:
        raise AssertionError(
            "{} summary is not at the fixed final epoch".format(run.id)
        )

    internal_history_artifacts = []
    for artifact in run.logged_artifacts():
        expected_name = "run-{}-history:v0".format(run.id)
        if (
            artifact.type != "wandb-history"
            or artifact.name != expected_name
            or not artifact.description.startswith(
                "Weights & Biases Run History Data for "
            )
        ):
            raise AssertionError(
                "{} logged undeclared artifact {} of type {}".format(
                    run.id, artifact.name, artifact.type
                )
            )
        internal_history_artifacts.append({
            "created_at": str(artifact.created_at),
            "name": artifact.name,
            "run_id": run.id,
            "size_bytes": artifact.size,
            "type": artifact.type,
        })

    row = {
        "config_id": config_id,
        "config_sha256": canonical_sha256(screening_registry()[config_id]),
        "run_id": run.id,
        "seed": seed,
        "state": run.state,
        "test_ap_last": test_ap,
        "test_auc_last": test_auc,
        "summary_step": summary_step,
        "val_ap_last": val_ap,
        "val_auc_last": val_auc,
    }
    return row, internal_history_artifacts


def aggregate(rows):
    registry = screening_registry()
    by_config = {config_id: [] for config_id in CONFIRMATION_CONFIG_IDS}
    for row in rows:
        by_config[row["config_id"]].append(row)

    aggregates = []
    for selection_rank, config_id in enumerate(CONFIRMATION_CONFIG_IDS, start=1):
        config_rows = sorted(by_config[config_id], key=lambda row: row["seed"])
        if [row["seed"] for row in config_rows] != list(
            PHASE_SEEDS["confirmation"]
        ):
            raise AssertionError("{} does not cover seeds 0-4".format(config_id))
        config = registry[config_id]
        aggregates.append({
            "config_id": config_id,
            "config_sha256": canonical_sha256(config),
            "mean_test_ap_last": statistics.mean(
                row["test_ap_last"] for row in config_rows
            ),
            "mean_test_auc_last": statistics.mean(
                row["test_auc_last"] for row in config_rows
            ),
            "mean_val_ap_last": statistics.mean(
                row["val_ap_last"] for row in config_rows
            ),
            "mean_val_auc_last": statistics.mean(
                row["val_auc_last"] for row in config_rows
            ),
            "run_ids": [row["run_id"] for row in config_rows],
            "search_axes": {key: config[key] for key in SEARCH_SPACE},
            "seeds": [row["seed"] for row in config_rows],
            "selection_rank": selection_rank,
            "std_test_ap_last": statistics.stdev(
                row["test_ap_last"] for row in config_rows
            ),
            "std_test_auc_last": statistics.stdev(
                row["test_auc_last"] for row in config_rows
            ),
            "std_val_ap_last": statistics.stdev(
                row["val_ap_last"] for row in config_rows
            ),
            "std_val_auc_last": statistics.stdev(
                row["val_auc_last"] for row in config_rows
            ),
        })
    return aggregates


def main():
    api = wandb.Api(timeout=60)
    sweep = api.sweep(SWEEP_PATH)
    if sweep.state != "FINISHED":
        raise AssertionError("confirmation sweep is not FINISHED")
    runs = list(sweep.runs)
    if len(runs) != PHASE_BUDGET["confirmation"]:
        raise AssertionError("confirmation run count mismatch")

    collected = [collect_run(run) for run in runs]
    rows = [item[0] for item in collected]
    internal_history_artifacts = [
        artifact for item in collected for artifact in item[1]
    ]
    order = {config_id: rank for rank, config_id in enumerate(CONFIRMATION_CONFIG_IDS)}
    rows.sort(key=lambda row: (order[row["config_id"]], row["seed"]))
    identities = [
        {"phase": "confirmation", "config_id": row["config_id"], "seed": row["seed"]}
        for row in rows
    ]
    if identities != confirmation_trial_identities():
        raise AssertionError("confirmation identities are missing or duplicated")

    payload = {
        "aggregates": aggregate(rows),
        "collected_at_utc": datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat().replace("+00:00", "Z"),
        "confirmation_config_ids": list(CONFIRMATION_CONFIG_IDS),
        "confirmation_config_ids_sha256": canonical_sha256(
            CONFIRMATION_CONFIG_IDS
        ),
        "failed_run_count": 0,
        "fixed_core": FIXED_CORE,
        "application_logged_artifact_count": 0,
        "pdf_reference": PDF_REFERENCE,
        "phase": "confirmation",
        "protocol_id": PROTOCOL_ID,
        "reporting_policy": (
            "pre-frozen config order; fixed-final mean and sample standard "
            "deviation; no Test-based reranking"
        ),
        "rows_sha256": canonical_sha256(rows),
        "run_count": len(rows),
        "run_ids_sha256": canonical_sha256(sorted(row["run_id"] for row in rows)),
        "runs": rows,
        "schema_version": 1,
        "sweep_id": SWEEP_PATH.rsplit("/", 1)[-1],
        "sweep_state_at_collection": sweep.state,
        "test_final_epoch_audit": (
            "exact execution SHA source guard plus W&B final summary step; "
            "scan_history is intentionally not used because the audit API "
            "materializes a server-side wandb-history object"
        ),
        "unique_identity_count": len(identities),
        "valid_run_count": len(rows),
        "wandb_internal_history_artifact_count": len(
            internal_history_artifacts
        ),
        "wandb_internal_history_artifacts": internal_history_artifacts,
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
