#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path

import wandb


EXPECTED_CODE_SHA = "65014dd11bed01b761aa7c3889c7718b7950884d"
EXPECTED_DATA_SHA = "2f502df4b87be8f8b5ed5ef8378876125c92b06afbc5b38ee58fe4b56b1b2023"
EXPECTED_GRADIENT_EPOCHS = [0, 1, 2, 5, 10, 20, 50, 100, 150]
EXPECTED_EVAL_EPOCHS = list(range(0, 151, 10))
EXPECTED_CELLS = {
    "control_2_20": (2.0, 20.0),
    "emb_only_0p1_20": (0.1, 20.0),
    "ring_only_2_1": (2.0, 1.0),
    "unified_0p1_1": (0.1, 1.0),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                records.append(json.loads(line))
            except Exception as exc:
                raise AssertionError(
                    "{}:{} invalid JSON: {}".format(path, line_number, exc)
                )
    return records


def assert_finite_tree(value, label):
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_tree(item, "{}.{}".format(label, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_tree(item, "{}[{}]".format(label, index))
    elif isinstance(value, float) and not math.isfinite(value):
        raise AssertionError("{} is non-finite".format(label))


def validate_diagnostics(path, run_id, summary):
    records = read_jsonl(path)
    if len(records) != 304:
        raise AssertionError("{} expected 304 records, got {}".format(run_id, len(records)))
    start, end = records[0], records[-1]
    updates = records[1:-1]
    if start.get("record_type") != "run_start" or end.get("record_type") != "run_end":
        raise AssertionError("{} diagnostics lacks start/end records".format(run_id))
    if start.get("run_id") != run_id or start.get("code_sha") != EXPECTED_CODE_SHA:
        raise AssertionError("{} diagnostics identity mismatch".format(run_id))
    if start.get("gradient_epochs") != EXPECTED_GRADIENT_EPOCHS:
        raise AssertionError("{} gradient schedule mismatch".format(run_id))

    expected_updates = [
        (epoch, batch, 2 * epoch + batch)
        for epoch in range(151)
        for batch in (0, 1)
    ]
    observed_updates = [
        (record.get("epoch"), record.get("batch_index"), record.get("global_update"))
        for record in updates
    ]
    if observed_updates != expected_updates:
        raise AssertionError("{} optimizer update coverage mismatch".format(run_id))

    for record in updates:
        if record.get("record_type") != "optimizer_update":
            raise AssertionError("{} has invalid update record".format(run_id))
        has_gradient = "gradient/parameter_count" in record
        expected_gradient = (
            record["batch_index"] == 0 and record["epoch"] in EXPECTED_GRADIENT_EPOCHS
        )
        if has_gradient != expected_gradient:
            raise AssertionError("{} gradient record coverage mismatch".format(run_id))
        assert_finite_tree(record, "{}.update".format(run_id))

    if end.get("optimizer_update_count") != 302 or end.get("fixed_final_epoch") != 150:
        raise AssertionError("{} terminal diagnostics mismatch".format(run_id))
    actual_sha = sha256_file(path)
    expected_summary = {
        "diagnostic/status": "complete",
        "diagnostic/optimizer_update_count": 302,
        "diagnostic/initial_model_sha256": start["initial_model_sha256"],
        "diagnostic/batch_trace_sha256": end["batch_trace_sha256"],
        "diagnostic/pseudo_source_trace_sha256": end["pseudo_source_trace_sha256"],
        "diagnostic/jsonl_sha256": actual_sha,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise AssertionError("{} summary {} mismatch".format(run_id, key))
    return {
        "path": str(path),
        "sha256": actual_sha,
        "record_count": len(records),
        "start": start,
        "end": end,
        "gradient_checkpoints": [
            record
            for record in updates
            if record["batch_index"] == 0 and record["epoch"] in EXPECTED_GRADIENT_EPOCHS
        ],
        "terminal_updates": [record for record in updates if record["epoch"] == 150],
    }


def selected_history(run):
    rows = []
    for raw in run.scan_history():
        row = {
            key: value
            for key, value in dict(raw).items()
            if key in {"_step", "AUC", "AP", "learning_rate"}
            or key.startswith("diagnostic/")
        }
        if row:
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-id", required=True)
    parser.add_argument("--diagnostics-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    api = wandb.Api()
    sweep = api.sweep("HCCS/GGADFormer/{}".format(args.sweep_id))
    runs = list(sweep.runs)
    if str(sweep.state).upper() != "FINISHED":
        raise AssertionError("sweep is not FINISHED: {}".format(sweep.state))
    if len(runs) != 4:
        raise AssertionError("expected 4 runs, got {}".format(len(runs)))

    observed = {}
    authoritative_runs = []
    for run in runs:
        config = dict(run.config)
        summary = dict(run.summary._json_dict)
        cell = config.get("mechanism_cell")
        if cell not in EXPECTED_CELLS or cell in observed:
            raise AssertionError("illegal or duplicate cell: {!r}".format(cell))
        if int(config.get("seed")) != 0 or str(run.state).lower() != "finished":
            raise AssertionError("{} seed/state mismatch".format(run.id))
        lambda_emb, lambda_hsc = EXPECTED_CELLS[cell]
        expected_config = {
            "variant": cell,
            "dataset": "elliptic",
            "lambda_rec_emb": lambda_emb,
            "ring_loss_weight": lambda_hsc,
            "num_epoch": 150,
            "batch_size": 32768,
            "data_split_seed": 42,
            "train_rate": 0.05,
            "mechanism_diagnostics": True,
            "code_sha": EXPECTED_CODE_SHA,
            "dataset_sha256": EXPECTED_DATA_SHA,
            "execution_host": "HCCS-85",
            "fixed_final_epoch_metric_policy": "AUC.last/AP.last at fixed training endpoint",
            "score_direction": "higher_logit_is_more_anomalous",
        }
        for key, expected in expected_config.items():
            actual = config.get(key)
            if isinstance(expected, float):
                if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                    raise AssertionError("{} config {} mismatch".format(run.id, key))
            elif actual != expected:
                raise AssertionError("{} config {} mismatch".format(run.id, key))

        history = selected_history(run)
        steps = [int(row["_step"]) for row in history]
        if steps != list(range(151)):
            raise AssertionError("{} W&B epoch history mismatch".format(run.id))
        eval_rows = [row for row in history if "AUC" in row or "AP" in row]
        if [int(row["_step"]) for row in eval_rows] != EXPECTED_EVAL_EPOCHS:
            raise AssertionError("{} evaluation epoch coverage mismatch".format(run.id))
        final_rows = [row for row in eval_rows if int(row["_step"]) == 150]
        if len(final_rows) != 1 or "AUC" not in final_rows[0] or "AP" not in final_rows[0]:
            raise AssertionError("{} lacks one fixed final metric row".format(run.id))
        for epoch, row in enumerate(history):
            for batch in (0, 1):
                key = "diagnostic/batch_{}/global_update".format(batch)
                if int(row.get(key, -1)) != 2 * epoch + batch:
                    raise AssertionError("{} W&B diagnostic coverage mismatch".format(run.id))

        diagnostic = validate_diagnostics(
            args.diagnostics_dir / "{}.jsonl".format(run.id), run.id, summary
        )
        observed[cell] = {
            "run_id": run.id,
            "url": run.url,
            "state": run.state,
            "final_epoch": 150,
            "AUC.last": float(final_rows[0]["AUC"]),
            "AP.last": float(final_rows[0]["AP"]),
            "diagnostics": diagnostic,
        }
        authoritative_runs.append({
            "run_id": run.id,
            "url": run.url,
            "state": run.state,
            "config": config,
            "summary": summary,
            "history": history,
            "diagnostics": diagnostic,
        })

    if set(observed) != set(EXPECTED_CELLS):
        raise AssertionError("four-cell coverage mismatch")
    init_hashes = {item["diagnostics"]["start"]["initial_model_sha256"] for item in observed.values()}
    batch_hashes = {item["diagnostics"]["end"]["batch_trace_sha256"] for item in observed.values()}
    source_hashes = {item["diagnostics"]["end"]["pseudo_source_trace_sha256"] for item in observed.values()}
    if len(init_hashes) != 1 or len(batch_hashes) != 1 or len(source_hashes) != 1:
        raise AssertionError("initialization or sampling trace mismatch across cells")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    authoritative = {
        "schema_version": 1,
        "sweep_id": args.sweep_id,
        "state": sweep.state,
        "runs": sorted(authoritative_runs, key=lambda item: item["config"]["mechanism_cell"]),
    }
    authoritative_path = args.output_dir / "authoritative-sweep.json"
    authoritative_path.write_text(
        json.dumps(authoritative, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    results = {
        "schema_version": 1,
        "evidence_level": "single_seed_mechanism_probe",
        "fixed_final_epoch": 150,
        "cells": observed,
        "shared_trace_identity": {
            "initial_model_sha256": next(iter(init_hashes)),
            "batch_trace_sha256": next(iter(batch_hashes)),
            "pseudo_source_trace_sha256": next(iter(source_hashes)),
        },
        "source_hashes": {
            "authoritative_sweep_sha256": sha256_file(authoritative_path),
            "diagnostic_jsonl_sha256": {
                cell: item["diagnostics"]["sha256"] for cell, item in observed.items()
            },
        },
    }
    results_path = args.output_dir / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "passed",
        "sweep_id": args.sweep_id,
        "authoritative_sweep_sha256": sha256_file(authoritative_path),
        "results_sha256": sha256_file(results_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
