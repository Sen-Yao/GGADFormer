#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_CELLS = {
    "control_2_20",
    "emb_only_0p1_20",
    "ring_only_2_1",
    "unified_0p1_1",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    authoritative_path = args.evidence_dir / "authoritative-sweep.json"
    results_path = args.evidence_dir / "results.json"
    output_path = args.evidence_dir / "replay.json"
    authoritative = load(authoritative_path)
    results = load(results_path)

    actual_authoritative_sha = sha256_file(authoritative_path)
    if results["source_hashes"]["authoritative_sweep_sha256"] != actual_authoritative_sha:
        raise AssertionError("authoritative sweep hash mismatch")
    if str(authoritative["state"]).upper() != "FINISHED" or len(authoritative["runs"]) != 4:
        raise AssertionError("authoritative sweep state/count mismatch")

    replay_cells = {}
    init_hashes = set()
    batch_hashes = set()
    source_hashes = set()
    for run in authoritative["runs"]:
        config = run["config"]
        cell = config["mechanism_cell"]
        if cell not in EXPECTED_CELLS or cell in replay_cells:
            raise AssertionError("illegal or duplicate cell")
        final_rows = [
            row
            for row in run["history"]
            if int(row["_step"]) == 150 and "AUC" in row and "AP" in row
        ]
        if len(final_rows) != 1:
            raise AssertionError("{} final history mismatch".format(cell))
        diagnostic = run["diagnostics"]
        diagnostic_path = args.evidence_dir / "diagnostics" / "{}.jsonl".format(run["run_id"])
        expected_sha = results["source_hashes"]["diagnostic_jsonl_sha256"][cell]
        if sha256_file(diagnostic_path) != expected_sha or diagnostic["sha256"] != expected_sha:
            raise AssertionError("{} diagnostic hash mismatch".format(cell))
        records = [json.loads(line) for line in diagnostic_path.read_text(encoding="utf-8").splitlines()]
        if len(records) != 304 or records[0] != diagnostic["start"] or records[-1] != diagnostic["end"]:
            raise AssertionError("{} diagnostic record mismatch".format(cell))
        updates = records[1:-1]
        expected_updates = [
            (epoch, batch, 2 * epoch + batch)
            for epoch in range(151)
            for batch in (0, 1)
        ]
        observed_updates = [
            (row["epoch"], row["batch_index"], row["global_update"]) for row in updates
        ]
        if observed_updates != expected_updates:
            raise AssertionError("{} update replay mismatch".format(cell))
        init_hashes.add(records[0]["initial_model_sha256"])
        batch_hashes.add(records[-1]["batch_trace_sha256"])
        source_hashes.add(records[-1]["pseudo_source_trace_sha256"])
        replay_cells[cell] = {
            "run_id": run["run_id"],
            "AUC.last": float(final_rows[0]["AUC"]),
            "AP.last": float(final_rows[0]["AP"]),
            "gradient_checkpoint_count": sum(
                1 for row in updates if "gradient/parameter_count" in row
            ),
            "optimizer_update_count": len(updates),
        }

    if set(replay_cells) != EXPECTED_CELLS:
        raise AssertionError("cell coverage mismatch")
    if len(init_hashes) != 1 or len(batch_hashes) != 1 or len(source_hashes) != 1:
        raise AssertionError("shared trace identity mismatch")
    for cell, replay in replay_cells.items():
        result = results["cells"][cell]
        for key in ("run_id", "AUC.last", "AP.last"):
            if replay[key] != result[key]:
                raise AssertionError("{}.{} result mismatch".format(cell, key))

    replay = {
        "status": "passed",
        "source_hashes": {
            "authoritative_sweep_sha256": actual_authoritative_sha,
            "results_sha256": sha256_file(results_path),
        },
        "cell_coverage": sorted(replay_cells),
        "cells": replay_cells,
        "shared_trace_identity": results["shared_trace_identity"],
    }
    output_path.write_text(
        json.dumps(replay, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "passed",
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
