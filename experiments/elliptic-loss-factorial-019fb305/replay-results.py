#!/usr/bin/env python3
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NEW_AUTH_PATH = ROOT / "authoritative-sweep.json"
PRIOR_AUTH_PATH = ROOT.parent / "elliptic-loss-unification-019fb305" / "authoritative-sweep.json"
PRIOR_RESULTS_PATH = ROOT.parent / "elliptic-loss-unification-019fb305" / "results.json"
RESULTS_PATH = ROOT / "results.json"
OUTPUT_PATH = ROOT / "replay.json"
EXPECTED_HASHES = {
    "new_authoritative": "c30424541e4e25200e239a8a07b60e48d75c3246964d8a644a5098f46ef7ddd8",
    "prior_authoritative": "efe5263a3666ad4287a3047ae722eadf1d401553a3644d081c60e51c05c6fb54",
    "prior_results": "8e27e0dc0f723c4ff7be729a5f73caa852021b39885b0da945342adf11d621b2",
    "results": "a047ca9bc442ff299b0e6a2bf3f3f4253ea8b16a7bc04a54c058e46ecceb5f34",
}
METRICS = ("AUC.last", "AP.last")
ALL_CELLS = ("control_2_20", "emb_only_0p1_20", "ring_only_2_1", "unified_0p1_1")
EFFECT_DEFINITIONS = {
    "lambda_rec_emb_low_minus_high_at_ring_20": {"emb_only_0p1_20": 1, "control_2_20": -1},
    "lambda_rec_emb_low_minus_high_at_ring_1": {"unified_0p1_1": 1, "ring_only_2_1": -1},
    "ring_loss_weight_low_minus_high_at_emb_2": {"ring_only_2_1": 1, "control_2_20": -1},
    "ring_loss_weight_low_minus_high_at_emb_0p1": {"unified_0p1_1": 1, "emb_only_0p1_20": -1},
    "interaction_difference_of_differences": {
        "unified_0p1_1": 1,
        "emb_only_0p1_20": -1,
        "ring_only_2_1": -1,
        "control_2_20": 1,
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_close(actual, expected, label):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"{label}: key mismatch")
        for key in expected:
            assert_close(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{label}: length mismatch")
        for index, expected_item in enumerate(expected):
            assert_close(actual[index], expected_item, f"{label}[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15):
            raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")
    elif actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


paths = {
    "new_authoritative": NEW_AUTH_PATH,
    "prior_authoritative": PRIOR_AUTH_PATH,
    "prior_results": PRIOR_RESULTS_PATH,
    "results": RESULTS_PATH,
}
actual_hashes = {name: sha256_file(path) for name, path in paths.items()}
if actual_hashes != EXPECTED_HASHES:
    raise AssertionError(f"evidence hash mismatch: {actual_hashes}")

new_auth = load(NEW_AUTH_PATH)
prior_auth = load(PRIOR_AUTH_PATH)
prior_results = load(PRIOR_RESULTS_PATH)
results = load(RESULTS_PATH)
if new_auth["state"] != "FINISHED" or prior_auth["state"] != "FINISHED":
    raise AssertionError("a source sweep is not FINISHED")

prior_history = {record["run_id"]: record for record in prior_auth["runs"]}
cells = {cell: {} for cell in ALL_CELLS}
for record in prior_results["runs"]:
    cell = record["variant"]
    history = prior_history[record["run_id"]]["history"]
    final = [row for row in history if int(row["_step"]) == 150]
    if len(final) != 1:
        raise AssertionError(f"prior run {record['run_id']} lacks one final row")
    cells[cell][int(record["seed"])] = {
        "AUC.last": float(final[0]["AUC"]),
        "AP.last": float(final[0]["AP"]),
    }

for record in new_auth["runs"]:
    history = record["history"]
    final = [row for row in history if int(row["_step"]) == 150]
    if len(final) != 1:
        raise AssertionError(f"new run {record['run_id']} lacks one final row")
    cells[record["factorial_cell"]][int(record["seed"])] = {
        "AUC.last": float(final[0]["AUC"]),
        "AP.last": float(final[0]["AP"]),
    }

for cell in ALL_CELLS:
    if sorted(cells[cell]) != list(range(5)):
        raise AssertionError(f"cell {cell} seed coverage mismatch")

aggregate = {}
for cell in ALL_CELLS:
    aggregate[cell] = {}
    for metric in METRICS:
        values = [cells[cell][seed][metric] for seed in range(5)]
        aggregate[cell][metric] = {
            "mean": statistics.mean(values),
            "sample_std_ddof1": statistics.stdev(values),
        }

effects = {}
for name, coefficients in EFFECT_DEFINITIONS.items():
    rows = []
    for seed in range(5):
        rows.append({
            "seed": seed,
            **{
                metric: sum(coefficients[cell] * cells[cell][seed][metric] for cell in coefficients)
                for metric in METRICS
            },
        })
    effects[name] = {
        "formula": " + ".join(f"{coefficient:+d}*{cell}" for cell, coefficient in coefficients.items()),
        "paired_differences_by_seed": rows,
        "paired_mean_difference": {
            metric: statistics.mean(row[metric] for row in rows) for metric in METRICS
        },
    }

for factor, names in {
    "lambda_rec_emb_low_minus_high_marginal": (
        "lambda_rec_emb_low_minus_high_at_ring_20",
        "lambda_rec_emb_low_minus_high_at_ring_1",
    ),
    "ring_loss_weight_low_minus_high_marginal": (
        "ring_loss_weight_low_minus_high_at_emb_2",
        "ring_loss_weight_low_minus_high_at_emb_0p1",
    ),
}.items():
    rows = []
    for seed in range(5):
        rows.append({
            "seed": seed,
            **{
                metric: statistics.mean(effects[name]["paired_differences_by_seed"][seed][metric] for name in names)
                for metric in METRICS
            },
        })
    effects[factor] = {
        "formula": f"0.5*({names[0]} + {names[1]})",
        "paired_differences_by_seed": rows,
        "paired_mean_difference": {
            metric: statistics.mean(row[metric] for row in rows) for metric in METRICS
        },
    }

assert_close(aggregate, results["factorial_cells"], "factorial_cells")
assert_close(effects, results["paired_effects"], "paired_effects")

replay = {
    "status": "passed",
    "source_hashes": actual_hashes,
    "fixed_final_step": 150,
    "factorial_cells": aggregate,
    "paired_effects": effects,
    "matches_results_json": True,
}
OUTPUT_PATH.write_text(json.dumps(replay, indent=2, sort_keys=False) + "\n", encoding="utf-8")
print(json.dumps({
    "status": replay["status"],
    "output": str(OUTPUT_PATH),
    "output_sha256": sha256_file(OUTPUT_PATH),
}, sort_keys=True))
