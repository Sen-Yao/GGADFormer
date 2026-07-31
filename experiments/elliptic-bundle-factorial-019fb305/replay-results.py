#!/usr/bin/env python3
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AUTHORITATIVE_PATH = ROOT / "authoritative-sweep.json"
RESULTS_PATH = ROOT / "results.json"
OUTPUT_PATH = ROOT / "replay.json"
LEVELS = ("current", "mixed")
SEEDS = tuple(range(5))
METRICS = ("AUC.last", "AP.last")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_close(actual, expected, label):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"{label}: key mismatch")
        for key in expected:
            assert_close(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{label}: length mismatch")
        for index, item in enumerate(expected):
            assert_close(actual[index], item, f"{label}[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15):
            raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")
    elif actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


authoritative = json.loads(AUTHORITATIVE_PATH.read_text(encoding="utf-8"))
results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
if authoritative["state"] != "FINISHED" or len(authoritative["runs"]) != 20:
    raise AssertionError("authoritative sweep is incomplete")

by_trial = {}
for run in authoritative["runs"]:
    final_step = int(run["final_step"])
    final = [row for row in run["performance_history"] if int(row["_step"]) == final_step]
    if len(final) != 1:
        raise AssertionError(f"run {run['run_id']} has invalid final history")
    trial = (run["optimization_bundle"], run["propagation_bundle"], int(run["seed"]))
    by_trial[trial] = {"AUC.last": float(final[0]["AUC"]), "AP.last": float(final[0]["AP"])}

cell_aggregate = {}
for optimization in LEVELS:
    for propagation in LEVELS:
        cell = f"opt_{optimization}__prop_{propagation}"
        cell_aggregate[cell] = {}
        for metric in METRICS:
            values = [by_trial[(optimization, propagation, seed)][metric] for seed in SEEDS]
            cell_aggregate[cell][metric] = {
                "mean": statistics.mean(values),
                "sample_std_ddof1": statistics.stdev(values),
            }

effects_by_seed = []
for seed in SEEDS:
    row = {"seed": seed}
    for metric in METRICS:
        cc = by_trial[("current", "current", seed)][metric]
        cm = by_trial[("current", "mixed", seed)][metric]
        mc = by_trial[("mixed", "current", seed)][metric]
        mm = by_trial[("mixed", "mixed", seed)][metric]
        row[metric] = {
            "optimization_main_mixed_minus_current": ((mc - cc) + (mm - cm)) / 2,
            "propagation_main_mixed_minus_current": ((cm - cc) + (mm - mc)) / 2,
            "optimization_at_current_propagation": mc - cc,
            "optimization_at_mixed_propagation": mm - cm,
            "propagation_at_current_optimization": cm - cc,
            "propagation_at_mixed_optimization": mm - mc,
            "interaction_difference_in_differences": mm - mc - cm + cc,
            "mixed_cell_minus_current_cell": mm - cc,
        }
    effects_by_seed.append(row)

effect_aggregate = {}
for metric in METRICS:
    effect_aggregate[metric] = {}
    for effect in effects_by_seed[0][metric]:
        values = [row[metric][effect] for row in effects_by_seed]
        effect_aggregate[metric][effect] = {
            "mean": statistics.mean(values),
            "sample_std_ddof1": statistics.stdev(values),
        }

assert_close(cell_aggregate, results["cell_aggregate"], "cell_aggregate")
assert_close(effects_by_seed, results["effects_by_seed"], "effects_by_seed")
assert_close(effect_aggregate, results["effect_aggregate"], "effect_aggregate")

replay = {
    "status": "passed",
    "authoritative_sweep_sha256": sha256_file(AUTHORITATIVE_PATH),
    "results_sha256": sha256_file(RESULTS_PATH),
    "cell_aggregate": cell_aggregate,
    "effect_aggregate": effect_aggregate,
    "matches_results_json": True,
}
OUTPUT_PATH.write_text(json.dumps(replay, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "passed", "output": str(OUTPUT_PATH), "output_sha256": sha256_file(OUTPUT_PATH)}, sort_keys=True))

