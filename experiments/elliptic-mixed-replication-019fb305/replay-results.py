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
if authoritative["state"] != "FINISHED" or len(authoritative["runs"]) != 10:
    raise AssertionError("authoritative sweep is incomplete")

by_trial = {}
for run in authoritative["runs"]:
    final = [row for row in run["performance_history"] if int(row["_step"]) == 200]
    if len(final) != 1:
        raise AssertionError(f"run {run['run_id']} has invalid final history")
    trial = (run["variant"], int(run["seed"]))
    by_trial[trial] = {
        "AUC.last": float(final[0]["AUC"]),
        "AP.last": float(final[0]["AP"]),
    }

aggregate = {}
for variant in ("control_2_20", "unified_0p1_1"):
    aggregate[variant] = {}
    for metric in ("AUC.last", "AP.last"):
        values = [by_trial[(variant, seed)][metric] for seed in range(5)]
        aggregate[variant][metric] = {
            "mean": statistics.mean(values),
            "sample_std_ddof1": statistics.stdev(values),
        }

paired = []
for seed in range(5):
    paired.append({
        "seed": seed,
        "AUC.last": by_trial[("unified_0p1_1", seed)]["AUC.last"]
        - by_trial[("control_2_20", seed)]["AUC.last"],
        "AP.last": by_trial[("unified_0p1_1", seed)]["AP.last"]
        - by_trial[("control_2_20", seed)]["AP.last"],
    })
paired_mean = {
    metric: statistics.mean(row[metric] for row in paired)
    for metric in ("AUC.last", "AP.last")
}

assert_close(aggregate, results["aggregate"], "aggregate")
assert_close(paired, results["paired_differences_unified_minus_control"], "paired")
assert_close(
    paired_mean,
    results["paired_mean_difference_unified_minus_control"],
    "paired_mean",
)

replay = {
    "status": "passed",
    "authoritative_sweep_sha256": sha256_file(AUTHORITATIVE_PATH),
    "results_sha256": sha256_file(RESULTS_PATH),
    "fixed_final_step": 200,
    "aggregate": aggregate,
    "paired_mean_difference_unified_minus_control": paired_mean,
    "matches_results_json": True,
}
OUTPUT_PATH.write_text(json.dumps(replay, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "status": "passed",
    "output": str(OUTPUT_PATH),
    "output_sha256": sha256_file(OUTPUT_PATH),
}, sort_keys=True))
