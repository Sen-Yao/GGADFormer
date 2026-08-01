#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import sys


VARIANTS = ["none", "random_dir", "random_mag", "random_both", "constant_mag"]
SEEDS = list(range(5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    args = parser.parse_args()
    with open(args.results, encoding="utf-8") as handle:
        payload = json.load(handle)

    errors = []
    runs = payload.get("runs", [])
    identities = [(row.get("ablation_mode"), row.get("seed")) for row in runs]
    expected = [(variant, seed) for variant in VARIANTS for seed in SEEDS]
    if len(runs) != 25:
        errors.append("expected 25 runs, got {}".format(len(runs)))
    if sorted(identities) != sorted(expected):
        errors.append("trial identity set mismatch")
    for row in runs:
        seed = row.get("seed")
        if row.get("state") != "finished":
            errors.append("{} not finished".format(row.get("run_id")))
        if row.get("commit") != payload.get("code_sha"):
            errors.append("{} commit mismatch".format(row.get("run_id")))
        if row.get("final_step") != 100:
            errors.append("{} final step mismatch".format(row.get("run_id")))
        if row.get("validation_errors"):
            errors.append("{} collector validation failed".format(row.get("run_id")))
        if seed in SEEDS:
            if row.get("ablation_direction_seed") != seed * 1000003 + 1729:
                errors.append("{} direction seed mismatch".format(row.get("run_id")))
            if row.get("ablation_magnitude_seed") != seed * 1000003 + 7919:
                errors.append("{} magnitude seed mismatch".format(row.get("run_id")))
        for metric in ["AUC.last", "AP.last"]:
            value = row.get(metric)
            if value is None or not math.isfinite(float(value)):
                errors.append("{} invalid {}".format(row.get("run_id"), metric))

    recomputed = {}
    for variant in VARIANTS:
        rows = sorted([row for row in runs if row.get("ablation_mode") == variant], key=lambda row: row["seed"])
        recomputed[variant] = {}
        for metric in ["AUC.last", "AP.last"]:
            values = [float(row[metric]) for row in rows]
            mean = statistics.mean(values)
            sample_std = statistics.stdev(values)
            recorded = payload["aggregate"][variant][metric]
            if not math.isclose(mean, recorded["mean"], rel_tol=1e-15, abs_tol=1e-15):
                errors.append("{} {} mean mismatch".format(variant, metric))
            if not math.isclose(sample_std, recorded["sample_std"], rel_tol=1e-15, abs_tol=1e-15):
                errors.append("{} {} std mismatch".format(variant, metric))
            recomputed[variant][metric] = {"mean": mean, "sample_std": sample_std}

    report = {"valid": not errors, "errors": errors, "aggregate": recomputed}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())

