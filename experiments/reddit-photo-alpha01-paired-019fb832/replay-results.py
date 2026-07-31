#!/usr/bin/env python
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from protocol import DATASETS, FINAL_STEP, SEEDS, VARIANTS


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values):
    return {
        "mean": statistics.mean(values),
        "sample_std_ddof1": statistics.stdev(values),
    }


def metric_record(history, metric):
    final_rows = [row for row in history if row["step"] == FINAL_STEP]
    if len(final_rows) != 1:
        raise RuntimeError(
            "expected one {} row at step {}, observed {}".format(
                metric, FINAL_STEP, len(final_rows)
            )
        )
    maximum = max(history, key=lambda row: (row[metric], -row["step"]))
    return {
        "last": float(final_rows[0][metric]),
        "max": float(maximum[metric]),
        "max_epoch": int(maximum["step"]),
    }


def replay(evidence, evidence_sha256):
    runs = evidence["runs"]
    expected_identities = {
        (dataset, variant, seed)
        for dataset in DATASETS
        for variant in VARIANTS
        for seed in SEEDS
    }
    observed = {}
    for run in runs:
        identity = (run["dataset"], run["variant"], int(run["seed"]))
        if identity in observed:
            raise RuntimeError("duplicate identity in evidence: {!r}".format(identity))
        observed[identity] = {
            "dataset": identity[0],
            "variant": identity[1],
            "seed": identity[2],
            "run_id": run["run_id"],
            "url": run["url"],
            "AUC": metric_record(run["history"], "AUC"),
            "AP": metric_record(run["history"], "AP"),
        }
    if set(observed) != expected_identities:
        raise RuntimeError(
            "identity mismatch: missing={!r}, extra={!r}".format(
                sorted(expected_identities - set(observed)),
                sorted(set(observed) - expected_identities),
            )
        )

    aggregate = {}
    paired = []
    for dataset in DATASETS:
        aggregate[dataset] = {}
        for variant in VARIANTS:
            variant_runs = [observed[(dataset, variant, seed)] for seed in SEEDS]
            aggregate[dataset][variant] = {
                "AUC.last": summary([run["AUC"]["last"] for run in variant_runs]),
                "AP.last": summary([run["AP"]["last"] for run in variant_runs]),
                "AUC.max_diagnostic": summary([run["AUC"]["max"] for run in variant_runs]),
                "AP.max_diagnostic": summary([run["AP"]["max"] for run in variant_runs]),
            }
        for seed in SEEDS:
            control = observed[(dataset, "historical_control", seed)]
            candidate = observed[(dataset, "alpha_0p1", seed)]
            paired.append({
                "dataset": dataset,
                "seed": seed,
                "delta_AUC.last_alpha_0p1_minus_control": candidate["AUC"]["last"] - control["AUC"]["last"],
                "delta_AP.last_alpha_0p1_minus_control": candidate["AP"]["last"] - control["AP"]["last"],
            })
        dataset_pairs = [row for row in paired if row["dataset"] == dataset]
        aggregate[dataset]["paired_delta_alpha_0p1_minus_control"] = {
            "AUC.last": summary([row["delta_AUC.last_alpha_0p1_minus_control"] for row in dataset_pairs]),
            "AP.last": summary([row["delta_AP.last_alpha_0p1_minus_control"] for row in dataset_pairs]),
        }

    return {
        "schema_version": 1,
        "protocol_id": evidence["protocol_id"],
        "sweep_id": evidence["sweep_id"],
        "source_evidence_sha256": evidence_sha256,
        "primary_metric_policy": "fixed step 200",
        "diagnostic_metric_policy": "independent per-metric maximum and epoch",
        "runs": [observed[key] for key in sorted(observed)],
        "aggregate": aggregate,
        "paired_differences": paired,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evidence_path = Path(args.evidence)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    results = replay(evidence, file_sha256(evidence_path))
    output = Path(args.output)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "run_count": len(results["runs"]),
        "source_evidence_sha256": results["source_evidence_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
