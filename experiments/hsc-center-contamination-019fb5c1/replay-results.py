#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path


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
        for key, value in expected.items():
            assert_close(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{label}: length mismatch")
        for index, value in enumerate(expected):
            assert_close(actual[index], value, f"{label}[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15):
            raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")
    elif actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authoritative", required=True, type=Path)
    parser.add_argument("--authoritative-sha256", required=True)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--results-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if sha256_file(args.authoritative) != args.authoritative_sha256:
        raise AssertionError("authoritative input hash mismatch")
    if sha256_file(args.results) != args.results_sha256:
        raise AssertionError("results input hash mismatch")

    authoritative = json.loads(args.authoritative.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    if authoritative["state"] != "FINISHED":
        raise AssertionError("authoritative sweep is not FINISHED")

    namespace = {}
    collector_path = Path(__file__).with_name("collect-evidence.py")
    exec(compile(collector_path.read_text(encoding="utf-8"), str(collector_path), "exec"), namespace)
    aggregate, paired = namespace["aggregate_records"](authoritative["runs"])
    assert_close(aggregate, results["aggregate"], "aggregate")
    assert_close(paired, results["paired_deltas_vs_default"], "paired")

    replay = {
        "status": "passed",
        "source_hashes": {
            "authoritative": args.authoritative_sha256,
            "results": args.results_sha256,
        },
        "sweep_id": authoritative["sweep_id"],
        "run_count": len(authoritative["runs"]),
        "decision_threshold": None,
        "aggregate": aggregate,
        "paired_deltas_vs_default": paired,
        "matches_results_json": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(replay, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(args.output),
                "output_sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
