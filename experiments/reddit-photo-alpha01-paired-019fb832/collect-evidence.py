#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import wandb

from protocol import FINAL_STEP, PROTOCOL_ID, canonical_sha256, expected_trials


ENTITY = "HCCS"
PROJECT = "GGADFormer"
METRIC_KEYS = ("AUC", "AP")


def normalize_scalar(value):
    if hasattr(value, "item"):
        value = value.item()
    return value


def validate_config(run, expected, expected_code_sha, execution_host):
    observed = dict(run.config)
    errors = []
    for key, expected_value in expected.items():
        observed_value = normalize_scalar(observed.get(key))
        if observed_value != expected_value:
            errors.append(
                "{} expected {!r}, observed {!r}".format(
                    key, expected_value, observed_value
                )
            )
    required_metadata = {
        "protocol_id": PROTOCOL_ID,
        "code_sha": expected_code_sha,
        "execution_host": execution_host,
    }
    for key, expected_value in required_metadata.items():
        if observed.get(key) != expected_value:
            errors.append(
                "{} expected {!r}, observed {!r}".format(
                    key, expected_value, observed.get(key)
                )
            )
    gpu_index = str(observed.get("gpu_index", ""))
    if not gpu_index.isdigit():
        errors.append("gpu_index must be one integer, observed {!r}".format(gpu_index))
    if errors:
        raise RuntimeError("run {} config mismatch: {}".format(run.id, "; ".join(errors)))
    return observed


def read_history(run):
    rows = []
    for row in run.scan_history(keys=["_step", "AUC", "AP"], page_size=1000):
        if any(key not in row for key in ("_step", "AUC", "AP")):
            continue
        rows.append({
            "step": int(row["_step"]),
            "AUC": float(row["AUC"]),
            "AP": float(row["AP"]),
        })
    rows.sort(key=lambda item: item["step"])
    if not rows:
        raise RuntimeError("run {} has no joint AUC/AP history".format(run.id))
    if rows[-1]["step"] != FINAL_STEP:
        raise RuntimeError(
            "run {} final step expected {}, observed {}".format(
                run.id, FINAL_STEP, rows[-1]["step"]
            )
        )
    return rows


def collect(sweep_id, expected_code_sha, execution_host):
    api = wandb.Api(timeout=30)
    sweep_path = "{}/{}/{}".format(ENTITY, PROJECT, sweep_id)
    sweep = api.sweep(sweep_path)
    expected = {
        (trial["dataset"], trial["declared_variant"], trial["seed"]): trial
        for trial in expected_trials()
    }
    observed = {}
    rejected = []
    for run in sweep.runs:
        identity = (
            run.config.get("dataset"),
            run.config.get("declared_variant"),
            normalize_scalar(run.config.get("seed")),
        )
        if identity not in expected:
            rejected.append({"run_id": run.id, "identity": list(identity), "state": run.state})
            continue
        if identity in observed:
            raise RuntimeError("duplicate expected identity: {!r}".format(identity))
        if str(run.state).lower() != "finished":
            raise RuntimeError("run {} is not finished: {}".format(run.id, run.state))
        config = validate_config(run, expected[identity], expected_code_sha, execution_host)
        history = read_history(run)
        observed[identity] = {
            "dataset": identity[0],
            "variant": identity[1],
            "seed": identity[2],
            "run_id": run.id,
            "name": run.name,
            "url": run.url,
            "state": str(run.state).lower(),
            "created_at": str(run.created_at),
            "code_sha": config["code_sha"],
            "execution_host": config["execution_host"],
            "gpu_index": str(config["gpu_index"]),
            "config": {key: normalize_scalar(config.get(key)) for key in sorted(expected[identity])},
            "config_sha256": canonical_sha256(
                {key: normalize_scalar(config.get(key)) for key in sorted(expected[identity])}
            ),
            "history": history,
            "history_sha256": canonical_sha256(history),
        }

    missing = sorted(set(expected) - set(observed))
    if missing:
        raise RuntimeError("missing expected identities: {!r}".format(missing))
    if rejected:
        raise RuntimeError("sweep contains undeclared runs: {!r}".format(rejected))

    runs = [observed[key] for key in sorted(observed)]
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "entity": ENTITY,
        "project": PROJECT,
        "sweep_id": sweep_id,
        "sweep_url": "https://wandb.ai/{}/{}/sweeps/{}".format(ENTITY, PROJECT, sweep_id),
        "expected_code_sha": expected_code_sha,
        "execution_host": execution_host,
        "expected_trial_count": len(expected),
        "observed_trial_count": len(runs),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-id", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--execution-host", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evidence = collect(args.sweep_id, args.expected_code_sha, args.execution_host)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "observed_trial_count": evidence["observed_trial_count"],
        "sha256": canonical_sha256(evidence),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
