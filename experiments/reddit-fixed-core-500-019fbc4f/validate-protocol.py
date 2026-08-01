#!/usr/bin/env python3
import json
from pathlib import Path

import yaml

from protocol import (
    FIXED_CORE,
    HARD_RUN_LIMIT,
    PHASE_BUDGET,
    PROTOCOL_ID,
    resolve_config,
    screening_registry,
    validate_protocol,
)


def main():
    task_root = Path(__file__).resolve().parent
    manifest = yaml.safe_load((task_root / "manifest.yaml").read_text())
    screening_sweep = yaml.safe_load((task_root / "sweep-screening.yaml").read_text())

    protocol = validate_protocol()
    if manifest["protocol"]["id"] != PROTOCOL_ID:
        raise AssertionError("manifest protocol identity mismatch")
    if manifest["budget"]["hard_limit"] != HARD_RUN_LIMIT:
        raise AssertionError("manifest hard budget mismatch")
    if manifest["budget"]["phase_allocation"] != PHASE_BUDGET:
        raise AssertionError("manifest phase allocation mismatch")
    if manifest["fixed_core"] != FIXED_CORE:
        raise AssertionError("manifest fixed core mismatch")
    if manifest["budget"]["allocated_plus_reserved"] > HARD_RUN_LIMIT:
        raise AssertionError("manifest allocates more than 500 records")

    params = screening_sweep["parameters"]
    config_ids = params["config_id"]["values"]
    seeds = params["seed"]["values"]
    if set(config_ids) != set(screening_registry()):
        raise AssertionError("screening sweep config registry mismatch")
    if len(config_ids) * len(seeds) != PHASE_BUDGET["screening"]:
        raise AssertionError("screening sweep trial count mismatch")
    if screening_sweep["metric"] != {"name": "Val/AUC.last", "goal": "maximize"}:
        raise AssertionError("screening objective must be validation AUROC")

    for config_id in config_ids:
        for seed in seeds:
            resolved = resolve_config("screening", config_id, seed)
            if {key: resolved[key] for key in FIXED_CORE} != FIXED_CORE:
                raise AssertionError("resolved fixed-core mismatch")

    print(json.dumps({
        "manifest_state": manifest["state"],
        "protocol": protocol,
        "screening_trials": len(config_ids) * len(seeds),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
