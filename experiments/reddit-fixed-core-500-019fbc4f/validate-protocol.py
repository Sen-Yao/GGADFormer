#!/usr/bin/env python3
import json
from pathlib import Path

import yaml

from protocol import (
    CONFIRMATION_CONFIG_IDS,
    FIXED_CORE,
    HARD_RUN_LIMIT,
    PHASE_BUDGET,
    PROMOTION_CONFIG_IDS,
    PROTOCOL_ID,
    canonical_sha256,
    confirmation_trial_identities,
    promotion_trial_identities,
    resolve_config,
    screening_registry,
    validate_protocol,
)


def main():
    task_root = Path(__file__).resolve().parent
    manifest = yaml.safe_load((task_root / "manifest.yaml").read_text())
    screening_sweep = yaml.safe_load((task_root / "sweep-screening.yaml").read_text())
    promotion_sweep = yaml.safe_load((task_root / "sweep-promotion.yaml").read_text())
    confirmation_sweep = yaml.safe_load(
        (task_root / "sweep-confirmation.yaml").read_text()
    )
    screening_results = json.loads((task_root / "screening-results.json").read_text())
    promotion_results = json.loads((task_root / "promotion-results.json").read_text())

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
    created_records = manifest["budget"]["created_records"]
    if not 445 <= created_records <= 475:
        raise AssertionError("confirmation accounting is outside ordinals 446-475")
    if manifest["budget"]["remaining_records"] != HARD_RUN_LIMIT - created_records:
        raise AssertionError("remaining record accounting mismatch")
    if created_records + manifest["budget"]["technical_retry_records_reserved"] > HARD_RUN_LIMIT:
        raise AssertionError("confirmation accounting consumes retry reserve")

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

    claimed_payload_sha256 = screening_results["payload_sha256"]
    payload = dict(screening_results)
    payload.pop("payload_sha256")
    if canonical_sha256(payload) != claimed_payload_sha256:
        raise AssertionError("screening results payload digest mismatch")
    if screening_results["top12_config_ids"] != list(PROMOTION_CONFIG_IDS):
        raise AssertionError("promotion freeze differs from screening ranking")
    if screening_results["run_count"] != PHASE_BUDGET["screening"]:
        raise AssertionError("screening result count mismatch")

    promotion_params = promotion_sweep["parameters"]
    if promotion_sweep["metric"] != {"name": "Val/AUC.last", "goal": "maximize"}:
        raise AssertionError("promotion objective must be validation AUROC")
    if promotion_params["phase"]["value"] != "promotion":
        raise AssertionError("promotion phase identity mismatch")
    if promotion_params["config_id"]["values"] != list(PROMOTION_CONFIG_IDS):
        raise AssertionError("promotion sweep config freeze mismatch")
    if promotion_params["seed"]["values"] != [2, 3, 4, 5, 6]:
        raise AssertionError("promotion seed freeze mismatch")
    if len(promotion_trial_identities()) != PHASE_BUDGET["promotion"]:
        raise AssertionError("promotion trial count mismatch")
    for config_id in PROMOTION_CONFIG_IDS:
        for seed in promotion_params["seed"]["values"]:
            resolved = resolve_config("promotion", config_id, seed)
            if {key: resolved[key] for key in FIXED_CORE} != FIXED_CORE:
                raise AssertionError("promoted fixed-core mismatch")

    claimed_promotion_payload_sha256 = promotion_results["payload_sha256"]
    promotion_payload = dict(promotion_results)
    promotion_payload.pop("payload_sha256")
    if canonical_sha256(promotion_payload) != claimed_promotion_payload_sha256:
        raise AssertionError("promotion results payload digest mismatch")
    if promotion_results["run_count"] != PHASE_BUDGET["promotion"]:
        raise AssertionError("promotion result count mismatch")
    if promotion_results["forbidden_test_metric_count"] != 0:
        raise AssertionError("promotion result contains a forbidden test metric")

    by_config = {}
    for row in promotion_results["runs"]:
        by_config.setdefault(row["config_id"], []).append(row)
    replay = []
    for config_id, rows in by_config.items():
        if sorted(row["seed"] for row in rows) != [2, 3, 4, 5, 6]:
            raise AssertionError("promotion seed identity mismatch")
        replay.append((
            config_id,
            sum(row["val_auc_last"] for row in rows) / len(rows),
            sum(row["val_ap_last"] for row in rows) / len(rows),
        ))
    replay.sort(key=lambda item: (-item[1], -item[2], item[0]))
    frozen_confirmation = [item[0] for item in replay[:6]]
    if frozen_confirmation != list(CONFIRMATION_CONFIG_IDS):
        raise AssertionError("confirmation freeze differs from promotion replay")
    if promotion_results["top6_config_ids"] != frozen_confirmation:
        raise AssertionError("promotion artifact top six mismatch")

    confirmation_params = confirmation_sweep["parameters"]
    if confirmation_sweep["method"] != "grid":
        raise AssertionError("confirmation sweep must exhaust its frozen grid")
    if "early_terminate" in confirmation_sweep:
        raise AssertionError("confirmation sweep must not terminate early")
    if confirmation_sweep["metric"] != {"name": "Val/AUC.last", "goal": "maximize"}:
        raise AssertionError("confirmation objective is non-selective validation AUROC")
    if confirmation_params["phase"]["value"] != "confirmation":
        raise AssertionError("confirmation phase identity mismatch")
    if confirmation_params["config_id"]["values"] != list(CONFIRMATION_CONFIG_IDS):
        raise AssertionError("confirmation sweep config freeze mismatch")
    if confirmation_params["seed"]["values"] != [0, 1, 2, 3, 4]:
        raise AssertionError("confirmation seed freeze mismatch")
    if len(confirmation_trial_identities()) != PHASE_BUDGET["confirmation"]:
        raise AssertionError("confirmation trial count mismatch")
    for config_id in CONFIRMATION_CONFIG_IDS:
        for seed in confirmation_params["seed"]["values"]:
            resolved = resolve_config("confirmation", config_id, seed)
            if {key: resolved[key] for key in FIXED_CORE} != FIXED_CORE:
                raise AssertionError("confirmed fixed-core mismatch")
            if resolved["evaluation_protocol"] != "frozen_test":
                raise AssertionError("confirmation must use frozen test evaluation")

    print(json.dumps({
        "manifest_state": manifest["state"],
        "protocol": protocol,
        "screening_trials": len(config_ids) * len(seeds),
        "promotion_trials": len(promotion_trial_identities()),
        "promotion_config_ids": list(PROMOTION_CONFIG_IDS),
        "confirmation_trials": len(confirmation_trial_identities()),
        "confirmation_config_ids": list(CONFIRMATION_CONFIG_IDS),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
