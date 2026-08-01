import importlib.util
import json
from pathlib import Path
import statistics

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "experiments" / "reddit-fixed-core-500-019fbc4f"


def load_protocol():
    spec = importlib.util.spec_from_file_location(
        "reddit_fixed_core_protocol", TASK_ROOT / "protocol.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_budget_is_an_exact_hard_cap():
    protocol = load_protocol()
    assert protocol.HARD_RUN_LIMIT == 500
    assert sum(protocol.PHASE_BUDGET.values()) == 500
    assert protocol.PHASE_BUDGET == {
        "smoke": 1,
        "screening": 384,
        "promotion": 60,
        "confirmation": 30,
        "technical_retry_reserve": 25,
    }


def test_every_resolved_screening_config_has_fixed_core():
    protocol = load_protocol()
    registry = protocol.screening_registry()
    assert len(registry) == 192
    assert len({protocol.canonical_sha256(value) for value in registry.values()}) == 192
    for config_id in registry:
        for seed in protocol.PHASE_SEEDS["screening"]:
            config = protocol.resolve_config("screening", config_id, seed)
            assert {key: config[key] for key in protocol.FIXED_CORE} == protocol.FIXED_CORE
            assert config["evaluation_protocol"] == "validation_only"
            assert config["wandb_log_training_metrics"] is False


def test_promotion_is_validation_only_and_confirmation_is_frozen_test():
    protocol = load_protocol()
    assert protocol.resolve_config(
        "smoke", "smoke-control", 0
    )["evaluation_protocol"] == "validation_only"
    for config_id in protocol.PROMOTION_CONFIG_IDS:
        assert protocol.resolve_config(
            "promotion", config_id, 2
        )["evaluation_protocol"] == "validation_only"
    assert len(protocol.CONFIRMATION_CONFIG_IDS) == 6
    assert len(set(protocol.CONFIRMATION_CONFIG_IDS)) == 6
    assert set(protocol.CONFIRMATION_CONFIG_IDS).issubset(
        protocol.PROMOTION_CONFIG_IDS
    )
    for config_id in protocol.CONFIRMATION_CONFIG_IDS:
        config = protocol.resolve_config("confirmation", config_id, 0)
        assert config["evaluation_protocol"] == "frozen_test"
        assert {key: config[key] for key in protocol.FIXED_CORE} == protocol.FIXED_CORE
    rejected = next(
        config_id
        for config_id in protocol.PROMOTION_CONFIG_IDS
        if config_id not in protocol.CONFIRMATION_CONFIG_IDS
    )
    with pytest.raises(ValueError, match="not confirmed"):
        protocol.resolve_config("confirmation", rejected, 0)
    with pytest.raises(ValueError, match="unsupported confirmation seed"):
        protocol.resolve_config("confirmation", protocol.CONFIRMATION_CONFIG_IDS[0], 5)


def test_screening_sweep_identity_and_trial_count():
    protocol = load_protocol()
    sweep = yaml.safe_load((TASK_ROOT / "sweep-screening.yaml").read_text())
    assert sweep["method"] == "grid"
    assert sweep["metric"] == {"name": "Val/AUC.last", "goal": "maximize"}
    parameters = sweep["parameters"]
    assert parameters["phase"]["value"] == "screening"
    assert set(parameters["config_id"]["values"]) == set(protocol.screening_registry())
    assert parameters["seed"]["values"] == [0, 1]
    assert len(parameters["config_id"]["values"]) * 2 == 384


def test_protocol_digests_are_stable():
    protocol = load_protocol()
    summary = protocol.validate_protocol()
    assert summary["screening_registry_sha256"] == (
        "4c1895d6c27189774fee3b1b8a4f26b10375ceccc2dcb810cd6b23a76e3c80ce"
    )
    assert summary["screening_trials_sha256"] == (
        "6c5b6ea56660443f623a95bbda4494ecf9e3ebf899072f4640f80929037cd273"
    )
    assert summary["promotion_config_ids_sha256"] == (
        "647346bd43308362620f9e63af6dda07f3b6823b225bc68b6551a626c53442ec"
    )
    assert summary["confirmation_config_ids_sha256"] == (
        "adfb7edae5c7e46f0363ccf7e2e7db6332da5ada79620ddea4241ce1ea9f1a7c"
    )
    assert summary["confirmation_trials_sha256"] == (
        "95fbdac5ad1df47a47adf91a16df12f4618662f3fc5ebcce544d360e692872d0"
    )


def test_promotion_sweep_matches_frozen_screening_ranking():
    protocol = load_protocol()
    results = json.loads((TASK_ROOT / "screening-results.json").read_text())
    sweep = yaml.safe_load((TASK_ROOT / "sweep-promotion.yaml").read_text())
    by_config = {}
    for row in results["runs"]:
        by_config.setdefault(row["config_id"], []).append(row)
    replay = []
    for config_id, rows in by_config.items():
        replay.append((
            config_id,
            statistics.mean(row["val_auc_last"] for row in rows),
            statistics.mean(row["val_ap_last"] for row in rows),
        ))
    replay.sort(key=lambda item: (-item[1], -item[2], item[0]))
    promoted = [item[0] for item in replay[:12]]
    assert promoted == list(protocol.PROMOTION_CONFIG_IDS)
    assert results["top12_config_ids"] == promoted
    assert sweep["parameters"]["config_id"]["values"] == promoted
    assert sweep["parameters"]["seed"]["values"] == [2, 3, 4, 5, 6]
    assert len(protocol.promotion_trial_identities()) == 60


def test_confirmation_sweep_matches_frozen_promotion_ranking():
    protocol = load_protocol()
    results = json.loads((TASK_ROOT / "promotion-results.json").read_text())
    sweep = yaml.safe_load((TASK_ROOT / "sweep-confirmation.yaml").read_text())
    by_config = {}
    for row in results["runs"]:
        by_config.setdefault(row["config_id"], []).append(row)
    replay = []
    for config_id, rows in by_config.items():
        replay.append((
            config_id,
            statistics.mean(row["val_auc_last"] for row in rows),
            statistics.mean(row["val_ap_last"] for row in rows),
        ))
    replay.sort(key=lambda item: (-item[1], -item[2], item[0]))
    confirmed = [item[0] for item in replay[:6]]
    assert confirmed == list(protocol.CONFIRMATION_CONFIG_IDS)
    assert results["top6_config_ids"] == confirmed
    assert sweep["method"] == "grid"
    assert "early_terminate" not in sweep
    assert sweep["metric"] == {"name": "Val/AUC.last", "goal": "maximize"}
    assert sweep["parameters"]["phase"]["value"] == "confirmation"
    assert sweep["parameters"]["config_id"]["values"] == confirmed
    assert sweep["parameters"]["seed"]["values"] == [0, 1, 2, 3, 4]
    assert len(protocol.confirmation_trial_identities()) == 30


def test_validation_path_does_not_index_test_labels_or_carry_training_labels():
    run_source = (ROOT / "run.py").read_text()
    utils_source = (ROOT / "utils.py").read_text()
    assert "Data.TensorDataset(concated_input_features, all_node_indices)" in run_source
    assert "Test labels withheld by validation-only protocol" in utils_source
