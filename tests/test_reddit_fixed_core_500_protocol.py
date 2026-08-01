import importlib.util
from pathlib import Path

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


def test_confirmation_is_the_only_test_reading_phase():
    protocol = load_protocol()
    assert protocol.resolve_config(
        "smoke", "smoke-control", 0
    )["evaluation_protocol"] == "validation_only"
    assert protocol.resolve_config(
        "promotion", "cfg-000", 2
    )["evaluation_protocol"] == "validation_only"
    assert protocol.resolve_config(
        "confirmation", "cfg-000", 0
    )["evaluation_protocol"] == "frozen_test"


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
