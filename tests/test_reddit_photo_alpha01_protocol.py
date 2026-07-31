import ast
import importlib.util
from pathlib import Path
import sys


PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "reddit-photo-alpha01-paired-019fb832"
    / "protocol.py"
)


def load_protocol():
    spec = importlib.util.spec_from_file_location("alpha01_protocol", PROTOCOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_trial_identity_is_complete_and_unique():
    protocol = load_protocol()
    trials = protocol.expected_trials()
    identities = {
        (trial["dataset"], trial["declared_variant"], trial["seed"])
        for trial in trials
    }
    assert len(trials) == 20
    assert len(identities) == 20


def test_paired_configs_only_change_alpha():
    protocol = load_protocol()
    for dataset in protocol.DATASETS:
        for seed in protocol.SEEDS:
            control = protocol.resolve_config(dataset, "historical_control", seed)
            candidate = protocol.resolve_config(dataset, "alpha_0p1", seed)
            differing = {
                key for key in control if control[key] != candidate[key]
            }
            assert differing == {"declared_variant", "progregate_alpha"}


def test_alpha_values_and_final_budget_are_frozen():
    protocol = load_protocol()
    assert protocol.resolve_config("reddit", "historical_control", 0)["progregate_alpha"] == 0.0
    assert protocol.resolve_config("photo", "historical_control", 0)["progregate_alpha"] == 0.05
    for dataset in protocol.DATASETS:
        candidate = protocol.resolve_config(dataset, "alpha_0p1", 0)
        assert candidate["progregate_alpha"] == 0.1
        assert candidate["num_epoch"] == protocol.FINAL_STEP == 200
        assert candidate["data_split_seed"] == 42


def test_all_scientific_arguments_are_explicit():
    protocol = load_protocol()
    argv = protocol.scientific_argv("photo", "alpha_0p1", 4)
    assert "--dataset=photo" in argv
    assert "--seed=4" in argv
    assert "--progregate_alpha=0.1" in argv
    assert "--batch_size=128" in argv
    assert "--pp_k=6" in argv
    assert "--mean=0.02" in argv
    assert "--var=0.01" in argv
    assert "--visualize=false" in argv
    assert "--rec_error_filter_ratio=1" in argv
    assert all(value is not None for value in protocol.resolve_config("photo", "alpha_0p1", 4).values())


def test_all_resolved_keys_are_accepted_by_run_parser():
    protocol = load_protocol()
    tree = ast.parse((PROTOCOL_PATH.parents[2] / "run.py").read_text(encoding="utf-8"))
    accepted = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                if argument.value.startswith("--"):
                    accepted.add(argument.value[2:])
    resolved = set(protocol.resolve_config("reddit", "alpha_0p1", 0))
    resolved.remove("declared_variant")
    assert resolved <= accepted


def test_sweep_grid_has_exactly_twenty_trials():
    import yaml

    sweep_path = PROTOCOL_PATH.with_name("sweep.yaml")
    sweep = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
    parameters = sweep["parameters"]
    count = 1
    for axis in ("dataset", "variant", "seed"):
        count *= len(parameters[axis]["values"])
    assert count == 20


def test_replay_tracks_independent_metric_max_epochs():
    replay_path = PROTOCOL_PATH.with_name("replay-results.py")
    sys.modules["protocol"] = load_protocol()
    spec = importlib.util.spec_from_file_location("alpha01_replay", replay_path)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)

    protocol = load_protocol()
    runs = []
    for trial in protocol.expected_trials():
        seed = trial["seed"]
        runs.append({
            "dataset": trial["dataset"],
            "variant": trial["declared_variant"],
            "seed": seed,
            "run_id": "{}-{}-{}".format(trial["dataset"], trial["declared_variant"], seed),
            "url": "https://example.invalid/run",
            "history": [
                {"step": 190, "AUC": 0.8 + seed / 100.0, "AP": 0.2},
                {"step": 200, "AUC": 0.7, "AP": 0.3 + seed / 100.0},
            ],
        })
    evidence = {
        "protocol_id": protocol.PROTOCOL_ID,
        "sweep_id": "test-sweep",
        "runs": runs,
    }
    results = replay.replay(evidence, "synthetic")
    first = results["runs"][0]
    assert first["AUC"]["max_epoch"] == 190
    assert first["AP"]["max_epoch"] == 200
    assert len(results["runs"]) == 20
