#!/usr/bin/env python3
import hashlib
import itertools
import json
import random


PROTOCOL_ID = "reddit-fixed-core-500-019fbc4f"
HARD_RUN_LIMIT = 500
SEARCH_SEED = 20260801
SCREENING_CONFIG_COUNT = 192

PHASE_BUDGET = {
    "smoke": 1,
    "screening": 384,
    "promotion": 60,
    "confirmation": 30,
    "technical_retry_reserve": 25,
}

PHASE_SEEDS = {
    "smoke": (0,),
    "screening": (0, 1),
    "promotion": (2, 3, 4, 5, 6),
    "confirmation": (0, 1, 2, 3, 4),
}

FIXED_CORE = {
    "dataset": "reddit",
    "progregate_alpha": 0.0,
    "lambda_rec_emb": 0.1,
    "ring_R_max": 1.0,
}

COMMON_CONFIG = {
    "GT_attention_dropout": 0.4,
    "GT_dropout": 0.4,
    "GT_ffn_dim": 256,
    "GT_num_heads": 2,
    "GT_num_layers": 3,
    "GNA_temp": 1.0,
    "ablation_mode": "none",
    "auc_test_rounds": 256,
    "bce_loss_weight": 1.0,
    "con_loss_temp": 10.0,
    "con_loss_weight": 0.1,
    "confidence_margin": 2.0,
    "data_split_seed": 42,
    "device": 0,
    "drop_prob": 0.0,
    "embedding_dim": 256,
    "evaluation_protocol": "validation_only",
    "lambda_rec_tok": 1.0,
    "margin_loss_weight": 0.0,
    "mean": 0.02,
    "model_type": "VecGAD",
    "negsamp_ratio": 1,
    "proj_R_max": 0.5,
    "proj_R_min": 0.1,
    "proj_dim": 64,
    "proj_loss_weight": 0.0,
    "readout": "avg",
    "rec_error_filter_ratio": 1.0,
    "reconstruction_loss_weight": 1.0,
    "sample_num_n": 7,
    "sample_num_p": 7,
    "sample_rate": 0.15,
    "sample_size": 10000,
    "tot_updates": 1000,
    "train_rate": 0.05,
    "var": 0.01,
    "visualize": False,
    "wandb_log_training_metrics": False,
    "warmup_epoch": 20,
}
COMMON_CONFIG.update(FIXED_CORE)

BASELINE_AXES = {
    "batch_size": 1024,
    "end_lr": 0.0001,
    "num_epoch": 200,
    "outlier_beta": 0.3,
    "peak_lr": 0.0005,
    "pp_k": 10,
    "rec_loss_weight": 1.0,
    "ring_R_min": 0.3,
    "ring_loss_weight": 1.0,
    "warmup_updates": 50,
    "weight_decay": 0.0,
}

SEARCH_SPACE = {
    "batch_size": (256, 512, 1024, 2048),
    "end_lr": (0.00001, 0.00003, 0.0001),
    "num_epoch": (100, 150, 200, 250),
    "outlier_beta": (0.1, 0.2, 0.3, 0.5),
    "peak_lr": (0.0001, 0.0002, 0.0003, 0.0005, 0.0008),
    "pp_k": (4, 6, 8, 10, 12),
    "rec_loss_weight": (0.25, 0.5, 1.0, 2.0),
    "ring_R_min": (0.1, 0.3, 0.5, 0.7),
    "ring_loss_weight": (0.25, 0.5, 1.0, 2.0),
    "warmup_updates": (5, 20, 50),
    "weight_decay": (0.0, 0.000001, 0.00001, 0.0001),
}


def canonical_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_resolved_config(config):
    observed = {key: config.get(key) for key in FIXED_CORE}
    if observed != FIXED_CORE:
        raise AssertionError(
            "fixed-core mismatch: expected {}, observed {}".format(
                FIXED_CORE, observed
            )
        )
    if config["evaluation_protocol"] not in ("validation_only", "frozen_test"):
        raise AssertionError("invalid evaluation protocol")
    if config["end_lr"] > config["peak_lr"]:
        raise AssertionError("end_lr must not exceed peak_lr")
    if not 0.0 <= config["ring_R_min"] < config["ring_R_max"]:
        raise AssertionError("ring radius ordering is invalid")
    if config["wandb_log_training_metrics"] is not False:
        raise AssertionError("training metric upload is outside authorization")


def _sample_axes(rng):
    return {key: rng.choice(values) for key, values in SEARCH_SPACE.items()}


def screening_registry():
    rng = random.Random(SEARCH_SEED)
    axes = [dict(BASELINE_AXES)]
    observed = {canonical_sha256(BASELINE_AXES)}
    while len(axes) < SCREENING_CONFIG_COUNT:
        candidate = _sample_axes(rng)
        digest = canonical_sha256(candidate)
        if digest not in observed:
            observed.add(digest)
            axes.append(candidate)

    registry = {}
    for index, axis_config in enumerate(axes):
        config_id = "cfg-{:03d}".format(index)
        resolved = dict(COMMON_CONFIG)
        resolved.update(axis_config)
        validate_resolved_config(resolved)
        registry[config_id] = resolved
    return registry


def smoke_config():
    resolved = dict(COMMON_CONFIG)
    resolved.update(BASELINE_AXES)
    resolved.update({"num_epoch": 10, "warmup_updates": 5})
    validate_resolved_config(resolved)
    return resolved


def resolve_config(phase, config_id, seed):
    if phase not in PHASE_SEEDS:
        raise ValueError("unsupported phase: {}".format(phase))
    if seed not in PHASE_SEEDS[phase]:
        raise ValueError("unsupported {} seed: {}".format(phase, seed))

    if phase == "smoke":
        if config_id != "smoke-control":
            raise ValueError("unsupported smoke config: {}".format(config_id))
        config = smoke_config()
    else:
        registry = screening_registry()
        if config_id not in registry:
            raise ValueError("unsupported config id: {}".format(config_id))
        config = dict(registry[config_id])

    if phase == "confirmation":
        config["evaluation_protocol"] = "frozen_test"
    else:
        config["evaluation_protocol"] = "validation_only"
    config["seed"] = seed
    validate_resolved_config(config)
    return config


def scientific_argv(phase, config_id, seed):
    config = resolve_config(phase, config_id, seed)
    argv = []
    for key in sorted(config):
        value = config[key]
        if isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, float):
            value = format(value, ".12g")
        argv.append("--{}={}".format(key, value))
    return argv


def screening_trial_identities():
    return [
        {
            "phase": "screening",
            "config_id": config_id,
            "seed": seed,
        }
        for config_id, seed in itertools.product(
            sorted(screening_registry()), PHASE_SEEDS["screening"]
        )
    ]


def validate_protocol():
    if sum(PHASE_BUDGET.values()) != HARD_RUN_LIMIT:
        raise AssertionError("phase allocation must equal hard budget")
    if len(screening_trial_identities()) != PHASE_BUDGET["screening"]:
        raise AssertionError("screening allocation mismatch")
    if 12 * len(PHASE_SEEDS["promotion"]) != PHASE_BUDGET["promotion"]:
        raise AssertionError("promotion allocation mismatch")
    if 6 * len(PHASE_SEEDS["confirmation"]) != PHASE_BUDGET["confirmation"]:
        raise AssertionError("confirmation allocation mismatch")
    registry = screening_registry()
    for config in registry.values():
        validate_resolved_config(config)
    for key, values in SEARCH_SPACE.items():
        observed = {config[key] for config in registry.values()}
        if observed != set(values):
            raise AssertionError("search level coverage mismatch for {}".format(key))
    return {
        "protocol_id": PROTOCOL_ID,
        "hard_run_limit": HARD_RUN_LIMIT,
        "phase_budget": PHASE_BUDGET,
        "screening_registry_sha256": canonical_sha256(registry),
        "screening_trials_sha256": canonical_sha256(screening_trial_identities()),
    }


if __name__ == "__main__":
    print(json.dumps(validate_protocol(), indent=2, sort_keys=True))
