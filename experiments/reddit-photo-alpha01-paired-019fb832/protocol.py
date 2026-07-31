#!/usr/bin/env python
import hashlib
import json


PROTOCOL_ID = "reddit-photo-alpha01-paired-019fb832"
DATASETS = ("reddit", "photo")
VARIANTS = ("historical_control", "alpha_0p1")
SEEDS = tuple(range(5))
FINAL_STEP = 200

COMMON_CONFIG = {
    "GT_attention_dropout": 0.4,
    "GT_dropout": 0.4,
    "GT_ffn_dim": 256,
    "GT_num_heads": 2,
    "GT_num_layers": 3,
    "ablation_mode": "none",
    "auc_test_rounds": 256,
    "batch_size": None,
    "bce_loss_weight": 1.0,
    "con_loss_temp": 10.0,
    "con_loss_weight": 0.1,
    "confidence_margin": 2.0,
    "data_split_seed": 42,
    "device": 0,
    "drop_prob": 0.0,
    "embedding_dim": 256,
    "end_lr": 0.0001,
    "GNA_temp": 1.0,
    "lambda_rec_emb": 0.1,
    "lambda_rec_tok": 1.0,
    "margin_loss_weight": 0.0,
    "mean": 0.02,
    "model_type": "VecGAD",
    "negsamp_ratio": 1,
    "num_epoch": 200,
    "outlier_beta": 0.3,
    "peak_lr": 0.0005,
    "pp_k": None,
    "progregate_alpha": None,
    "proj_loss_weight": 0.0,
    "proj_dim": 64,
    "proj_R_max": 0.5,
    "proj_R_min": 0.1,
    "readout": "avg",
    "rec_error_filter_ratio": 1.0,
    "rec_loss_weight": 1.0,
    "reconstruction_loss_weight": 1.0,
    "ring_R_max": 1.0,
    "ring_R_min": 0.3,
    "ring_loss_weight": 1.0,
    "sample_num_n": 7,
    "sample_num_p": 7,
    "sample_rate": 0.15,
    "sample_size": 10000,
    "tot_updates": 1000,
    "train_rate": 0.05,
    "var": 0.01,
    "visualize": False,
    "warmup_epoch": 20,
    "warmup_updates": 50,
    "weight_decay": 0.0,
}

DATASET_CONFIG = {
    "reddit": {"batch_size": 1024, "pp_k": 10},
    "photo": {"batch_size": 128, "pp_k": 6},
}

ALPHA_CONFIG = {
    "reddit": {"historical_control": 0.0, "alpha_0p1": 0.1},
    "photo": {"historical_control": 0.05, "alpha_0p1": 0.1},
}


def resolve_config(dataset, variant, seed):
    if dataset not in DATASETS:
        raise ValueError("unsupported dataset: {}".format(dataset))
    if variant not in VARIANTS:
        raise ValueError("unsupported variant: {}".format(variant))
    if seed not in SEEDS:
        raise ValueError("unsupported seed: {}".format(seed))

    config = dict(COMMON_CONFIG)
    config.update(DATASET_CONFIG[dataset])
    config.update({
        "dataset": dataset,
        "declared_variant": variant,
        "progregate_alpha": ALPHA_CONFIG[dataset][variant],
        "seed": seed,
    })
    if any(value is None for value in config.values()):
        raise AssertionError("resolved config contains an unset value")
    return config


def scientific_argv(dataset, variant, seed):
    config = resolve_config(dataset, variant, seed)
    argv = []
    for key in sorted(config):
        if key == "declared_variant":
            continue
        value = config[key]
        if isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, float):
            value = format(value, ".12g")
        argv.append("--{}={}".format(key, value))
    return argv


def expected_trials():
    return [
        resolve_config(dataset, variant, seed)
        for dataset in DATASETS
        for variant in VARIANTS
        for seed in SEEDS
    ]


def canonical_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def protocol_digest():
    return canonical_sha256(expected_trials())
