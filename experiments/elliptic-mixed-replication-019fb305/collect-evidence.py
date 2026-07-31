#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import wandb


ROOT = Path(__file__).resolve().parent
SCIENTIFIC_BASE_SHA = "655d6293bb76633bc6aa6fd21166a49c3b91d504"
EXECUTION_SHA = "40986c9f8b460f8fd9baaefb985573209f96e572"
PROTOCOL_ID = "elliptic-mixed-replication-019fb305"
DATASET_SHA256 = "2f502df4b87be8f8b5ed5ef8378876125c92b06afbc5b38ee58fe4b56b1b2023"
EXPECTED_PERFORMANCE_STEPS = list(range(0, 201, 10))
EXPECTED_DIAGNOSTIC_STEPS = list(range(201))
VARIANT_CONFIG = {
    "control_2_20": {"lambda_rec_emb": 2.0, "ring_loss_weight": 20.0},
    "unified_0p1_1": {"lambda_rec_emb": 0.1, "ring_loss_weight": 1.0},
}
COMMON_CONFIG = {
    "batch_size": 8192,
    "dataset": "elliptic",
    "data_split_seed": 42,
    "end_lr": 0.0001,
    "num_epoch": 200,
    "outlier_beta": 0.3,
    "peak_lr": 0.0005,
    "pp_k": 8,
    "progregate_alpha": 0.8,
    "rec_loss_weight": 1.0,
    "ring_R_max": 1.0,
    "ring_R_min": 0.3,
    "train_rate": 0.05,
    "warmup_updates": 50,
    "hsc_diagnostics": True,
    "ablation_mode": "none",
}
DIAGNOSTIC_SUFFIXES = (
    "pseudo_count",
    "hsc/shell_hit_rate",
    "hsc/inner_violation_rate",
    "hsc/outer_violation_rate",
    "hsc/distance_mean",
    "loss/bce_raw",
    "loss/rec_combined_raw",
    "loss/hsc_raw",
    "loss/bce_weighted",
    "loss/rec_combined_weighted",
    "loss/hsc_weighted",
    "loss/true_weighted_total",
)
DIAGNOSTIC_KEYS = tuple(
    f"diagnostic/batch_{batch}/{suffix}"
    for batch in range(6)
    for suffix in DIAGNOSTIC_SUFFIXES
)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_equal(actual, expected, label):
    if isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")
    elif actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def summary_last(summary_value, label):
    if not isinstance(summary_value, Mapping) or "last" not in summary_value:
        raise AssertionError(f"missing {label}.last summary")
    return float(summary_value["last"])


def terminal_diagnostics(row):
    pseudo_total = sum(int(row[f"diagnostic/batch_{batch}/pseudo_count"]) for batch in range(6))
    if pseudo_total <= 0:
        raise AssertionError("terminal pseudo count is not positive")
    output = {"pseudo_count": pseudo_total}
    for suffix in DIAGNOSTIC_SUFFIXES[1:]:
        weighted = sum(
            float(row[f"diagnostic/batch_{batch}/{suffix}"])
            * int(row[f"diagnostic/batch_{batch}/pseudo_count"])
            for batch in range(6)
        ) / pseudo_total
        if not math.isfinite(weighted):
            raise AssertionError(f"non-finite terminal diagnostic {suffix}")
        output[suffix] = weighted
    rate_sum = sum(
        output[name]
        for name in (
            "hsc/shell_hit_rate",
            "hsc/inner_violation_rate",
            "hsc/outer_violation_rate",
        )
    )
    if not math.isclose(rate_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise AssertionError(f"terminal HSC rates sum to {rate_sum}")
    return output


parser = argparse.ArgumentParser()
parser.add_argument("--sweep-id", required=True)
parser.add_argument("--agent-records", type=Path, required=True)
args = parser.parse_args()

agent_records = json.loads(args.agent_records.read_text(encoding="utf-8"))
if len(agent_records) != 8:
    raise AssertionError(f"expected 8 agent records, found {len(agent_records)}")
agent_by_gpu = {}
for agent in agent_records:
    gpu = str(agent["gpu"])
    if gpu in agent_by_gpu:
        raise AssertionError(f"duplicate agent GPU {gpu}")
    if int(agent["exit_code"]) != 0:
        raise AssertionError(f"agent GPU {gpu} exit code is {agent['exit_code']}")
    if len(str(agent["log_sha256"])) != 64:
        raise AssertionError(f"agent GPU {gpu} has invalid log hash")
    agent_by_gpu[gpu] = agent

api = wandb.Api(timeout=60)
sweep_path = f"HCCS/GGADFormer/{args.sweep_id}"
sweep = api.sweep(sweep_path)
if sweep.state != "FINISHED":
    raise AssertionError(f"sweep state is {sweep.state}")
if len(sweep.runs) != 10:
    raise AssertionError(f"expected 10 runs, found {len(sweep.runs)}")

observed = set()
runs = []
for run in sorted(
    sweep.runs,
    key=lambda item: (item.config.get("variant", ""), int(item.config.get("seed", -1))),
):
    config = {key: value for key, value in run.config.items() if not key.startswith("_")}
    variant = config.get("variant")
    seed = int(config.get("seed", -1))
    trial = (variant, seed)
    if variant not in VARIANT_CONFIG or seed not in range(5):
        raise AssertionError(f"unexpected trial {trial} in run {run.id}")
    if trial in observed:
        raise AssertionError(f"duplicate trial {trial}")
    observed.add(trial)
    if run.state != "finished":
        raise AssertionError(f"run {run.id} state is {run.state}")

    for key, expected in {**COMMON_CONFIG, **VARIANT_CONFIG[variant]}.items():
        assert_equal(config.get(key), expected, f"run {run.id} config.{key}")
    for key, expected in {
        "code_sha": EXECUTION_SHA,
        "scientific_base_sha": SCIENTIFIC_BASE_SHA,
        "protocol_identity": PROTOCOL_ID,
        "execution_host": "HCCS-90",
        "dataset_sha256": DATASET_SHA256,
        "optimizer_updates_per_epoch": "6",
        "final_history_step": "200",
    }.items():
        assert_equal(config.get(key), expected, f"run {run.id} config.{key}")

    gpu = str(config.get("gpu_index"))
    if gpu not in agent_by_gpu:
        raise AssertionError(f"run {run.id} has unexpected GPU {gpu}")

    performance_history = [
        row
        for row in run.scan_history(keys=["_step", "AUC", "AP"], page_size=1000)
        if row.get("AUC") is not None and row.get("AP") is not None
    ]
    performance_steps = [int(row["_step"]) for row in performance_history]
    if performance_steps != EXPECTED_PERFORMANCE_STEPS:
        raise AssertionError(f"run {run.id} performance steps {performance_steps}")
    final_performance = performance_history[-1]
    auc = float(final_performance["AUC"])
    ap = float(final_performance["AP"])
    assert_equal(summary_last(run.summary.get("AUC"), "AUC"), auc, f"run {run.id} AUC.last")
    assert_equal(summary_last(run.summary.get("AP"), "AP"), ap, f"run {run.id} AP.last")

    diagnostic_history = list(
        run.scan_history(keys=["_step", *DIAGNOSTIC_KEYS], page_size=1000)
    )
    diagnostic_steps = [int(row["_step"]) for row in diagnostic_history]
    if diagnostic_steps != EXPECTED_DIAGNOSTIC_STEPS:
        raise AssertionError(f"run {run.id} diagnostic steps {diagnostic_steps}")
    for row in diagnostic_history:
        for key in DIAGNOSTIC_KEYS:
            value = row.get(key)
            if value is None or not math.isfinite(float(value)):
                raise AssertionError(f"run {run.id} step {row['_step']} invalid {key}")
    terminal = terminal_diagnostics(diagnostic_history[-1])
    agent = agent_by_gpu[gpu]
    runs.append({
        "variant": variant,
        "seed": seed,
        "run_id": run.id,
        "name": run.name,
        "url": run.url,
        "state": run.state,
        "created_at": run.created_at,
        "runtime_seconds": float(run.summary.get("_runtime")),
        "gpu_index": gpu,
        "pane": agent["pane"],
        "code_sha": config["code_sha"],
        "scientific_base_sha": config["scientific_base_sha"],
        "lambda_rec_emb": float(config["lambda_rec_emb"]),
        "ring_loss_weight": float(config["ring_loss_weight"]),
        "AUC.last": auc,
        "AP.last": ap,
        "final_step": 200,
        "terminal_diagnostics": terminal,
        "config": config,
        "config_sha256": sha256_bytes(canonical_bytes(config)),
        "performance_history": performance_history,
        "performance_history_sha256": sha256_bytes(canonical_bytes(performance_history)),
        "diagnostic_history": diagnostic_history,
        "diagnostic_history_sha256": sha256_bytes(canonical_bytes(diagnostic_history)),
        "agent_log_path": agent["log_path"],
        "agent_log_sha256": agent["log_sha256"],
        "agent_exit_code": int(agent["exit_code"]),
    })

expected_trials = {(variant, seed) for variant in VARIANT_CONFIG for seed in range(5)}
if observed != expected_trials:
    raise AssertionError(f"trial coverage mismatch: {sorted(observed)}")

by_trial = {(run["variant"], run["seed"]): run for run in runs}
aggregate = {}
for variant in VARIANT_CONFIG:
    aggregate[variant] = {}
    for metric in ("AUC.last", "AP.last"):
        values = [by_trial[(variant, seed)][metric] for seed in range(5)]
        aggregate[variant][metric] = {
            "mean": statistics.mean(values),
            "sample_std_ddof1": statistics.stdev(values),
        }

paired = []
for seed in range(5):
    paired.append({
        "seed": seed,
        "AUC.last": by_trial[("unified_0p1_1", seed)]["AUC.last"]
        - by_trial[("control_2_20", seed)]["AUC.last"],
        "AP.last": by_trial[("unified_0p1_1", seed)]["AP.last"]
        - by_trial[("control_2_20", seed)]["AP.last"],
    })
paired_mean = {
    metric: statistics.mean(row[metric] for row in paired)
    for metric in ("AUC.last", "AP.last")
}
auc_pass = paired_mean["AUC.last"] >= -0.01
ap_pass = paired_mean["AP.last"] >= -0.02
candidate = aggregate["unified_0p1_1"]

diagnostic_aggregate = {}
for variant in VARIANT_CONFIG:
    diagnostic_aggregate[variant] = {
        suffix: {
            "mean": statistics.mean(
                by_trial[(variant, seed)]["terminal_diagnostics"][suffix]
                for seed in range(5)
            ),
            "sample_std_ddof1": statistics.stdev(
                by_trial[(variant, seed)]["terminal_diagnostics"][suffix]
                for seed in range(5)
            ),
        }
        for suffix in DIAGNOSTIC_SUFFIXES[1:]
    }

authoritative = {
    "queried_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "source": "wandb.Api sweep plus explicit performance and diagnostic scan_history",
    "entity": "HCCS",
    "project": "GGADFormer",
    "sweep_id": args.sweep_id,
    "sweep_url": f"https://wandb.ai/HCCS/GGADFormer/sweeps/{args.sweep_id}",
    "state": sweep.state,
    "scientific_base_sha": SCIENTIFIC_BASE_SHA,
    "execution_code_sha": EXECUTION_SHA,
    "expected_trials": 10,
    "agent_records": agent_records,
    "runs": runs,
}
results = {
    "sweep_id": args.sweep_id,
    "sweep_url": authoritative["sweep_url"],
    "scientific_base_sha": SCIENTIFIC_BASE_SHA,
    "execution_code_sha": EXECUTION_SHA,
    "fixed_final_step": 200,
    "aggregate": aggregate,
    "paired_differences_unified_minus_control": paired,
    "paired_mean_difference_unified_minus_control": paired_mean,
    "terminal_diagnostic_aggregate": diagnostic_aggregate,
    "decision": {
        "AUROC_equivalence_pass": auc_pass,
        "AUPRC_equivalence_pass": ap_pass,
        "joint_practical_equivalence_pass": auc_pass and ap_pass,
        "candidate_above_GGAD_AUROC": candidate["AUC.last"]["mean"] > 0.7006,
        "candidate_above_GGAD_AUPRC": candidate["AP.last"]["mean"] > 0.2565,
        "optimization_x_propagation_followup_allowed": auc_pass and ap_pass,
    },
    "runs": [
        {key: value for key, value in run.items() if key not in ("config", "performance_history", "diagnostic_history")}
        for run in runs
    ],
    "protocol_deviations": [],
}

authoritative_path = ROOT / "authoritative-sweep.json"
results_path = ROOT / "results.json"
authoritative_path.write_text(json.dumps(authoritative, indent=2) + "\n", encoding="utf-8")
results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "status": "passed",
    "sweep_id": args.sweep_id,
    "runs": len(runs),
    "authoritative_sha256": sha256_file(authoritative_path),
    "results_sha256": sha256_file(results_path),
    "joint_practical_equivalence_pass": auc_pass and ap_pass,
}, sort_keys=True))
