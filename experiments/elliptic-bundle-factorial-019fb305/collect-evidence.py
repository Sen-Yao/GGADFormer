#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import wandb


ROOT = Path(__file__).resolve().parent
SCIENTIFIC_BASE_SHA = "655d6293bb76633bc6aa6fd21166a49c3b91d504"
EXECUTION_SHA = "PENDING_EXECUTION_SHA"
PROTOCOL_ID = "elliptic-bundle-factorial-019fb305"
PARENT_SWEEP_ID = "k5lbpsg9"
DATASET_SHA256 = "2f502df4b87be8f8b5ed5ef8378876125c92b06afbc5b38ee58fe4b56b1b2023"
LEVELS = ("current", "mixed")
SEEDS = tuple(range(5))
OPTIMIZATION = {
    "current": {
        "batch_size": 32768,
        "end_lr": 0.0003,
        "num_epoch": 150,
        "final_step": 150,
        "batches_per_epoch": 2,
    },
    "mixed": {
        "batch_size": 8192,
        "end_lr": 0.0001,
        "num_epoch": 200,
        "final_step": 200,
        "batches_per_epoch": 6,
    },
}
PROPAGATION = {
    "current": {"pp_k": 7, "progregate_alpha": 0.6},
    "mixed": {"pp_k": 8, "progregate_alpha": 0.8},
}
COMMON_CONFIG = {
    "dataset": "elliptic",
    "data_split_seed": 42,
    "lambda_rec_emb": 0.1,
    "ring_loss_weight": 1.0,
    "outlier_beta": 0.3,
    "peak_lr": 0.0005,
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
METRICS = ("AUC.last", "AP.last")


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
    try:
        return float(summary_value["last"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AssertionError(f"missing or invalid {label}.last summary") from exc


def diagnostic_keys(batch_count):
    return tuple(
        f"diagnostic/batch_{batch}/{suffix}"
        for batch in range(batch_count)
        for suffix in DIAGNOSTIC_SUFFIXES
    )


def terminal_diagnostics(row, batch_count):
    pseudo_total = sum(
        int(row[f"diagnostic/batch_{batch}/pseudo_count"])
        for batch in range(batch_count)
    )
    if pseudo_total <= 0:
        raise AssertionError("terminal pseudo count is not positive")
    output = {"pseudo_count": pseudo_total}
    for suffix in DIAGNOSTIC_SUFFIXES[1:]:
        output[suffix] = sum(
            float(row[f"diagnostic/batch_{batch}/{suffix}"])
            * int(row[f"diagnostic/batch_{batch}/pseudo_count"])
            for batch in range(batch_count)
        ) / pseudo_total
        if not math.isfinite(output[suffix]):
            raise AssertionError(f"non-finite terminal diagnostic {suffix}")
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


def effect_rows(by_trial):
    rows = []
    for seed in SEEDS:
        row = {"seed": seed}
        for metric in METRICS:
            cc = by_trial[("current", "current", seed)][metric]
            cm = by_trial[("current", "mixed", seed)][metric]
            mc = by_trial[("mixed", "current", seed)][metric]
            mm = by_trial[("mixed", "mixed", seed)][metric]
            row[metric] = {
                "optimization_main_mixed_minus_current": ((mc - cc) + (mm - cm)) / 2,
                "propagation_main_mixed_minus_current": ((cm - cc) + (mm - mc)) / 2,
                "optimization_at_current_propagation": mc - cc,
                "optimization_at_mixed_propagation": mm - cm,
                "propagation_at_current_optimization": cm - cc,
                "propagation_at_mixed_optimization": mm - mc,
                "interaction_difference_in_differences": mm - mc - cm + cc,
                "mixed_cell_minus_current_cell": mm - cc,
            }
        rows.append(row)
    return rows


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
sweep = api.sweep(f"HCCS/GGADFormer/{args.sweep_id}")
if sweep.state != "FINISHED":
    raise AssertionError(f"sweep state is {sweep.state}")
if len(sweep.runs) != 20:
    raise AssertionError(f"expected 20 runs, found {len(sweep.runs)}")

observed = set()
runs = []
for run in sorted(
    sweep.runs,
    key=lambda item: (
        item.config.get("optimization_bundle", ""),
        item.config.get("propagation_bundle", ""),
        int(item.config.get("seed", -1)),
    ),
):
    config = {key: value for key, value in run.config.items() if not key.startswith("_")}
    optimization = config.get("optimization_bundle")
    propagation = config.get("propagation_bundle")
    seed = int(config.get("seed", -1))
    trial = (optimization, propagation, seed)
    if optimization not in LEVELS or propagation not in LEVELS or seed not in SEEDS:
        raise AssertionError(f"unexpected trial {trial} in run {run.id}")
    if trial in observed:
        raise AssertionError(f"duplicate trial {trial}")
    observed.add(trial)
    if run.state != "finished":
        raise AssertionError(f"run {run.id} state is {run.state}")

    expected_config = {
        **COMMON_CONFIG,
        **{key: value for key, value in OPTIMIZATION[optimization].items() if key in ("batch_size", "end_lr", "num_epoch")},
        **PROPAGATION[propagation],
    }
    for key, expected in expected_config.items():
        assert_equal(config.get(key), expected, f"run {run.id} config.{key}")
    final_step = OPTIMIZATION[optimization]["final_step"]
    batch_count = OPTIMIZATION[optimization]["batches_per_epoch"]
    factorial_cell = f"opt_{optimization}__prop_{propagation}"
    for key, expected in {
        "variant": "unified_0p1_1",
        "factorial_cell": factorial_cell,
        "code_sha": EXECUTION_SHA,
        "scientific_base_sha": SCIENTIFIC_BASE_SHA,
        "protocol_identity": PROTOCOL_ID,
        "parent_sweep_id": PARENT_SWEEP_ID,
        "execution_host": "HCCS-90",
        "dataset_sha256": DATASET_SHA256,
        "optimizer_updates_per_epoch": str(batch_count),
        "final_history_step": str(final_step),
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
    expected_performance_steps = list(range(0, final_step + 1, 10))
    performance_steps = [int(row["_step"]) for row in performance_history]
    if performance_steps != expected_performance_steps:
        raise AssertionError(f"run {run.id} performance steps {performance_steps}")
    final_performance = performance_history[-1]
    auc = float(final_performance["AUC"])
    ap = float(final_performance["AP"])
    assert_equal(summary_last(run.summary.get("AUC"), "AUC"), auc, f"run {run.id} AUC.last")
    assert_equal(summary_last(run.summary.get("AP"), "AP"), ap, f"run {run.id} AP.last")

    keys = diagnostic_keys(batch_count)
    diagnostic_history = list(run.scan_history(keys=["_step", *keys], page_size=1000))
    diagnostic_steps = [int(row["_step"]) for row in diagnostic_history]
    if diagnostic_steps != list(range(final_step + 1)):
        raise AssertionError(f"run {run.id} diagnostic steps {diagnostic_steps}")
    for row in diagnostic_history:
        for key in keys:
            value = row.get(key)
            if value is None or not math.isfinite(float(value)):
                raise AssertionError(f"run {run.id} step {row['_step']} invalid {key}")
    terminal = terminal_diagnostics(diagnostic_history[-1], batch_count)
    agent = agent_by_gpu[gpu]
    runs.append({
        "optimization_bundle": optimization,
        "propagation_bundle": propagation,
        "factorial_cell": factorial_cell,
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
        "AUC.last": auc,
        "AP.last": ap,
        "final_step": final_step,
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

expected_trials = {(o, p, s) for o in LEVELS for p in LEVELS for s in SEEDS}
if observed != expected_trials:
    raise AssertionError(f"trial coverage mismatch: {sorted(observed)}")

by_trial = {
    (run["optimization_bundle"], run["propagation_bundle"], run["seed"]): run
    for run in runs
}
cell_aggregate = {}
for optimization in LEVELS:
    for propagation in LEVELS:
        cell = f"opt_{optimization}__prop_{propagation}"
        cell_aggregate[cell] = {}
        for metric in METRICS:
            values = [by_trial[(optimization, propagation, seed)][metric] for seed in SEEDS]
            cell_aggregate[cell][metric] = {
                "mean": statistics.mean(values),
                "sample_std_ddof1": statistics.stdev(values),
            }

effects_by_seed = effect_rows(by_trial)
effect_aggregate = {}
effect_names = tuple(effects_by_seed[0][METRICS[0]])
for metric in METRICS:
    effect_aggregate[metric] = {}
    for effect in effect_names:
        values = [row[metric][effect] for row in effects_by_seed]
        effect_aggregate[metric][effect] = {
            "mean": statistics.mean(values),
            "sample_std_ddof1": statistics.stdev(values),
        }

diagnostic_aggregate = {}
for optimization in LEVELS:
    for propagation in LEVELS:
        cell = f"opt_{optimization}__prop_{propagation}"
        diagnostic_aggregate[cell] = {
            suffix: {
                "mean": statistics.mean(
                    by_trial[(optimization, propagation, seed)]["terminal_diagnostics"][suffix]
                    for seed in SEEDS
                ),
                "sample_std_ddof1": statistics.stdev(
                    by_trial[(optimization, propagation, seed)]["terminal_diagnostics"][suffix]
                    for seed in SEEDS
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
    "parent_sweep_id": PARENT_SWEEP_ID,
    "scientific_base_sha": SCIENTIFIC_BASE_SHA,
    "execution_code_sha": EXECUTION_SHA,
    "expected_trials": 20,
    "agent_records": agent_records,
    "runs": runs,
}
results = {
    "sweep_id": args.sweep_id,
    "sweep_url": authoritative["sweep_url"],
    "parent_sweep_id": PARENT_SWEEP_ID,
    "scientific_base_sha": SCIENTIFIC_BASE_SHA,
    "execution_code_sha": EXECUTION_SHA,
    "endpoint_policy": {"current_optimization": 150, "mixed_optimization": 200},
    "cell_aggregate": cell_aggregate,
    "effects_by_seed": effects_by_seed,
    "effect_aggregate": effect_aggregate,
    "terminal_diagnostic_aggregate": diagnostic_aggregate,
    "decision": {
        "descriptive_factorial_complete": True,
        "post_result_search_allowed": False,
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
}, sort_keys=True))

