#!/usr/bin/env python3
import datetime as dt
import hashlib
import json
import statistics
from pathlib import Path

import wandb


TASK_ROOT = Path("/root/gpufree-data/linziyao/VecGAD-elliptic-loss-factorial-019fb305")
WORKTREE = TASK_ROOT / "worktree"
EVIDENCE_DIR = TASK_ROOT / "evidence"
LOG_DIR = TASK_ROOT / "logs"
SWEEP_PATH = "HCCS/GGADFormer/rmhd15po"
SWEEP_URL = "https://wandb.ai/HCCS/GGADFormer/sweeps/rmhd15po"
CODE_SHA = "655d6293bb76633bc6aa6fd21166a49c3b91d504"
PROTOCOL_ID = "elliptic-loss-factorial-019fb305"
EXPECTED_STEPS = list(range(0, 151, 10))
PANE_BY_GPU = {
    "0": "%44",
    "1": "%45",
    "2": "%46",
    "4": "%47",
    "5": "%48",
    "6": "%49",
    "7": "%50",
}
PRIOR_RESULTS_PATH = WORKTREE / "experiments/elliptic-loss-unification-019fb305/results.json"
PRIOR_RESULTS_SHA256 = "8e27e0dc0f723c4ff7be729a5f73caa852021b39885b0da945342adf11d621b2"
PRIOR_AUTH_PATH = WORKTREE / "experiments/elliptic-loss-unification-019fb305/authoritative-sweep.json"
PRIOR_AUTH_SHA256 = "efe5263a3666ad4287a3047ae722eadf1d401553a3644d081c60e51c05c6fb54"

COMMON_CONFIG = {
    "batch_size": 32768,
    "dataset": "elliptic",
    "end_lr": 0.0003,
    "num_epoch": 150,
    "outlier_beta": 0.3,
    "peak_lr": 0.0005,
    "pp_k": 7,
    "progregate_alpha": 0.6,
    "rec_loss_weight": 1,
    "ring_R_max": 1,
    "ring_R_min": 0.3,
    "train_rate": 0.05,
    "warmup_updates": 50,
    "data_split_seed": 42,
    "split_protocol_identity": "elliptic:train_rate=0.05:val_rate=0.1:data_split_seed=42",
    "code_sha": CODE_SHA,
    "execution_host": "HCCS-90",
    "protocol_identity": PROTOCOL_ID,
    "fixed_final_epoch_metric_policy": "AUC.last/AP.last at fixed training endpoint",
}
CELL_CONFIG = {
    "emb_only_0p1_20": {
        "lambda_rec_emb": 0.1,
        "ring_loss_weight": 20,
        "variant": "lambda_rec_emb=0.1;ring_loss_weight=20",
    },
    "ring_only_2_1": {
        "lambda_rec_emb": 2,
        "ring_loss_weight": 1,
        "variant": "lambda_rec_emb=2;ring_loss_weight=1",
    },
}
EXPECTED_TRIALS = {(cell, seed) for cell in CELL_CONFIG for seed in range(5)}
ALL_CELLS = ("control_2_20", "emb_only_0p1_20", "ring_only_2_1", "unified_0p1_1")
METRICS = ("AUC.last", "AP.last")


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


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
        if abs(float(actual) - expected) > 1e-12:
            raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")
    elif actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def effect_record(seed, cells, coefficients):
    row = {"seed": seed}
    for metric in METRICS:
        row[metric] = sum(coefficients[cell] * cells[cell][seed][metric] for cell in coefficients)
    return row


EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

agent_records = []
checksum_lines = [f"# HCCS-90 task root: {LOG_DIR}"]
for path in sorted(LOG_DIR.glob("agent-gpu*")):
    checksum_lines.append(f"{sha256_file(path)}  {path}")

for gpu, pane in PANE_BY_GPU.items():
    logs = sorted(LOG_DIR.glob(f"agent-gpu{gpu}-*.log"))
    if len(logs) != 1:
        raise AssertionError(f"GPU {gpu}: expected one agent log, found {len(logs)}")
    log_path = logs[0]
    stem = log_path.with_suffix("")
    start_path = Path(str(stem) + ".start-utc")
    finish_path = Path(str(stem) + ".finish-utc")
    exit_path = Path(str(stem) + ".exitcode")
    for evidence_path in (start_path, finish_path, exit_path):
        if not evidence_path.is_file():
            raise AssertionError(f"missing agent evidence: {evidence_path}")
    exit_code = int(exit_path.read_text().strip())
    if exit_code != 0:
        raise AssertionError(f"GPU {gpu}: agent exit code {exit_code}")
    agent_records.append({
        "gpu": int(gpu),
        "pane": pane,
        "start_utc": start_path.read_text().strip(),
        "finish_utc": finish_path.read_text().strip(),
        "exit_code": exit_code,
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
    })

log_hashes_path = EVIDENCE_DIR / "remote-log-sha256.txt"
log_hashes_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

api = wandb.Api(timeout=30)
sweep = api.sweep(SWEEP_PATH)
if sweep.state != "FINISHED":
    raise AssertionError(f"sweep state is {sweep.state}, expected FINISHED")
if len(sweep.runs) != 10:
    raise AssertionError(f"expected 10 runs, found {len(sweep.runs)}")

run_records = []
observed_trials = set()
normalized_configs = []
for run in sorted(sweep.runs, key=lambda item: (item.config.get("factorial_cell", ""), int(item.config.get("seed", -1)))):
    config = {key: value for key, value in run.config.items() if not key.startswith("_")}
    cell = config.get("factorial_cell")
    seed = int(config.get("seed", -1))
    trial = (cell, seed)
    if trial not in EXPECTED_TRIALS:
        raise AssertionError(f"unexpected trial {trial} in run {run.id}")
    if trial in observed_trials:
        raise AssertionError(f"duplicate trial {trial}")
    observed_trials.add(trial)
    if run.state != "finished":
        raise AssertionError(f"run {run.id} state is {run.state}")

    expected_config = {**COMMON_CONFIG, **CELL_CONFIG[cell], "factorial_cell": cell, "seed": seed}
    for key, expected in expected_config.items():
        assert_equal(config.get(key), expected, f"run {run.id} config.{key}")
    gpu = str(config.get("gpu_index"))
    if gpu not in PANE_BY_GPU:
        raise AssertionError(f"run {run.id}: unexpected GPU {gpu}")

    normalized = dict(config)
    for key in ("factorial_cell", "variant", "seed", "gpu_index", "lambda_rec_emb", "ring_loss_weight"):
        normalized.pop(key, None)
    normalized_configs.append((run.id, normalized))

    history = list(run.scan_history(keys=["_step", "AUC", "AP"], page_size=500))
    steps = [int(row["_step"]) for row in history]
    if steps != EXPECTED_STEPS:
        raise AssertionError(f"run {run.id}: history steps {steps}")
    final_rows = [row for row in history if int(row["_step"]) == 150]
    if len(final_rows) != 1:
        raise AssertionError(f"run {run.id}: expected one final history row, found {len(final_rows)}")
    final = final_rows[0]
    auc = float(final["AUC"])
    ap = float(final["AP"])
    assert_equal(run.summary.get("AUC").get("last"), auc, f"run {run.id} summary AUC.last")
    assert_equal(run.summary.get("AP").get("last"), ap, f"run {run.id} summary AP.last")

    agent = next(item for item in agent_records if str(item["gpu"]) == gpu)
    run_records.append({
        "factorial_cell": cell,
        "seed": seed,
        "run_id": run.id,
        "name": run.name,
        "url": run.url,
        "state": run.state,
        "created_at": run.created_at,
        "runtime_seconds": float(run.summary.get("_runtime")),
        "code_sha": config["code_sha"],
        "execution_host": config["execution_host"],
        "gpu_index": gpu,
        "pane": agent["pane"],
        "protocol_identity": config["protocol_identity"],
        "split_protocol_identity": config["split_protocol_identity"],
        "fixed_final_epoch_metric_policy": config["fixed_final_epoch_metric_policy"],
        "lambda_rec_emb": float(config["lambda_rec_emb"]),
        "ring_loss_weight": float(config["ring_loss_weight"]),
        "AUC.last": auc,
        "AP.last": ap,
        "final_step": 150,
        "config": config,
        "config_sha256": sha256_bytes(canonical_bytes(config)),
        "history": history,
        "history_sha256": sha256_bytes(canonical_bytes(history)),
        "agent_log_path": agent["log_path"],
        "agent_log_sha256": agent["log_sha256"],
        "agent_exit_code": agent["exit_code"],
    })

if observed_trials != EXPECTED_TRIALS:
    raise AssertionError(f"trial coverage mismatch: {sorted(observed_trials)}")
base_id, base_config = normalized_configs[0]
for run_id, normalized in normalized_configs[1:]:
    if normalized != base_config:
        changed = sorted(key for key in set(base_config) | set(normalized) if base_config.get(key) != normalized.get(key))
        raise AssertionError(f"unexpected effective config drift in run {run_id} vs {base_id}: {changed}")

if sha256_file(PRIOR_RESULTS_PATH) != PRIOR_RESULTS_SHA256:
    raise AssertionError("prior results hash mismatch")
if sha256_file(PRIOR_AUTH_PATH) != PRIOR_AUTH_SHA256:
    raise AssertionError("prior authoritative sweep hash mismatch")
prior_results = json.loads(PRIOR_RESULTS_PATH.read_text(encoding="utf-8"))
prior_auth = json.loads(PRIOR_AUTH_PATH.read_text(encoding="utf-8"))
if prior_results.get("code_sha") != CODE_SHA or prior_results.get("sweep_id") != "l6ubfjxt":
    raise AssertionError("prior result identity mismatch")
if prior_auth.get("state") != "FINISHED" or len(prior_auth.get("runs", [])) != 10:
    raise AssertionError("prior authoritative sweep is incomplete")

prior_history_by_id = {record["run_id"]: record for record in prior_auth["runs"]}
cells = {cell: {} for cell in ALL_CELLS}
for record in prior_results["runs"]:
    cell = record["variant"]
    if cell not in ("control_2_20", "unified_0p1_1"):
        raise AssertionError(f"unexpected prior cell {cell}")
    history_record = prior_history_by_id.get(record["run_id"])
    if history_record is None or history_record["history_sha256"] != record["history_sha256"]:
        raise AssertionError(f"prior history identity mismatch for {record['run_id']}")
    final_rows = [row for row in history_record["history"] if int(row["_step"]) == 150]
    if len(final_rows) != 1:
        raise AssertionError(f"prior run {record['run_id']} lacks one step-150 row")
    for metric, history_key in (("AUC.last", "AUC"), ("AP.last", "AP")):
        assert_equal(record[metric], final_rows[0][history_key], f"prior run {record['run_id']} {metric}")
    cells[cell][int(record["seed"])] = record

for record in run_records:
    cells[record["factorial_cell"]][record["seed"]] = record
for cell in ALL_CELLS:
    if sorted(cells[cell]) != list(range(5)):
        raise AssertionError(f"cell {cell} seed coverage mismatch")

aggregate = {}
for cell in ALL_CELLS:
    aggregate[cell] = {}
    for metric in METRICS:
        values = [cells[cell][seed][metric] for seed in range(5)]
        aggregate[cell][metric] = {
            "mean": statistics.mean(values),
            "sample_std_ddof1": statistics.stdev(values),
        }

effect_definitions = {
    "lambda_rec_emb_low_minus_high_at_ring_20": {"emb_only_0p1_20": 1, "control_2_20": -1},
    "lambda_rec_emb_low_minus_high_at_ring_1": {"unified_0p1_1": 1, "ring_only_2_1": -1},
    "ring_loss_weight_low_minus_high_at_emb_2": {"ring_only_2_1": 1, "control_2_20": -1},
    "ring_loss_weight_low_minus_high_at_emb_0p1": {"unified_0p1_1": 1, "emb_only_0p1_20": -1},
    "interaction_difference_of_differences": {
        "unified_0p1_1": 1,
        "emb_only_0p1_20": -1,
        "ring_only_2_1": -1,
        "control_2_20": 1,
    },
}
effects = {}
for name, coefficients in effect_definitions.items():
    by_seed = [effect_record(seed, cells, coefficients) for seed in range(5)]
    effects[name] = {
        "formula": " + ".join(f"{coefficient:+d}*{cell}" for cell, coefficient in coefficients.items()),
        "paired_differences_by_seed": by_seed,
        "paired_mean_difference": {
            metric: statistics.mean(row[metric] for row in by_seed) for metric in METRICS
        },
    }

for factor, names in {
    "lambda_rec_emb_low_minus_high_marginal": (
        "lambda_rec_emb_low_minus_high_at_ring_20",
        "lambda_rec_emb_low_minus_high_at_ring_1",
    ),
    "ring_loss_weight_low_minus_high_marginal": (
        "ring_loss_weight_low_minus_high_at_emb_2",
        "ring_loss_weight_low_minus_high_at_emb_0p1",
    ),
}.items():
    by_seed = []
    for seed in range(5):
        by_seed.append({
            "seed": seed,
            **{
                metric: statistics.mean(
                    effects[name]["paired_differences_by_seed"][seed][metric] for name in names
                )
                for metric in METRICS
            },
        })
    effects[factor] = {
        "formula": f"0.5*({names[0]} + {names[1]})",
        "paired_differences_by_seed": by_seed,
        "paired_mean_difference": {
            metric: statistics.mean(row[metric] for row in by_seed) for metric in METRICS
        },
    }

authoritative = {
    "queried_at_utc": utc_now(),
    "source": f'wandb.Api().sweep("{SWEEP_PATH}") plus run.scan_history(keys=["_step","AUC","AP"])',
    "entity": "HCCS",
    "project": "GGADFormer",
    "sweep_id": "rmhd15po",
    "sweep_url": SWEEP_URL,
    "state": sweep.state,
    "program": sweep.config.get("program"),
    "method": sweep.config.get("method"),
    "metric": sweep.config.get("metric"),
    "sweep_parameters": sweep.config.get("parameters"),
    "code_sha": CODE_SHA,
    "expected_trials": 10,
    "identity_validation": {
        "all_expected_trials_present_once": True,
        "all_runs_finished": True,
        "all_agent_exit_codes_zero": True,
        "all_final_history_steps_present": True,
        "summary_matches_final_history": True,
        "only_declared_config_axes_differ": True,
    },
    "agent_records": agent_records,
    "runs": run_records,
}
authoritative_path = EVIDENCE_DIR / "authoritative-sweep.json"
authoritative_path.write_text(json.dumps(authoritative, indent=2, sort_keys=False) + "\n", encoding="utf-8")

results = {
    "collected_at_utc": authoritative["queried_at_utc"],
    "entity": "HCCS",
    "project": "GGADFormer",
    "new_sweep_id": "rmhd15po",
    "new_sweep_url": SWEEP_URL,
    "new_sweep_final_state": sweep.state,
    "new_sweep_stopped_after_expected_trials": True,
    "prior_sweep_id": "l6ubfjxt",
    "prior_sweep_url": "https://wandb.ai/HCCS/GGADFormer/sweeps/l6ubfjxt",
    "code_sha": CODE_SHA,
    "protocol_id": PROTOCOL_ID,
    "evidence_hashes": {
        "authoritative-sweep.json": sha256_file(authoritative_path),
        "remote-log-sha256.txt": sha256_file(log_hashes_path),
        "prior-results.json": PRIOR_RESULTS_SHA256,
        "prior-authoritative-sweep.json": PRIOR_AUTH_SHA256,
    },
    "new_runs": [{key: record[key] for key in (
        "factorial_cell", "seed", "run_id", "name", "url", "state", "created_at", "runtime_seconds",
        "code_sha", "execution_host", "gpu_index", "pane", "lambda_rec_emb", "ring_loss_weight",
        "AUC.last", "AP.last", "final_step", "config_sha256", "history_sha256", "agent_log_sha256",
        "agent_exit_code",
    )} for record in run_records],
    "factorial_cells": aggregate,
    "paired_effects": effects,
    "protocol_deviations": [],
    "operational_notes": [
        "Seven agent panes required two windows within one tmux session; this did not change the scientific protocol.",
        "Panes 44-48 were no longer retained after their window completed; immutable start-command evidence and zero-exit agent files were preserved.",
    ],
}
results_path = EVIDENCE_DIR / "results.json"
results_path.write_text(json.dumps(results, indent=2, sort_keys=False) + "\n", encoding="utf-8")

print(json.dumps({
    "authoritative_path": str(authoritative_path),
    "authoritative_sha256": sha256_file(authoritative_path),
    "results_path": str(results_path),
    "results_sha256": sha256_file(results_path),
    "log_hashes_path": str(log_hashes_path),
    "log_hashes_sha256": sha256_file(log_hashes_path),
    "factorial_cells": aggregate,
    "paired_effects": {name: value["paired_mean_difference"] for name, value in effects.items()},
}, indent=2, sort_keys=True))
