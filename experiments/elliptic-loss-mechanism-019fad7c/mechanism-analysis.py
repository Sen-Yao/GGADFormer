#!/usr/bin/env python3
"""Deterministically summarize the frozen four-cell Elliptic mechanism probe."""

import argparse
import hashlib
import json
from pathlib import Path


CELLS = (
    "control_2_20",
    "emb_only_0p1_20",
    "ring_only_2_1",
    "unified_0p1_1",
)
GRADIENT_EPOCHS = (0, 1, 2, 5, 10, 20, 50, 100, 150)
RAW_INITIAL_KEYS = (
    "loss/bce_raw",
    "loss/token_rec_raw",
    "loss/emb_rec_raw",
    "loss/hsc_raw",
    "gradient/bce/raw_norm",
    "gradient/token_rec/raw_norm",
    "gradient/emb_rec/raw_norm",
    "gradient/hsc/raw_norm",
    "gradient_cosine/bce__token_rec",
    "gradient_cosine/bce__emb_rec",
    "gradient_cosine/bce__hsc",
    "gradient_cosine/token_rec__emb_rec",
    "gradient_cosine/token_rec__hsc",
    "gradient_cosine/emb_rec__hsc",
)
GRADIENT_KEYS = (
    "gradient/bce/weighted_norm",
    "gradient/token_rec/weighted_norm",
    "gradient/emb_rec/weighted_norm",
    "gradient/hsc/weighted_norm",
    "gradient/weighted_total_norm",
    "gradient_cosine/bce__token_rec",
    "gradient_cosine/bce__emb_rec",
    "gradient_cosine/bce__hsc",
    "gradient_cosine/token_rec__emb_rec",
    "gradient_cosine/token_rec__hsc",
    "gradient_cosine/emb_rec__hsc",
)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weighted_mean(rows, value_key, count_key):
    total = sum(float(row[count_key]) for row in rows)
    return sum(float(row[value_key]) * float(row[count_key]) for row in rows) / total


def pooled_variance(rows, prefix):
    count_key = prefix + "/count"
    mean_key = prefix + "/mean"
    variance_key = prefix + "/variance"
    total = sum(int(row[count_key]) for row in rows)
    mean = weighted_mean(rows, mean_key, count_key)
    second_moment = sum(
        int(row[count_key])
        * (float(row[variance_key]) + float(row[mean_key]) ** 2)
        for row in rows
    ) / total
    return second_moment - mean ** 2


def summarize_terminal(rows):
    all_count = "score/train_all/count"
    normal_count = "score/train_normal/count"
    pseudo_count = "score/train_pseudo/count"
    normal_mean = weighted_mean(rows, "score/train_normal/mean", normal_count)
    pseudo_mean = weighted_mean(rows, "score/train_pseudo/mean", pseudo_count)
    return {
        "loss": {
            "bce_raw": weighted_mean(rows, "loss/bce_raw", all_count),
            "token_rec_raw": weighted_mean(rows, "loss/token_rec_raw", pseudo_count),
            "embedding_rec_raw": weighted_mean(rows, "loss/emb_rec_raw", pseudo_count),
            "hsc_raw": weighted_mean(rows, "loss/hsc_raw", pseudo_count),
            "true_weighted_total": weighted_mean(
                rows, "loss/true_weighted_total", all_count
            ),
        },
        "hsc": {
            "shell_hit_rate": weighted_mean(rows, "hsc/shell_hit_rate", pseudo_count),
            "inner_violation_rate": weighted_mean(
                rows, "hsc/inner_violation_rate", pseudo_count
            ),
            "outer_violation_rate": weighted_mean(
                rows, "hsc/outer_violation_rate", pseudo_count
            ),
        },
        "score": {
            "normal_mean": normal_mean,
            "pseudo_mean": pseudo_mean,
            "pseudo_minus_normal_mean_gap": pseudo_mean - normal_mean,
            "all_variance": pooled_variance(rows, "score/train_all"),
            "normal_variance": pooled_variance(rows, "score/train_normal"),
            "pseudo_variance": pooled_variance(rows, "score/train_pseudo"),
        },
        "geometry": {
            "normal_to_center_mean": weighted_mean(
                rows, "distance/normal_to_center/mean", normal_count
            ),
            "pseudo_to_center_mean": weighted_mean(
                rows, "distance/pseudo_to_center/mean", pseudo_count
            ),
            "normal_embedding_norm_mean": weighted_mean(
                rows, "norm/normal_embedding/mean", normal_count
            ),
            "pseudo_embedding_norm_mean": weighted_mean(
                rows, "norm/pseudo_embedding/mean", pseudo_count
            ),
            "reconstruction_displacement_norm_mean": weighted_mean(
                rows, "norm/reconstruction_displacement/mean", pseudo_count
            ),
            "normal_centered_rms": weighted_mean(
                rows, "collapse/normal_centered_rms", normal_count
            ),
            "pseudo_centered_rms": weighted_mean(
                rows, "collapse/pseudo_centered_rms", pseudo_count
            ),
        },
    }


def factorial_effects(cells):
    def effects(metric):
        control = cells["control_2_20"][metric]
        emb_only = cells["emb_only_0p1_20"][metric]
        ring_only = cells["ring_only_2_1"][metric]
        unified = cells["unified_0p1_1"][metric]
        return {
            "embedding_2_minus_0p1_at_hsc_20": control - emb_only,
            "embedding_2_minus_0p1_at_hsc_1": ring_only - unified,
            "hsc_20_minus_1_at_embedding_2": control - ring_only,
            "hsc_20_minus_1_at_embedding_0p1": emb_only - unified,
            "difference_in_differences": control - emb_only - ring_only + unified,
        }

    return {"AUROC": effects("AUROC"), "AUPRC": effects("AUPRC")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence_dir = args.evidence_dir
    output = args.output or evidence_dir / "mechanism-summary.json"

    authoritative_path = evidence_dir / "authoritative-sweep.json"
    results_path = evidence_dir / "results.json"
    replay_path = evidence_dir / "replay.json"
    authoritative = load_json(authoritative_path)
    results = load_json(results_path)
    replay = load_json(replay_path)
    if str(authoritative["state"]).upper() != "FINISHED" or replay["status"] != "passed":
        raise AssertionError("terminal evidence or replay is not valid")
    if sha256_file(authoritative_path) != results["source_hashes"]["authoritative_sweep_sha256"]:
        raise AssertionError("authoritative sweep hash mismatch")

    runs = {run["config"]["mechanism_cell"]: run for run in authoritative["runs"]}
    if set(runs) != set(CELLS):
        raise AssertionError("four-cell coverage mismatch")

    cell_summaries = {}
    initial_raw = {}
    metric_cells = {}
    for cell in CELLS:
        run = runs[cell]
        run_id = run["run_id"]
        diagnostic_path = evidence_dir / "diagnostics" / (run_id + ".jsonl")
        records = load_jsonl(diagnostic_path)
        if len(records) != 304:
            raise AssertionError("{} diagnostic record count mismatch".format(cell))
        updates = records[1:-1]
        terminal = [row for row in updates if row["epoch"] == 150]
        gradients = {
            str(epoch): {
                key: next(
                    row
                    for row in updates
                    if row["epoch"] == epoch and row["batch_index"] == 0
                )[key]
                for key in GRADIENT_KEYS
            }
            for epoch in GRADIENT_EPOCHS
        }
        initial = next(
            row for row in updates if row["epoch"] == 0 and row["batch_index"] == 0
        )
        initial_raw[cell] = {key: initial[key] for key in RAW_INITIAL_KEYS}
        history = [
            {"epoch": int(row["_step"]), "AUROC": row["AUC"], "AUPRC": row["AP"]}
            for row in run["history"]
            if "AUC" in row and "AP" in row
        ]
        final = history[-1]
        metric_cells[cell] = {"AUROC": final["AUROC"], "AUPRC": final["AUPRC"]}
        cell_summaries[cell] = {
            "run_id": run_id,
            "weights": {
                "lambda_rec_emb": run["config"]["lambda_rec_emb"],
                "ring_loss_weight": run["config"]["ring_loss_weight"],
            },
            "final_test": final,
            "evaluation_history": history,
            "terminal_training": summarize_terminal(terminal),
            "gradient_checkpoints_batch_0": gradients,
            "diagnostic_sha256": sha256_file(diagnostic_path),
        }

    reference_initial = initial_raw[CELLS[0]]
    summary = {
        "schema_version": 1,
        "evidence_level": "single_seed_mechanism_probe",
        "selection_policy": "fixed_epoch_150_no_best_checkpoint",
        "source_hashes": {
            "authoritative_sweep_sha256": sha256_file(authoritative_path),
            "results_sha256": sha256_file(results_path),
            "replay_sha256": sha256_file(replay_path),
        },
        "shared_trace_identity": results["shared_trace_identity"],
        "initial_state_audit": {
            "selected_raw_values_exactly_equal_across_cells": all(
                initial_raw[cell] == reference_initial for cell in CELLS
            ),
            "raw_values": reference_initial,
        },
        "cells": cell_summaries,
        "fixed_four_cell_effects": factorial_effects(metric_cells),
        "interpretation_boundary": {
            "continuous_response_surface_supported": False,
            "cross_dataset_mechanism_supported": False,
            "causal_mechanism_proven": False,
            "test_metrics_used_for_selection": False,
        },
    }
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(output), "sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
