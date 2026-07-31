import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from mechanism_diagnostics import (
    append_jsonl,
    build_update_record,
    distribution_metrics,
    gradient_metrics,
    model_state_sha256,
    sha256_file,
    update_index_trace,
    wandb_update_metrics,
)


class MechanismDiagnosticsTest(unittest.TestCase):
    def test_gradient_metrics_accepts_parameter_generator(self):
        model = torch.nn.Linear(2, 1, bias=True)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, -2.0]]))
            model.bias.copy_(torch.tensor([0.5]))
        output = model(torch.tensor([[2.0, 3.0]]))
        losses = {
            "squared": output.square().mean(),
            "linear": output.mean(),
        }
        metrics = gradient_metrics(
            losses,
            model.parameters(),
            {"squared": 2.0, "linear": 3.0},
        )

        self.assertEqual(metrics["gradient/parameter_count"], 3)
        self.assertTrue(math.isfinite(metrics["gradient/weighted_total_norm"]))
        self.assertAlmostEqual(
            metrics["gradient_cosine/squared__linear"], -1.0, places=6
        )

    def test_distribution_rejects_nonfinite_values(self):
        with self.assertRaises(FloatingPointError):
            distribution_metrics("bad", torch.tensor([0.0, float("nan")]))

    def test_update_record_reconstructs_weighted_objective(self):
        model = SimpleNamespace(
            args=SimpleNamespace(ring_R_min=0.3, ring_R_max=1.0),
            last_h_mean=torch.zeros(1, 2),
            last_reconstruction_displacement=torch.tensor([[3.0, 4.0]]),
        )
        record = build_update_record(
            epoch=0,
            batch_index=0,
            global_update=0,
            model=model,
            emb=torch.tensor([[[0.1, 0.0], [0.5, 0.0]]]),
            logits=torch.tensor([[[0.2], [0.7]]]),
            outlier_emb=torch.tensor([[0.5, 0.0]]),
            local_normal_indices=torch.tensor([0]),
            losses={
                "bce": torch.tensor(2.0),
                "token_rec": torch.tensor(3.0),
                "emb_rec": torch.tensor(4.0),
                "rec_combined": torch.tensor(11.0),
                "hsc": torch.tensor(5.0),
                "objective": torch.tensor(113.0),
            },
            weights={
                "bce": 1.0,
                "token_rec": 1.0,
                "emb_rec": 2.0,
                "rec_combined": 1.0,
                "hsc": 20.0,
            },
        )

        self.assertEqual(record["loss/token_rec_weighted"], 3.0)
        self.assertEqual(record["loss/emb_rec_weighted"], 8.0)
        self.assertEqual(record["loss/rec_combined_weighted"], 11.0)
        self.assertEqual(record["loss/true_weighted_total"], 113.0)
        self.assertEqual(record["hsc/shell_hit_rate"], 1.0)
        wandb_metrics = wandb_update_metrics(record)
        self.assertEqual(wandb_metrics["diagnostic/batch_0/global_update"], 0)

    def test_trace_and_file_hashes_are_deterministic(self):
        first = hashlib.sha256()
        second = hashlib.sha256()
        indices = torch.tensor([4, 2, 4, 1])
        update_index_trace(first, "batch", 1, 0, indices)
        update_index_trace(second, "batch", 1, 0, indices)
        self.assertEqual(first.hexdigest(), second.hexdigest())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.jsonl"
            append_jsonl(path, {"finite": 1.0})
            parsed = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(parsed, {"finite": 1.0})
            self.assertEqual(sha256_file(path), sha256_file(path))

    def test_model_state_hash_changes_with_parameters(self):
        model = torch.nn.Linear(2, 1)
        before = model_state_sha256(model)
        with torch.no_grad():
            model.weight.add_(1.0)
        self.assertNotEqual(before, model_state_sha256(model))


if __name__ == "__main__":
    unittest.main()
