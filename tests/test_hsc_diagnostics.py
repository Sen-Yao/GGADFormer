import math
import unittest

import torch

from hsc_diagnostics import hsc_batch_metrics


class HscDiagnosticsTest(unittest.TestCase):
    def test_shell_rates_and_losses(self):
        emb = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], requires_grad=True)
        outlier_emb = torch.tensor(
            [[0.1, 0.0], [0.5, 0.0], [1.1, 0.0]], requires_grad=True
        )
        loss_bce = torch.tensor(2.0, requires_grad=True)
        loss_rec = torch.tensor(3.0, requires_grad=True)
        loss_ring = torch.tensor(4.0, requires_grad=True)
        loss_total = loss_bce + loss_rec + 20.0 * loss_ring

        metrics = hsc_batch_metrics(
            batch_index=2,
            emb=emb,
            outlier_emb=outlier_emb,
            loss_bce=loss_bce,
            loss_rec=loss_rec,
            loss_ring=loss_ring,
            loss_total=loss_total,
            ring_r_min=0.3,
            ring_r_max=1.0,
            rec_loss_weight=1.0,
            ring_loss_weight=20.0,
            bce_loss_weight=1.0,
        )

        prefix = "diagnostic/batch_2"
        self.assertEqual(metrics[f"{prefix}/pseudo_count"], 3)
        self.assertAlmostEqual(metrics[f"{prefix}/hsc/shell_hit_rate"], 1 / 3)
        self.assertAlmostEqual(
            metrics[f"{prefix}/hsc/inner_violation_rate"], 1 / 3
        )
        self.assertAlmostEqual(
            metrics[f"{prefix}/hsc/outer_violation_rate"], 1 / 3
        )
        self.assertEqual(metrics[f"{prefix}/loss/hsc_weighted"], 80.0)
        self.assertEqual(metrics[f"{prefix}/loss/true_weighted_total"], 85.0)

    def test_diagnostics_do_not_attach_to_autograd(self):
        parameter = torch.tensor([2.0], requires_grad=True)
        emb = (parameter * torch.ones(1, 2, 1)).requires_grad_()
        outlier_emb = parameter * torch.tensor([[0.5], [1.5]])
        loss = parameter.square().sum()

        metrics = hsc_batch_metrics(
            batch_index=0,
            emb=emb,
            outlier_emb=outlier_emb,
            loss_bce=loss,
            loss_rec=loss,
            loss_ring=loss,
            loss_total=loss,
            ring_r_min=0.3,
            ring_r_max=1.0,
            rec_loss_weight=1.0,
            ring_loss_weight=1.0,
            bce_loss_weight=1.0,
        )
        loss.backward()

        self.assertEqual(parameter.grad.item(), 4.0)
        self.assertTrue(all(math.isfinite(float(value)) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
