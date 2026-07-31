import os
import sys
import unittest

import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hsc_center import (  # noqa: E402
    HSC_CENTER_Q,
    compute_center_components,
    compute_hsc_center,
    compute_shell_statistics,
)


class HscCenterTest(unittest.TestCase):
    def setUp(self):
        self.emb = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [4.0, 2.0],
                    [8.0, 4.0],
                    [10.0, 6.0],
                ]
            ],
            dtype=torch.float64,
        )
        self.labels = torch.tensor([0, 0, 0, 0, 1])

    def test_default_matches_batch_mean_without_labels(self):
        actual = compute_hsc_center(self.emb, None, "default")
        expected = self.emb.mean(dim=1, keepdim=True)
        self.assertTrue(torch.equal(actual, expected))

    def test_q_endpoints_and_interpolation(self):
        q0 = compute_center_components(self.emb, self.labels, "q0")
        q40 = compute_center_components(self.emb, self.labels, "q40")

        self.assertTrue(torch.equal(q0.selected, q0.normal))
        expected_q40 = 0.6 * q40.normal + 0.4 * q40.anomaly
        self.assertTrue(torch.allclose(q40.selected, expected_q40, atol=0.0, rtol=0.0))

    def test_q_equal_batch_anomaly_fraction_matches_default(self):
        components = compute_center_components(self.emb, self.labels, "q20")

        self.assertEqual(HSC_CENTER_Q["q20"], float(components.anomaly_fraction))
        self.assertTrue(
            torch.allclose(
                components.selected,
                components.default,
                atol=1e-12,
                rtol=0.0,
            )
        )

    def test_center_retains_autograd(self):
        emb = self.emb.clone().requires_grad_(True)
        center = compute_hsc_center(emb, self.labels, "q30")

        center.square().sum().backward()

        self.assertIsNotNone(emb.grad)
        self.assertGreater(float(emb.grad.norm()), 0.0)

    def test_center_helpers_do_not_advance_rng(self):
        torch.manual_seed(934)
        before = torch.get_rng_state().clone()

        for condition in HSC_CENTER_Q:
            compute_hsc_center(self.emb, self.labels, condition)

        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_invalid_condition_and_missing_classes_fail_closed(self):
        with self.assertRaises(ValueError):
            compute_hsc_center(self.emb, self.labels, "q50")
        with self.assertRaises(ValueError):
            compute_hsc_center(self.emb, torch.zeros(5), "q10")
        with self.assertRaises(ValueError):
            compute_hsc_center(self.emb, torch.ones(5), "q10")
        with self.assertRaises(ValueError):
            compute_hsc_center(self.emb, torch.tensor([0, 0, 0, 0, 2]), "q10")

    def test_shell_statistics_partition_and_loss_identity(self):
        center = torch.zeros((1, 1, 2), dtype=torch.float64)
        outliers = torch.tensor(
            [[0.1, 0.0], [0.5, 0.0], [1.0, 0.0], [1.4, 0.0]],
            dtype=torch.float64,
        )

        stats = compute_shell_statistics(outliers, center, 0.3, 1.0)

        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["inner_count"], 1)
        self.assertEqual(stats["shell_count"], 2)
        self.assertEqual(stats["outer_count"], 1)
        self.assertEqual(
            stats["count"],
            stats["inner_count"] + stats["shell_count"] + stats["outer_count"],
        )
        self.assertAlmostEqual(stats["hsc_loss_sum"], 0.6, places=12)

    def test_invalid_shell_bounds_fail_closed(self):
        center = torch.zeros((1, 1, 2))
        outliers = torch.ones((2, 2))

        with self.assertRaises(ValueError):
            compute_shell_statistics(outliers, center, 1.0, 0.5)


if __name__ == "__main__":
    unittest.main()
