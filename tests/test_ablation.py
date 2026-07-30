import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ablation import apply_perturbation_ablation, permute_magnitudes_cyclic


class PerturbationAblationTest(unittest.TestCase):
    def make_generator(self, seed):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return generator

    def test_permutation_exact_multiset_and_no_fixed_points(self):
        magnitudes = torch.tensor([[1.0], [2.0], [3.0], [4.0]])

        permuted = permute_magnitudes_cyclic(
            magnitudes, generator=self.make_generator(123)
        )

        self.assertTrue(torch.equal(torch.sort(permuted.flatten()).values, magnitudes.flatten()))
        self.assertFalse(torch.any(permuted.flatten() == magnitudes.flatten()).item())

    def test_permutation_singleton_is_explicit_identity(self):
        magnitudes = torch.tensor([[3.5]])

        permuted = permute_magnitudes_cyclic(
            magnitudes, generator=self.make_generator(123)
        )

        self.assertTrue(torch.equal(permuted, magnitudes))
        self.assertNotEqual(permuted.data_ptr(), magnitudes.data_ptr())

    def test_invalid_magnitude_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            permute_magnitudes_cyclic(torch.ones(3))

    def test_random_mag_and_random_both_share_permutation_semantics(self):
        vectors = torch.tensor(
            [[3.0, 4.0], [0.0, 2.0], [5.0, 12.0], [8.0, 15.0]],
            dtype=torch.float32,
        )

        random_mag = apply_perturbation_ablation(
            vectors,
            "random_mag",
            magnitude_generator=self.make_generator(777),
        )
        random_both = apply_perturbation_ablation(
            vectors,
            "random_both",
            direction_generator=self.make_generator(888),
            magnitude_generator=self.make_generator(777),
        )

        self.assertTrue(
            torch.allclose(
                torch.norm(random_mag, p=2, dim=1),
                torch.norm(random_both, p=2, dim=1),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.equal(
                torch.sort(torch.norm(random_mag, p=2, dim=1)).values,
                torch.sort(torch.norm(vectors, p=2, dim=1)).values,
            )
        )

    def test_explicit_generators_do_not_advance_global_rng(self):
        vectors = torch.randn(5, 3)
        torch.manual_seed(999)
        before = torch.get_rng_state().clone()

        _ = apply_perturbation_ablation(
            vectors,
            "random_both",
            direction_generator=self.make_generator(111),
            magnitude_generator=self.make_generator(222),
        )

        after = torch.get_rng_state()
        self.assertTrue(torch.equal(before, after))

    def test_explicit_generator_determinism(self):
        vectors = torch.randn(6, 4)

        first = apply_perturbation_ablation(
            vectors,
            "random_both",
            direction_generator=self.make_generator(1),
            magnitude_generator=self.make_generator(2),
        )
        second = apply_perturbation_ablation(
            vectors,
            "random_both",
            direction_generator=self.make_generator(1),
            magnitude_generator=self.make_generator(2),
        )

        self.assertTrue(torch.equal(first, second))

    def test_zero_norm_rows_do_not_create_nan(self):
        vectors = torch.zeros(3, 4)

        for mode in ["random_dir", "random_mag", "random_both", "constant_mag"]:
            output = apply_perturbation_ablation(
                vectors,
                mode,
                direction_generator=self.make_generator(10),
                magnitude_generator=self.make_generator(20),
            )
            self.assertEqual(tuple(output.shape), tuple(vectors.shape))
            self.assertFalse(torch.isnan(output).any().item())
            self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_gradient_flows_through_permuted_magnitudes(self):
        vectors = torch.tensor(
            [[3.0, 4.0], [0.0, 2.0], [5.0, 12.0]], requires_grad=True
        )

        random_both = apply_perturbation_ablation(
            vectors,
            "random_both",
            direction_generator=self.make_generator(42),
            magnitude_generator=self.make_generator(24),
        )
        loss = random_both.pow(2).sum()
        loss.backward()

        self.assertIsNotNone(vectors.grad)
        self.assertFalse(torch.isnan(vectors.grad).any().item())
        self.assertGreater(torch.norm(vectors.grad).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
