import ast
import importlib
import random
import shlex
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import scipy.sparse as sp
import torch


PACKAGE = Path(__file__).resolve().parents[1] / "package-v2"
sys.path.insert(0, str(PACKAGE))

import controls
import data
import tokenization


class TokenizationTests(unittest.TestCase):
    def setUp(self):
        adjacency = sp.coo_matrix(
            np.array(
                [
                    [1.0, 0.5, 0.0, 0.0],
                    [0.5, 1.0, 0.25, 0.0],
                    [0.0, 0.25, 1.0, 0.75],
                    [0.0, 0.0, 0.75, 1.0],
                ],
                dtype=np.float32,
            )
        )
        self.adjacency = data.scipy_to_torch_sparse(adjacency)
        self.features = torch.tensor(
            [[1.0, 2.0], [0.5, -1.0], [3.0, 0.0], [-2.0, 4.0]],
            dtype=torch.float32,
        )

    def legacy_tokens(self, num_hops, alpha):
        tokens = [self.features]
        for hop in range(1, num_hops + 1):
            current = self.features
            for _ in range(hop):
                current = (
                    (1.0 - alpha) * torch.sparse.mm(self.adjacency, current)
                    + alpha * self.features
                )
            tokens.append(current)
        return torch.stack(tokens, dim=1)

    def test_incremental_matches_each_legacy_hop(self):
        expected = self.legacy_tokens(num_hops=5, alpha=0.3)
        actual = tokenization.incremental_tokenization(
            self.features, self.adjacency, num_hops=5, alpha=0.3
        )
        for hop in range(6):
            torch.testing.assert_close(actual[:, hop], expected[:, hop], rtol=1e-6, atol=1e-7)

    def test_exactly_k_sparse_propagations(self):
        original = tokenization.propagate_once
        calls = []

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        with mock.patch.object(tokenization, "propagate_once", side_effect=counted):
            tokenization.incremental_tokenization(
                self.features, self.adjacency, num_hops=7, alpha=0.2
            )
        self.assertEqual(len(calls), 7)


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.vector = torch.tensor(
            [[3.0, 4.0, 0.0], [0.0, 0.0, 2.0], [1.0, 2.0, 2.0]],
            requires_grad=True,
        )

    @staticmethod
    def generator(seed):
        result = torch.Generator(device="cpu")
        result.manual_seed(seed)
        return result

    def test_random_magnitude_preserves_exact_multiset(self):
        output = controls.apply_control(
            self.vector, "random_mag", self.generator(10), self.generator(20)
        )
        before = torch.linalg.vector_norm(self.vector, dim=1).sort().values
        after = torch.linalg.vector_norm(output, dim=1).sort().values
        torch.testing.assert_close(after, before, rtol=0, atol=0)

    def test_direction_rng_is_independent_of_magnitude_rng(self):
        first = controls.apply_control(
            self.vector, "random_dir", self.generator(123), self.generator(1)
        )
        second = controls.apply_control(
            self.vector, "random_dir", self.generator(123), self.generator(999)
        )
        torch.testing.assert_close(first, second, rtol=0, atol=0)

        first_both = controls.apply_control(
            self.vector, "random_both", self.generator(123), self.generator(20)
        )
        second_both = controls.apply_control(
            self.vector, "random_both", self.generator(123), self.generator(21)
        )
        torch.testing.assert_close(
            controls.normalize_direction(first_both),
            controls.normalize_direction(second_both),
            rtol=1e-6,
            atol=1e-7,
        )

        third_both = controls.apply_control(
            self.vector, "random_both", self.generator(124), self.generator(20)
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(first_both, dim=1),
            torch.linalg.vector_norm(third_both, dim=1),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_zero_vectors_are_finite(self):
        zeros = torch.zeros(3, 4)
        for name in controls.CONTROL_NAMES:
            output = controls.apply_control(
                zeros, name, self.generator(4), self.generator(5)
            )
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(torch.count_nonzero(output), 0)

    def test_singleton_magnitude_control_is_identity(self):
        singleton = torch.tensor([[3.0, 4.0]])
        output = controls.apply_control(
            singleton, "random_mag", self.generator(6), self.generator(7)
        )
        torch.testing.assert_close(output, singleton, rtol=0, atol=0)

    def test_all_controls_retain_a_gradient_path(self):
        for name in controls.CONTROL_NAMES:
            vector = self.vector.detach().clone().requires_grad_(True)
            output = controls.apply_control(
                vector, name, self.generator(8), self.generator(9)
            )
            output.sum().backward()
            self.assertIsNotNone(vector.grad)
            self.assertTrue(torch.isfinite(vector.grad).all())


class ProtocolTests(unittest.TestCase):
    def test_data_split_preserves_global_rng_states(self):
        labels = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0], dtype=np.float32)
        random.seed(17)
        np.random.seed(19)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        data._split_indices(labels, 0.2, 0.1, 42)
        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_before[0])
        np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
        self.assertEqual(numpy_after[2:], numpy_before[2:])

    def test_standard_operator_matches_formal_definition(self):
        adjacency = sp.csr_matrix(
            np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 0.0]])
        )
        degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
        diagonal = sp.diags(np.power(degrees, -0.5))
        expected = diagonal.dot(adjacency).dot(diagonal) + sp.eye(3)
        actual = data._standard_adjacency(adjacency)
        np.testing.assert_allclose(actual.toarray(), expected.toarray(), rtol=1e-12, atol=1e-12)

    def test_formal_module_parameter_order(self):
        run = importlib.import_module("run")
        parser = run.build_parser()
        args = parser.parse_args(
            "--dataset Amazon --batch_size 16 --num_epoch 10 --peak_lr 0.001 "
            "--end_lr 0.0001 --warmup_updates 2 --pp_k 2 --progregate_alpha 0.4".split()
        )
        args.noise_mean = 0.0
        args.noise_std = 0.0
        model_module = importlib.import_module("vecgad")
        model = model_module.VecGAD(5, args, torch.device("cpu"))
        names = list(model._modules)
        expected_prefix = ["gcn1", "gcn2", "fc1", "fc2", "fc3", "fc4"]
        self.assertEqual(names[:6], expected_prefix)
        self.assertEqual(model.layers[0].attention.scale, 1.0)

    def test_synthetic_training_objective_backpropagates(self):
        run = importlib.import_module("run")
        args = run.build_parser().parse_args(
            "--dataset Amazon --batch_size 8 --num_epoch 1 --peak_lr 0.001 "
            "--end_lr 0.0001 --warmup_updates 1 --pp_k 2 --progregate_alpha 0.4 "
            "--sample_rate 0.5 --embedding_dim 16 --ffn_dim 16 --num_heads 2 "
            "--num_layers 2".split()
        )
        args.noise_mean = 0.0
        args.noise_std = 0.0
        model_module = importlib.import_module("vecgad")
        model = model_module.VecGAD(5, args, torch.device("cpu"))
        tokens = torch.randn(8, 3, 5)
        logits, reconstruction_loss, hsc_loss, source_count = model.training_objectives(
            tokens, torch.tensor([0, 1, 2, 3], dtype=torch.long)
        )
        self.assertEqual(tuple(logits.shape), (1, 4 + source_count, 1))
        loss = logits.mean() + reconstruction_loss + hsc_loss
        loss.backward()
        self.assertIsNotNone(model.reconstruction_projection[0].weight.grad)
        self.assertTrue(torch.isfinite(model.reconstruction_projection[0].weight.grad).all())


class ArtifactTests(unittest.TestCase):
    def test_exact_public_file_set(self):
        expected = {
            "README.md",
            "REVIEW_USE.md",
            "environment.yml",
            "run.py",
            "vecgad.py",
            "data.py",
            "tokenization.py",
            "controls.py",
            "reproduction.sh",
        }
        actual = {path.name for path in PACKAGE.iterdir() if path.is_file()}
        self.assertEqual(actual, expected)

    def test_python_files_compile(self):
        for path in PACKAGE.glob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_anonymous_content_scan(self):
        forbidden = (
            "/Users/",
            "/root/",
            "wandb",
            "HCCS",
            "github.com/",
            "api_key",
            "checkpoint",
            "torch.save",
        )
        for path in PACKAGE.iterdir():
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker.lower(), text.lower(), str(path))

    def test_command_catalog_is_non_executing_and_parseable(self):
        completed = subprocess.run(
            ["bash", str(PACKAGE / "reproduction.sh")],
            cwd=PACKAGE,
            check=True,
            capture_output=True,
            text=True,
        )
        commands = [line for line in completed.stdout.splitlines() if line.startswith("python run.py")]
        self.assertEqual(len(commands), 7)
        run = importlib.import_module("run")
        parser = run.build_parser()
        parsed = []
        for command in commands:
            arguments = shlex.split(command)[2:]
            parsed.append(parser.parse_args(arguments))
        self.assertEqual({args.dataset for args in parsed}, set(data.DATASETS))
        for args in parsed:
            self.assertEqual(args.seed, 0)
            self.assertEqual(args.data_split_seed, 42)
            self.assertEqual(args.sample_rate, 0.15)
            self.assertEqual(args.control, "full")

        expected = {
            "Amazon": (1024, 100, 5, 0.4, 0.1, 1.0, 50),
            "reddit": (1024, 200, 10, 0.1, 0.1, 1.0, 50),
            "photo": (128, 200, 6, 0.1, 0.1, 1.0, 50),
            "elliptic": (32768, 150, 7, 0.6, 2.0, 20.0, 50),
            "t_finance": (8192, 40, 7, 0.3, 0.1, 1.0, 50),
            "tolokers": (1024, 100, 10, 0.9, 0.1, 1.0, 5),
            "dgraph": (65536, 200, 10, 0.9, 0.1, 1.0, 5),
        }
        for args in parsed:
            observed = (
                args.batch_size,
                args.num_epoch,
                args.pp_k,
                args.progregate_alpha,
                args.lambda_rec_emb,
                args.ring_loss_weight,
                args.warmup_updates,
            )
            self.assertEqual(observed, expected[args.dataset])


if __name__ == "__main__":
    unittest.main(verbosity=2)
