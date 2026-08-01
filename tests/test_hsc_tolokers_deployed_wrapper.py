import importlib.util
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
WRAPPER_PATH = os.path.join(
    ROOT,
    "experiments",
    "hsc-tolokers-deployed-019fbb3f",
    "run-sweep-trial.py",
)
SPEC = importlib.util.spec_from_file_location("hsc_tolokers_deployed", WRAPPER_PATH)
WRAPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRAPPER)


class HscTolokersDeployedWrapperTest(unittest.TestCase):
    def test_axis_product_is_exactly_thirty_trials(self):
        self.assertEqual(len(WRAPPER.CONDITIONS) * len(WRAPPER.SEEDS), 30)

    def test_authoritative_arguments_are_exact(self):
        argv = WRAPPER.build_run_argv("tolokers", "q30", 4, executable="python")

        self.assertEqual(argv[:2], ["python", "run.py"])
        self.assertEqual(
            argv[2:-2],
            list(WRAPPER.AUTHORITATIVE_ARGS),
        )
        self.assertEqual(argv[-2:], ["--hsc_center_condition=q30", "--seed=4"])

    def test_authoritative_values_match_deployed_default(self):
        argv = WRAPPER.build_run_argv("tolokers", "default", 0)
        required = {
            "--batch_size=1024",
            "--data_split_seed=42",
            "--end_lr=0.00001",
            "--lambda_rec_emb=0.1",
            "--num_epoch=100",
            "--outlier_beta=0.3",
            "--peak_lr=0.0001",
            "--pp_k=10",
            "--progregate_alpha=0.9",
            "--rec_loss_weight=1",
            "--ring_R_max=1",
            "--ring_R_min=0.3",
            "--ring_loss_weight=1",
            "--sample_rate=0.15",
            "--train_rate=0.05",
            "--warmup_updates=5",
        }
        self.assertTrue(required.issubset(argv))

    def test_old_tolokers_protocol_is_absent(self):
        argv = WRAPPER.build_run_argv("tolokers", "default", 0)
        for obsolete in (
            "--lambda_rec_emb=0.5",
            "--num_epoch=70",
            "--pp_k=3",
            "--progregate_alpha=0.3",
            "--rec_loss_weight=0.1",
            "--ring_R_min=0.5",
            "--ring_R_max=0.5",
            "--ring_loss_weight=20",
            "--warmup_updates=50",
        ):
            self.assertNotIn(obsolete, argv)

    def test_invalid_axis_values_fail_closed(self):
        with self.assertRaises(ValueError):
            WRAPPER.build_run_argv("tolokers", "q50", 0)
        with self.assertRaises(ValueError):
            WRAPPER.build_run_argv("Amazon", "q10", 0)
        with self.assertRaises(ValueError):
            WRAPPER.build_run_argv("tolokers", "q10", 5)


if __name__ == "__main__":
    unittest.main()
