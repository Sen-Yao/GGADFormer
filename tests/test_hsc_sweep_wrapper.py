import importlib.util
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
WRAPPER_PATH = os.path.join(
    ROOT,
    "experiments",
    "hsc-center-contamination-019fb5c1",
    "run-sweep-trial.py",
)
SPEC = importlib.util.spec_from_file_location("hsc_sweep_trial", WRAPPER_PATH)
WRAPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRAPPER)


class HscSweepWrapperTest(unittest.TestCase):
    def test_axis_product_is_exactly_sixty_trials(self):
        self.assertEqual(len(WRAPPER.DATASET_ARGS) * len(WRAPPER.CONDITIONS) * 5, 60)

    def test_amazon_authoritative_arguments(self):
        argv = WRAPPER.build_run_argv("Amazon", "q30", 4, executable="python")

        self.assertEqual(argv[:2], ["python", "run.py"])
        self.assertIn("--dataset=Amazon", argv)
        self.assertIn("--num_epoch=100", argv)
        self.assertIn("--lambda_rec_emb=0.1", argv)
        self.assertIn("--ring_loss_weight=1", argv)
        self.assertIn("--hsc_center_condition=q30", argv)
        self.assertEqual(argv[-1], "--seed=4")

    def test_tolokers_uses_old_main_table_arguments(self):
        argv = WRAPPER.build_run_argv("tolokers", "default", 0, executable="python")

        self.assertIn("--dataset=tolokers", argv)
        self.assertIn("--num_epoch=70", argv)
        self.assertIn("--lambda_rec_emb=0.5", argv)
        self.assertIn("--rec_loss_weight=0.1", argv)
        self.assertIn("--ring_R_min=0.5", argv)
        self.assertIn("--ring_R_max=0.5", argv)
        self.assertIn("--ring_loss_weight=20", argv)

    def test_dataset_specific_arguments_do_not_cross(self):
        amazon = WRAPPER.build_run_argv("Amazon", "default", 0)
        tolokers = WRAPPER.build_run_argv("tolokers", "default", 0)

        self.assertIn("--peak_lr=0.0003", amazon)
        self.assertNotIn("--peak_lr=0.0003", tolokers)
        self.assertIn("--peak_lr=0.0001", tolokers)
        self.assertNotIn("--lambda_rec_emb=0.5", amazon)

    def test_invalid_axis_values_fail_closed(self):
        with self.assertRaises(ValueError):
            WRAPPER.build_run_argv("Amazon", "q50", 0)
        with self.assertRaises(ValueError):
            WRAPPER.build_run_argv("unknown", "q10", 0)
        with self.assertRaises(ValueError):
            WRAPPER.build_run_argv("Amazon", "q10", 5)


if __name__ == "__main__":
    unittest.main()
