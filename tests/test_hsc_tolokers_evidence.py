import importlib.util
import os
import statistics
import sys
import types
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
EXPERIMENT = os.path.join(ROOT, "experiments", "hsc-tolokers-deployed-019fbb3f")


def load_module(name, filename):
    previous = sys.modules.get("wandb")
    sys.modules["wandb"] = types.SimpleNamespace()
    try:
        spec = importlib.util.spec_from_file_location(name, os.path.join(EXPERIMENT, filename))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            del sys.modules["wandb"]
        else:
            sys.modules["wandb"] = previous


COLLECTOR = load_module("hsc_tolokers_collector", "collect-evidence.py")
REPLAY = load_module("hsc_tolokers_replay", "replay-results.py")


class HscTolokersEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.records = []
        for condition_index, condition in enumerate(COLLECTOR.CONDITIONS):
            for seed in COLLECTOR.SEEDS:
                record = {
                    "dataset": "tolokers",
                    "hsc_center_condition": condition,
                    "seed": seed,
                }
                for metric_index, metric in enumerate(COLLECTOR.SCALAR_METRICS):
                    record[metric] = condition_index + seed / 10 + metric_index / 100
                self.records.append(record)

    def test_collector_uses_arithmetic_mean_and_sample_std(self):
        aggregate, paired = COLLECTOR.aggregate_records(self.records)
        values = [2 + seed / 10 for seed in COLLECTOR.SEEDS]

        self.assertEqual(aggregate["tolokers"]["q10"]["AUC.last"]["mean"], statistics.mean(values))
        self.assertEqual(
            aggregate["tolokers"]["q10"]["AUC.last"]["sample_std_ddof1"],
            statistics.stdev(values),
        )
        self.assertEqual(paired["tolokers"]["q10_minus_default"]["AUC.last"]["mean"], 2.0)

    def test_replay_recomputes_same_primary_aggregates(self):
        rows = [
            {
                "hsc_center_condition": row["hsc_center_condition"],
                "seed": row["seed"],
                "AUC.last": row["AUC.last"],
                "AP.last": row["AP.last"],
            }
            for row in self.records
        ]
        aggregate, paired = REPLAY.compute_aggregate(rows)

        self.assertEqual(aggregate["q40"]["AUC.last"]["mean"], 5.2)
        self.assertAlmostEqual(paired["q40"]["AP.last"]["mean"], 5.0)
        self.assertTrue(paired["q40"]["AUC.last"]["all_positive"])

    def test_provider_generated_history_artifact_is_the_only_allowed_artifact(self):
        class File:
            name = "0000.parquet"
            size = 42
            digest = "file-digest"

        class Artifact:
            name = "run-run123-history:v0"
            type = "wandb-history"
            state = "COMMITTED"
            entity = "HCCS"
            project = "GGADFormer"
            description = "Weights & Biases Run History Data for run123"
            aliases = ["latest"]
            metadata = {}
            size = 42
            digest = "artifact-digest"

            def files(self):
                return [File()]

        class Run:
            id = "run123"

            def logged_artifacts(self):
                return [Artifact()]

            def used_artifacts(self):
                return []

        audit = COLLECTOR.audit_wandb_artifacts(Run())
        self.assertTrue(audit["only_provider_generated_history_artifact"])
        self.assertEqual(audit["logged_artifacts"][0]["files"][0]["name"], "0000.parquet")
        self.assertTrue(REPLAY.audit_wandb_artifacts(Run())["only_provider_generated_history_artifact"])

    def test_user_artifact_is_rejected(self):
        class Run:
            id = "run123"

            def logged_artifacts(self):
                return []

            def used_artifacts(self):
                return []

        with self.assertRaises(AssertionError):
            COLLECTOR.audit_wandb_artifacts(Run())


if __name__ == "__main__":
    unittest.main()
