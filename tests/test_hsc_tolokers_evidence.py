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


if __name__ == "__main__":
    unittest.main()
