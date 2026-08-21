import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX_DIRECTORY = ROOT / "benchmarks/single_gpu"
sys.path.insert(0, str(MATRIX_DIRECTORY))
SPEC = importlib.util.spec_from_file_location(
    "compare_kv_cache_policies",
    MATRIX_DIRECTORY / "compare_kv_cache_policies.py")
POLICIES = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(POLICIES)


class KvCachePolicyComparisonTest(unittest.TestCase):
    def test_summary_requires_complete_pairs_and_computes_medians(self):
        model = {"name": "tiny", "revision": "fixed"}
        records = []
        for policy, throughputs, prepare in (
                ("uniform", [100.0, 110.0, 120.0], [10.0, 11.0, 12.0]),
                ("candidate", [90.0, 99.0, 108.0], [8.0, 9.0, 10.0])):
            for throughput, prepare_ms in zip(throughputs, prepare):
                records.append({
                    "model": "tiny", "context": 32, "batch": 1,
                    "policy": policy, "status": "pass",
                    "throughput_tokens_per_second": throughput,
                    "mean_cache_prepare_ms": prepare_ms,
                    "mean_end_to_end_generation_ms": prepare_ms + 5.0,
                    "peak_bytes": 1000, "kv_cache_actual_bytes": 100,
                    "generated_tokens": [1, 2],
                })
        summary = POLICIES.summarize(records, [model], [32], [1], 3)
        row = summary["rows"][0]
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(row["uniform_throughput_tokens_per_second"], 110.0)
        self.assertEqual(row["candidate_throughput_tokens_per_second"], 99.0)
        self.assertAlmostEqual(row["throughput_ratio_candidate_over_uniform"], 0.9)
        self.assertAlmostEqual(row["prepare_speedup"], 11.0 / 9.0)
        self.assertTrue(row["tokens_equal"])

        incomplete = POLICIES.summarize(records[:-1], [model], [32], [1], 3)
        self.assertEqual(incomplete["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
