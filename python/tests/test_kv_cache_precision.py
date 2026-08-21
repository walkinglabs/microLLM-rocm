import importlib.util
import json
import tempfile
import unittest
from array import array
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "compare_kv_cache_precision",
    ROOT / "benchmarks/single_gpu/compare_kv_cache_precision.py")
PRECISION = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PRECISION)


class KvCachePrecisionTest(unittest.TestCase):
    def test_positive_list_rejects_duplicates_and_nonpositive_values(self):
        self.assertEqual(PRECISION.positive_list("1,8"), [1, 8])
        with self.assertRaisesRegex(Exception, "unique positive"):
            PRECISION.positive_list("1,1")
        with self.assertRaisesRegex(Exception, "unique positive"):
            PRECISION.positive_list("0")

    def test_comparison_requires_bytes_logits_and_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"vocab_size": 4}), encoding="utf-8")
            model = {"name": "tiny", "revision": "fixed", "config": str(config),
                     "inference": {"token_ids": [1, 2]}}
            args = type("Args", (), {"decode_tokens": 4,
                                      "max_absolute_error": 0.25,
                                      "maximum_rmse": 0.05})()
            raw = {
                "generated_tokens": [2, 3, 1, 0],
                "kv_cache_actual_bytes": 80,
                "decode_tokens_per_second": 10.0,
                "engine_peak_bytes": 1000,
            }

            def fake_run(_args, _model, _context, _batch, dtype, _path):
                if dtype == "fp32":
                    return raw, array("f", [0.0, 1.0, 3.0, 2.0])
                bf16 = {**raw, "kv_cache_actual_bytes": 40,
                        "decode_tokens_per_second": 11.0,
                        "engine_peak_bytes": 960}
                return bf16, array("f", [0.01, 1.01, 2.98, 2.0])

            with mock.patch.object(PRECISION, "one_run", side_effect=fake_run):
                result = PRECISION.compare_case(
                    args, model, context=8, batch=1, temporary=Path(directory))
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["cache_byte_reduction"], 2.0)
            self.assertTrue(result["all_logits_finite"])
            self.assertTrue(result["top_tokens_equal"])
            self.assertTrue(result["generated_tokens_equal"])
            self.assertAlmostEqual(result["decode_throughput_ratio_bf16_over_fp32"], 1.1)


if __name__ == "__main__":
    unittest.main()
