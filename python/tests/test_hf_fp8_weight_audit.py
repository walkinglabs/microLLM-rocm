import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "hf_fp8_weight_audit",
    ROOT / "benchmarks/single_gpu/hf_fp8_weight_audit.py")
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


class HfFp8WeightAuditTest(unittest.TestCase):
    def test_classifies_only_supported_linear_families(self):
        self.assertEqual(AUDIT.classify_weight(
            "model.layers.0.self_attn.q_proj.weight"), "attention")
        self.assertEqual(AUDIT.classify_weight(
            "model.layers.0.mlp.down_proj.weight"), "ffn")
        self.assertEqual(AUDIT.classify_weight("lm_head.weight"), "output_head")
        self.assertIsNone(AUDIT.classify_weight("model.embed_tokens.weight"))
        self.assertIsNone(AUDIT.classify_weight("model.layers.0.input_layernorm.weight"))

    def test_summary_combines_squared_error_before_relative_l2(self):
        rows = [
            {"model": "m", "group": "attention", "elements": 2,
             "scalar_squared_error": 4.0, "column_squared_error": 1.0,
             "reference_squared_sum": 16.0, "scalar_max_abs": 2.0,
             "column_max_abs": 1.0},
            {"model": "m", "group": "attention", "elements": 2,
             "scalar_squared_error": 5.0, "column_squared_error": 3.0,
             "reference_squared_sum": 20.0, "scalar_max_abs": 3.0,
             "column_max_abs": 1.5},
        ]
        aggregates = AUDIT.summarize(rows)
        attention = next(row for row in aggregates
                         if row["group"] == "attention")
        self.assertAlmostEqual(attention["scalar_relative_l2"], 0.5)
        self.assertAlmostEqual(attention["column_relative_l2"], 1.0 / 3.0)
        self.assertEqual(attention["tensors"], 2)


if __name__ == "__main__":
    unittest.main()
