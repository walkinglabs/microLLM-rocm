import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks/single_gpu"
sys.path.insert(0, str(BENCHMARKS))
SPEC = importlib.util.spec_from_file_location(
    "hf_fp8_block_detail", BENCHMARKS / "hf_fp8_block_detail.py")
DETAIL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DETAIL)


class HfFp8BlockDetailTest(unittest.TestCase):
    def test_compare_requires_identical_order_and_reports_error(self):
        left = [{"name": "inference.blocks.21.ffn_norm", "shape": [1], "values": [1.0]}]
        right = [{"name": "inference.blocks.21.ffn_norm", "shape": [1], "values": [1.5]}]
        self.assertEqual(DETAIL.compare(left, right)[0]["max_abs"], 0.5)
        right[0]["name"] = "wrong"
        with self.assertRaises(RuntimeError):
            DETAIL.compare(left, right)

    def test_default_layers_are_evidence_selected(self):
        self.assertEqual(DETAIL.DEFAULT_LAYERS["qwen2.5-0.5b"], 21)
        self.assertEqual(DETAIL.DEFAULT_LAYERS[
            "deepseek-r1-distill-qwen-1.5b"], 27)


if __name__ == "__main__":
    unittest.main()
