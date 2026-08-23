import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "mi300_int8_probe", ROOT / "benchmarks/single_gpu/mi300_int8_probe.py")
PROBE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PROBE)


class Mi300Int8ProbeTest(unittest.TestCase):
    def test_enrich_adds_integer_roofline_and_preserves_exact_contract(self):
        result = PROBE.enrich({
            "shape": [256, 256, 256], "input_dtype": "int8",
            "output_dtype": "int32", "accuracy_passed": True,
            "maximum_sample_error": 0, "achieved_tops": 10.0}, 256)
        self.assertEqual(result["arithmetic_intensity_ops_per_byte"], 256 / 3)
        self.assertAlmostEqual(result["bandwidth_bound_tops"], 5.3 * 256 / 3)
        self.assertLess(result["roofline_utilization"], 1.0)

    def test_nonexact_or_wrong_dtype_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "contract or exact"):
            PROBE.enrich({
                "shape": [2, 2, 2], "input_dtype": "int8",
                "output_dtype": "int32", "accuracy_passed": False,
                "maximum_sample_error": 1, "achieved_tops": 1.0}, 2)

    def test_boundary_does_not_claim_tensor_or_transformer_support(self):
        source = (ROOT / "benchmarks/single_gpu/mi300_int8_probe.py").read_text()
        self.assertIn("no public Tensor dtype", source)
        self.assertIn("no public Tensor dtype, quantizer", source)


if __name__ == "__main__":
    unittest.main()
