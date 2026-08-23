import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "mi300_precision_roofline",
    ROOT / "benchmarks/single_gpu/mi300_precision_roofline.py")
ROOFLINE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ROOFLINE)


class Mi300PrecisionRooflineTest(unittest.TestCase):
    def test_roofline_record_computes_executed_and_hardware_bounds(self):
        record = {
            "shape": [512, 512, 512], "dtype": "bf16",
            "median_ms": 0.05, "max_abs_error": 0.01,
            "accuracy_passed": True,
        }
        result = ROOFLINE.roofline_record(record, 512)
        self.assertAlmostEqual(result["achieved_tflops"], 5.36870912)
        self.assertEqual(result["official_peak_tflops"], 1307.4)
        self.assertAlmostEqual(
            result["arithmetic_intensity_flops_per_byte"], 512 / 3)
        self.assertLessEqual(result["roofline_bound_tflops"], 1307.4)
        self.assertLess(result["official_peak_utilization"], 0.01)

    def test_fp8_counts_one_byte_inputs_and_bf16_output(self):
        record = {
            "shape": [256, 256, 256], "dtype": "fp8_e4m3_fnuz",
            "median_ms": 0.04, "max_abs_error": 0.05,
            "accuracy_passed": True,
        }
        result = ROOFLINE.roofline_record(record, 256)
        self.assertEqual(result["official_peak_tflops"], 2614.9)
        self.assertEqual(result["arithmetic_intensity_flops_per_byte"], 128)
        self.assertAlmostEqual(result["bandwidth_bound_tflops"], 678.4)

    def test_bad_shape_or_accuracy_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "unknown dtype or shape"):
            ROOFLINE.roofline_record({
                "shape": [1, 2, 3], "dtype": "bf16", "median_ms": 1,
                "accuracy_passed": True}, 2)
        with self.assertRaisesRegex(RuntimeError, "timing or accuracy"):
            ROOFLINE.roofline_record({
                "shape": [2, 2, 2], "dtype": "bf16", "median_ms": 1,
                "accuracy_passed": False}, 2)

    def test_runner_exposes_cpu_and_fp32_reference_modes(self):
        source = (ROOT / "benchmarks/single_gpu/mi300_precision_roofline.py").read_text()
        self.assertIn('choices=("cpu", "fp32")', source)
        self.assertIn('"--reference", args.reference', source)


if __name__ == "__main__":
    unittest.main()
