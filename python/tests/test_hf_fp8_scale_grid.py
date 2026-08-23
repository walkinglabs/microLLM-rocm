import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks/single_gpu"
sys.path.insert(0, str(BENCHMARKS))
SPEC = importlib.util.spec_from_file_location(
    "hf_fp8_scale_grid", BENCHMARKS / "hf_fp8_scale_grid.py")
GRID = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GRID)


class HfFp8ScaleGridTest(unittest.TestCase):
    def test_positive_grid_rejects_duplicates_and_nonpositive_values(self):
        self.assertEqual(GRID.parse_positive_grid("0.1,0.2", "scale"), [0.1, 0.2])
        with self.assertRaises(Exception):
            GRID.parse_positive_grid("0.1,0.1", "scale")
        with self.assertRaises(Exception):
            GRID.parse_positive_grid("0", "scale")

    def test_command_passes_each_fixed_scale_pair(self):
        args = type("Args", (), {
            "binary": Path("micro"), "warmup": 1, "steps": 3, "context": 8})()
        model = {"config": "config.json", "weights": "model.bin",
                 "inference": {"token_ids": [1, 2]}}
        command = GRID.command(args, model, "fp8", Path("logits.bin"), 0.0125, 0.0025)
        self.assertEqual(command[command.index("--fp8-activation-scale") + 1], "0.0125")
        self.assertEqual(command[command.index("--fp8-weight-scale") + 1], "0.0025")

    def test_selection_uses_complete_logit_gate_before_speed(self):
        slow_pass = {"precision_gate_passed": True, "top_token_equal": True,
                     "root_mean_square_error": 0.04, "maximum_absolute_error": 0.1,
                     "prefill_tokens_per_second": 1.0}
        fast_fail = {"precision_gate_passed": False, "top_token_equal": True,
                     "root_mean_square_error": 0.01, "maximum_absolute_error": 0.1,
                     "prefill_tokens_per_second": 100.0}
        self.assertIs(GRID.select_best([fast_fail, slow_pass]), slow_pass)


if __name__ == "__main__":
    unittest.main()
