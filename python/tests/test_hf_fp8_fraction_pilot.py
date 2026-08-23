import argparse
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "hf_fp8_fraction_pilot",
    ROOT / "benchmarks/single_gpu/hf_fp8_fraction_pilot.py")
PILOT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PILOT)


class HfFp8FractionPilotTest(unittest.TestCase):
    def test_fraction_parser_requires_unique_control_and_valid_range(self):
        self.assertEqual(PILOT.parse_fractions("1,0.5,0.25"), [1.0, 0.5, 0.25])
        for invalid in ("0.5,0.25", "1,0", "1,1", "1,nan"):
            with self.assertRaises(ValueError):
                PILOT.parse_fractions(invalid)

    def test_worker_command_uses_retained_weight_scope_and_fraction(self):
        args = argparse.Namespace(binary=Path("micro"))
        model = {"config": "config.json", "weights": "model.bin",
                 "inference": {"token_ids": [1, 2]}}
        command = PILOT.worker_command(
            args, model, 8, 0.5, Path("logits.bin"))
        self.assertEqual(command[command.index("--fp8-weight-scale-scope") + 1],
                         "attention-output-only")
        self.assertEqual(command[command.index("--fp8-activation-amax-fraction") + 1],
                         "0.5")
        self.assertEqual(command[command.index("--prefill-warmup") + 1], "0")

    def test_selection_requires_strict_worst_rms_improvement_and_stable_top(self):
        rows = [
            {"fraction": 1.0, "worst_normalized_rms": 5.0,
             "worst_normalized_max": 6.0, "top_token_equal_all": True},
            {"fraction": 0.5, "worst_normalized_rms": 4.0,
             "worst_normalized_max": 7.0, "top_token_equal_all": True},
            {"fraction": 0.25, "worst_normalized_rms": 3.0,
             "worst_normalized_max": 3.0, "top_token_equal_all": False},
        ]
        self.assertEqual(PILOT.select_fraction(rows)["selected_fraction"], 0.5)
        rows[1]["worst_normalized_rms"] = 5.1
        self.assertEqual(PILOT.select_fraction(rows)["selected_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
