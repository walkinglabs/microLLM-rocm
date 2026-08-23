import argparse
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "hf_fp8_native_roundtrip",
    ROOT / "benchmarks/single_gpu/hf_fp8_native_roundtrip.py")
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


class HfFp8NativeRoundtripTest(unittest.TestCase):
    def test_policy_orders_rotate_all_three_positions(self):
        for policy in ("fp32", "full", "both-roundtrip"):
            self.assertEqual(
                sorted(order.index(policy) for order in RUNNER.POLICY_ORDERS),
                [0, 1, 2])

    def test_worker_command_names_both_roundtrip_and_fixed_measurement(self):
        args = argparse.Namespace(
            binary=Path("micro"), fp8_activation_minimum_scale=0.0001,
            fp8_weight_scale=0.005)
        model = {"config": "config.json", "weights": "model.bin",
                 "inference": {"token_ids": [1, 2]}}
        command = RUNNER.worker_command(
            args, model, 8, "both-roundtrip", Path("logits.bin"))
        self.assertEqual(command[command.index("--fp8-diagnostic-mode") + 1],
                         "both-roundtrip")
        self.assertEqual(command[command.index("--prefill-warmup") + 1], "0")
        self.assertEqual(command[command.index("--prefill-steps") + 1], "1")

    def test_bundle_compares_native_roundtrip_and_fp32_directly(self):
        bundle = RUNNER.comparison_bundle(
            [1.0, 2.2], [1.0, 2.1], [1.0, 2.0])
        self.assertEqual(set(bundle), {
            "full_vs_fp32", "both_roundtrip_vs_fp32",
            "full_vs_both_roundtrip"})
        self.assertGreater(
            bundle["full_vs_fp32"]["root_mean_square_error"],
            bundle["both_roundtrip_vs_fp32"]["root_mean_square_error"])


if __name__ == "__main__":
    unittest.main()
