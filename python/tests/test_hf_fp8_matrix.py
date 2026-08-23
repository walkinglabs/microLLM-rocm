import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "hf_fp8_matrix", ROOT / "benchmarks/single_gpu/hf_fp8_matrix.py")
MATRIX = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MATRIX)


class HfFp8MatrixTest(unittest.TestCase):
    def test_complete_logit_metrics_and_gate(self):
        exact = MATRIX.compare_logits([1.0, 2.0], [1.0, 2.0])
        self.assertTrue(exact["precision_gate_passed"])
        failed = MATRIX.compare_logits([1.0, 3.0], [1.0, 2.0])
        self.assertFalse(failed["precision_gate_passed"])
        self.assertEqual(failed["top_token"], failed["reference_top_token"])

    def test_policy_orders_rotate_framework_position(self):
        self.assertEqual(len(MATRIX.POLICY_ORDERS), 3)
        for policy in ("fp32", "bf16", "fp8"):
            self.assertEqual(
                sorted(order.index(policy) for order in MATRIX.POLICY_ORDERS),
                [0, 1, 2])

    def test_command_keeps_fp8_scales_explicit(self):
        args = type("Args", (), {
            "binary": Path("micro"), "warmup": 1, "steps": 3,
            "fp8_activation_scale": 0.025, "fp8_weight_scale": 0.005})()
        model = {"config": "config.json", "weights": "model.bin",
                 "inference": {"token_ids": [1, 2]}}
        command = MATRIX.command(args, model, 8, "fp8", Path("logits.bin"))
        self.assertEqual(command[command.index("--fp8-linear") + 1], "true")
        self.assertEqual(command[command.index("--fp8-activation-scale") + 1],
                         "0.025")
        self.assertNotIn("--bf16-ffn", command)


if __name__ == "__main__":
    unittest.main()
