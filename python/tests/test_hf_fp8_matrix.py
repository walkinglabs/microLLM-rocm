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
            "fp8_activation_scale": 0.025, "fp8_weight_scale": 0.005,
            "fp8_weight_scale_mode": "fixed"})()
        model = {"config": "config.json", "weights": "model.bin",
                 "inference": {"token_ids": [1, 2]}}
        command = MATRIX.command(args, model, 8, "fp8", Path("logits.bin"))
        self.assertEqual(command[command.index("--fp8-linear") + 1], "true")
        self.assertEqual(command[command.index("--fp8-activation-scale") + 1],
                         "0.025")
        self.assertEqual(command[command.index("--fp8-weight-scale-mode") + 1],
                         "fixed")
        self.assertNotIn("--bf16-ffn", command)

    def test_command_exposes_tensor_amax_weight_policy(self):
        args = type("Args", (), {
            "binary": Path("micro"), "warmup": 1, "steps": 3,
            "fp8_activation_scale": 0.2, "fp8_weight_scale": 0.005,
            "fp8_weight_scale_mode": "tensor-amax"})()
        model = {"config": "config.json", "weights": "model.bin",
                 "inference": {"token_ids": [1, 2]}}
        command = MATRIX.command(args, model, 8, "fp8", Path("logits.bin"))
        self.assertEqual(command[command.index("--fp8-weight-scale-mode") + 1],
                         "tensor-amax")

    def test_complete_logit_metrics_do_not_replace_preparation_evidence(self):
        comparison = MATRIX.compare_logits([1.0, 2.0], [1.0, 2.0])
        self.assertTrue(comparison["precision_gate_passed"])
        self.assertNotIn("weight_preparation_ms", comparison)

    def test_boundary_names_the_actual_weight_scale_policy(self):
        self.assertIn("static global weight", MATRIX.experiment_boundary("fixed"))
        tensor = MATRIX.experiment_boundary("tensor-amax")
        self.assertIn("per-Tensor weight amax", tensor)
        self.assertIn("fixed global activation", tensor)


if __name__ == "__main__":
    unittest.main()
