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
            "fp8_activation_minimum_scale": 0.0001,
            "fp8_weight_scale_mode": "fixed",
            "fp8_activation_scale_mode": "fixed",
            "fp8_diagnostic_mode": "full",
            "fp8_fp32_layers": ""})()
        model = {"config": "config.json", "weights": "model.bin",
                 "inference": {"token_ids": [1, 2]}}
        command = MATRIX.command(args, model, 8, "fp8", Path("logits.bin"))
        self.assertEqual(command[command.index("--fp8-linear") + 1], "true")
        self.assertEqual(command[command.index("--fp8-activation-scale") + 1],
                         "0.025")
        self.assertEqual(
            command[command.index("--fp8-activation-minimum-scale") + 1],
            "0.0001")
        self.assertEqual(command[command.index("--fp8-weight-scale-mode") + 1],
                         "fixed")
        self.assertNotIn("--bf16-ffn", command)

    def test_command_exposes_tensor_amax_weight_policy(self):
        args = type("Args", (), {
            "binary": Path("micro"), "warmup": 1, "steps": 3,
            "fp8_activation_scale": 0.2, "fp8_weight_scale": 0.005,
            "fp8_activation_minimum_scale": 0.0001,
            "fp8_weight_scale_mode": "tensor-amax",
            "fp8_activation_scale_mode": "tensor-amax",
            "fp8_diagnostic_mode": "weight-only",
            "fp8_fp32_layers": "21"})()
        model = {"config": "config.json", "weights": "model.bin",
                 "inference": {"token_ids": [1, 2]}}
        command = MATRIX.command(args, model, 8, "fp8", Path("logits.bin"))
        self.assertEqual(command[command.index("--fp8-weight-scale-mode") + 1],
                         "tensor-amax")
        self.assertEqual(command[command.index("--fp8-activation-scale-mode") + 1],
                         "tensor-amax")
        self.assertEqual(command[command.index("--fp8-fp32-layers") + 1], "21")
        self.assertEqual(command[command.index("--fp8-diagnostic-mode") + 1],
                         "weight-only")

    def test_complete_logit_metrics_do_not_replace_preparation_evidence(self):
        comparison = MATRIX.compare_logits([1.0, 2.0], [1.0, 2.0])
        self.assertTrue(comparison["precision_gate_passed"])
        self.assertNotIn("weight_preparation_ms", comparison)

    def test_runner_source_keeps_weight_and_scale_residency_separate(self):
        source = (ROOT / "benchmarks/single_gpu/hf_fp8_matrix.py").read_text()
        self.assertIn('"fp8_weight_bytes_retained"', source)
        self.assertIn('"fp8_scale_bytes_retained"', source)

    def test_boundary_names_the_actual_weight_scale_policy(self):
        self.assertIn("static global weight", MATRIX.experiment_boundary("fixed"))
        tensor = MATRIX.experiment_boundary("tensor-amax")
        self.assertIn("per-Tensor weight amax", tensor)
        self.assertIn("fixed global activation", tensor)
        dynamic = MATRIX.experiment_boundary("tensor-amax", "tensor-amax")
        self.assertIn("device per-input-Tensor activation amax", dynamic)
        ffn = MATRIX.experiment_boundary("tensor-amax", "ffn-outer-row")
        self.assertIn("FFN-only outer-row activation scales", ffn)
        device = MATRIX.experiment_boundary("device-tensor-amax", "fixed")
        self.assertIn("device per-Tensor weight amax", device)
        columns = MATRIX.experiment_boundary("output-channel-amax", "tensor-amax")
        self.assertIn("device per-output-channel weight amax", columns)
        activation_only = MATRIX.experiment_boundary(
            "device-tensor-amax", "tensor-amax", "activation-only")
        self.assertIn("diagnostic mode=activation-only", activation_only)
        both = MATRIX.experiment_boundary(
            "device-tensor-amax", "tensor-amax", "both-roundtrip")
        self.assertIn("diagnostic mode=both-roundtrip", both)
        attention = MATRIX.experiment_boundary(
            "output-channel-amax", "tensor-amax", "full", "attention-only")
        self.assertIn("weight scale scope=attention-only", attention)
        output = MATRIX.experiment_boundary(
            "output-channel-amax", "tensor-amax", "full",
            "attention-output-only")
        self.assertIn("weight scale scope=attention-output-only", output)


if __name__ == "__main__":
    unittest.main()
