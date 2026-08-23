#!/usr/bin/env python3
"""Contract tests for the official Attention layout-fusion A/B runner."""

import importlib.util
import pathlib
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "benchmarks/single_gpu/compare_attention_layout_fusion.py"
SPEC = importlib.util.spec_from_file_location("attention_layout_runner", MODULE_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)
MATRIX_PATH = ROOT / "benchmarks/single_gpu/attention_layout_matrix.py"
MATRIX_SPEC = importlib.util.spec_from_file_location(
    "attention_layout_matrix", MATRIX_PATH)
MATRIX = importlib.util.module_from_spec(MATRIX_SPEC)
assert MATRIX_SPEC.loader is not None
MATRIX_SPEC.loader.exec_module(MATRIX)


class AttentionLayoutFusionRunnerTest(unittest.TestCase):
    def setUp(self):
        self.model = {
            "name": "fixture",
            "config": "/fixture/config.json",
            "weights": "/fixture/model.safetensors",
            "training": {"tokens": "7,9", "learning_rate": 1.0e-5},
        }
        self.args = types.SimpleNamespace(
            binary=pathlib.Path("/fixture/microllm_hf_train_step"),
            context=4,
            warmup=1,
            steps=2,
        )

    def test_training_context_supplies_one_extra_target_token(self):
        tokens = RUNNER.expanded_tokens(self.model, self.args.context).split(",")
        self.assertEqual(tokens, ["7", "9", "7", "9", "7"])
        self.assertEqual(len(tokens) - 1, self.args.context)

    def test_same_binary_policy_changes_only_explicit_layout_switch(self):
        materialized = RUNNER.command(self.args, self.model, False)
        fused = RUNNER.command(self.args, self.model, True)
        flag = materialized.index("--attention-rope-layout-fusion")
        self.assertEqual(materialized[flag + 1], "false")
        self.assertEqual(fused[flag + 1], "true")
        self.assertEqual(materialized[:flag], fused[:flag])
        self.assertEqual(materialized[flag + 2:], fused[flag + 2:])
        self.assertIn("--tied-embedding-sparse-add", fused)
        self.assertIn("--bf16-weight-mirrors", fused)

    def test_diagnostics_are_separate_single_step_runs(self):
        destination = pathlib.Path("/fixture/diagnostics.json")
        command = RUNNER.command(self.args, self.model, True, destination)
        self.assertEqual(command[command.index("--warmup") + 1], "0")
        self.assertEqual(command[command.index("--steps") + 1], "1")
        self.assertEqual(
            command[command.index("--diagnostics-output") + 1], str(destination))

    def test_operator_matrix_preserves_batch_head_sequence_width(self):
        shape = MATRIX.parse_shape("qwen:2:14:512:64")
        self.assertEqual(
            shape,
            {"name": "qwen", "batch": 2, "heads": 14,
             "sequence": 512, "width": 64},
        )
        matrix_args = types.SimpleNamespace(
            binary=pathlib.Path("/fixture/microllm_bench_attention_layout"),
            warmup=3,
            repetitions=20,
        )
        command = MATRIX.command(matrix_args, shape, "interleaved")
        self.assertEqual(command[command.index("--batch") + 1], "2")
        self.assertEqual(command[command.index("--heads") + 1], "14")
        self.assertEqual(command[command.index("--sequence") + 1], "512")
        self.assertEqual(command[command.index("--width") + 1], "64")


if __name__ == "__main__":
    unittest.main(verbosity=2)
