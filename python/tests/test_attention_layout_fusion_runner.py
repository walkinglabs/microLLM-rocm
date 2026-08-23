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


if __name__ == "__main__":
    unittest.main(verbosity=2)
