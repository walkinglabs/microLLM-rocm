#!/usr/bin/env python3
"""Contract test for the optimizer Graph official-shape model gate."""

from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "benchmarks/single_gpu/optimizer_graph_model_matrix.py"
SPEC = importlib.util.spec_from_file_location(
    "optimizer_graph_model_matrix", MODULE_PATH)
MATRIX = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MATRIX)


class OptimizerGraphModelMatrixTest(unittest.TestCase):
    def test_exact_but_slower_graph_is_rejected(self):
        records = []
        for run in range(1, 4):
            for model, context in MATRIX.CASES:
                for mode, optimizer_ms, step_ms in (
                        ("eager", 4.0, 20.0), ("graph", 5.0, 21.0)):
                    records.append({
                        "process_run": run, "model": model,
                        "context": context, "mode": mode,
                        "mean_optimizer_ms": optimizer_ms,
                        "mean_step_ms": step_ms, "preparation_ms": 2.0,
                        "optimizer_host_to_device_calls": 0 if mode == "graph" else 2,
                        "captured_nodes": 2 if mode == "graph" else 0,
                        "gradient_snapshot_matches": True,
                        "losses": [1.0, 0.9], "observed_parameter": 0.5,
                    })
            records.append({
                "process_run": run, "model": "deepseek", "context": 512,
                "mode": "preflight", "gradient_snapshot_matches": False,
                "graph_launched": False,
            })
        summary = MATRIX.summarize(records, 3, 2)
        self.assertEqual(summary["status"], "pass")
        self.assertTrue(summary["gates"]["loss_and_parameter_exact"])
        self.assertFalse(summary["gates"]["optimizer_speedup_at_least_1_01"])
        self.assertEqual(
            summary["decision"],
            "reject model optimizer Graph route; close optimizer-only Graph track")


if __name__ == "__main__":
    unittest.main(verbosity=2)
