#!/usr/bin/env python3
"""Contract test for optimizer Graph model preflight rejection."""

from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "benchmarks/single_gpu/optimizer_graph_model_preflight.py"
SPEC = importlib.util.spec_from_file_location(
    "optimizer_graph_model_preflight", MODULE_PATH)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREFLIGHT)


class OptimizerGraphModelPreflightTest(unittest.TestCase):
    def test_all_failed_snapshots_are_a_successful_safety_result(self):
        records = []
        for run in range(1, 4):
            for model, context in PREFLIGHT.CASES:
                records.append({
                    "process_run": run, "model": model,
                    "context": context, "gradient_snapshot_matches": False,
                    "caching_allocator_enabled": False,
                    "graph_launched": False, "captured_nodes": 0,
                    "preparation_ms": 2.0,
                })
        summary = PREFLIGHT.summarize(records, 3)
        self.assertEqual(summary["status"], "pass")
        self.assertTrue(all(summary["gates"].values()))
        self.assertEqual(sum(row["graph_launches"]
                             for row in summary["comparisons"]), 0)
        self.assertIn("Stream-aware retirement", summary["decision"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
