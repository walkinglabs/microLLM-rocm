#!/usr/bin/env python3
"""Contract test for explicit quiescent allocator handoff results."""

from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "benchmarks/single_gpu/quiescent_allocator_handoff_matrix.py"
SPEC = importlib.util.spec_from_file_location(
    "quiescent_allocator_handoff_matrix", MODULE_PATH)
HANDOFF = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(HANDOFF)


class QuiescentAllocatorHandoffMatrixTest(unittest.TestCase):
    def test_three_cases_are_rescued_and_deep_long_remains_rejected(self):
        records = []
        for run in range(1, 4):
            for model, context in HANDOFF.CASES:
                for handoff in HANDOFF.POLICIES:
                    matches = handoff and not (
                        model == "deepseek" and context == 512)
                    records.append({
                        "process_run": run, "model": model,
                        "context": context, "quiescent_handoff": handoff,
                        "gradient_snapshot_matches": matches,
                        "caching_allocator_enabled": handoff,
                        "quiescent_handoff_count": 3 if handoff else 0,
                        "graph_launched": False, "preparation_ms": 2.0,
                    })
        summary = HANDOFF.summarize(records, 3)
        self.assertEqual(summary["status"], "pass")
        self.assertTrue(all(summary["gates"].values()))
        self.assertEqual(sum(row["rescued"]
                             for row in summary["comparisons"]), 3)
        self.assertIn("retain DeepSeek T512 rejection", summary["decision"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
