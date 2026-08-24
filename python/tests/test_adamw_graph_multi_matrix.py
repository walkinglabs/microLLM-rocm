#!/usr/bin/env python3
"""Contract test for stable-descriptor AdamW multi-tensor Graph results."""

from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "benchmarks/single_gpu/adamw_graph_multi_matrix.py"
SPEC = importlib.util.spec_from_file_location("adamw_graph_multi_matrix", MODULE_PATH)
MATRIX = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MATRIX)


class AdamWGraphMultiMatrixTest(unittest.TestCase):
    def test_expected_nodes_distinguish_all_three_paths(self):
        self.assertEqual(MATRIX.expected_nodes("eager", 64), 0)
        self.assertEqual(MATRIX.expected_nodes("graph", 64), 65)
        self.assertEqual(MATRIX.expected_nodes("graph-multi", 64), 2)

    def test_summary_requires_state_alignment_and_exposes_large_failure(self):
        records = []
        for run in range(1, 4):
            for precision in MATRIX.PRECISIONS:
                for tensors, elements in ((64, 1024), (16, 262144)):
                    for mode, wall in (
                            ("eager", 2.0), ("graph", 2.5),
                            ("graph-multi", 1.0)):
                        if elements > 1024 and mode == "graph-multi":
                            wall = 2.2
                        records.append({
                            "process_run": run, "precision": precision,
                            "mode": mode, "tensors": tensors,
                            "elements": elements, "warmup": 1,
                            "repetitions": 3, "final_step": 4,
                            "captured_nodes": MATRIX.expected_nodes(mode, tensors),
                            "event_ms_per_step": wall,
                            "wall_ms_per_step": wall,
                            "preparation_ms": 0.2 if mode == "graph-multi" else 0.0,
                            "setup_ms": 0.1 if mode != "eager" else 0.0,
                            "timed_host_to_device_calls": 0,
                            "timed_device_to_host_calls": 0,
                            "timed_device_to_device_calls": 0,
                            "parameter_sample": [1.0],
                            "first_moment_sample": [0.1],
                            "second_moment_sample": [0.01],
                            "mirror_sample": [1.0],
                        })
        summary = MATRIX.summarize(
            records, [(64, 1024), (16, 262144)], 3)
        self.assertEqual(summary["status"], "pass")
        self.assertTrue(summary["gates"]["bf16_many_small_rescued"])
        self.assertFalse(summary["gates"]["large_tensor_rescued"])
        self.assertEqual(
            summary["decision"],
            "keep explicit two-node multi-tensor Graph candidate")


if __name__ == "__main__":
    unittest.main(verbosity=2)
