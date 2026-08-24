#!/usr/bin/env python3
"""Contract test for the device-owned AdamW Graph matrix."""

from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "benchmarks/single_gpu/adamw_graph_replay_matrix.py"
SPEC = importlib.util.spec_from_file_location("adamw_graph_replay_matrix", MODULE_PATH)
MATRIX = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MATRIX)


class AdamWGraphReplayMatrixTest(unittest.TestCase):
    def test_case_parser_rejects_duplicates_and_invalid_values(self):
        self.assertEqual(MATRIX.cases("1x2,3x4"), [(1, 2), (3, 4)])
        for value in ("", "0x1", "1x-2", "1x2,1x2", "bad"):
            with self.assertRaises(Exception):
                MATRIX.cases(value)

    def test_summary_keeps_primitive_and_rejects_large_universal_route(self):
        records = []
        for run in range(1, 4):
            for precision in MATRIX.PRECISIONS:
                for tensors, elements in ((64, 1024), (16, 262144)):
                    for mode, wall in (("eager", 2.0), ("graph", 1.0)):
                        if elements > 1024 and mode == "graph":
                            wall = 2.5
                        records.append({
                            "process_run": run,
                            "precision": precision,
                            "mode": mode,
                            "tensors": tensors,
                            "elements": elements,
                            "warmup": 1,
                            "repetitions": 3,
                            "final_step": 4,
                            "captured_nodes": tensors + 1 if mode == "graph" else 0,
                            "event_ms_per_step": wall,
                            "wall_ms_per_step": wall,
                            "setup_ms": 1.0 if mode == "graph" else 0.0,
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
        self.assertTrue(summary["gates"][
            "fp32_many_small_tensors_wall_speedup_at_least_1_05"])
        self.assertTrue(summary["gates"][
            "bf16_many_small_tensors_wall_speedup_at_least_1_05"])
        self.assertFalse(summary["gates"]["large_tensor_universal_policy"])
        self.assertIn("reject universal and BF16", summary["decision"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
