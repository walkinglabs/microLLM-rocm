#!/usr/bin/env python3
"""Contract tests for the caller-owned HIP Graph matrix."""

import argparse
import importlib.util
import pathlib
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "benchmarks/single_gpu/hip_graph_matrix.py"
SPEC = importlib.util.spec_from_file_location("hip_graph_matrix", MODULE_PATH)
MATRIX = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MATRIX)


class HipGraphMatrixTest(unittest.TestCase):
    def test_positive_csv_rejects_empty_zero_and_text(self):
        self.assertEqual(MATRIX.positive_csv("1,8,32", "--nodes"), [1, 8, 32])
        for value in ("", "0,1", "-1", "one"):
            with self.assertRaises(argparse.ArgumentTypeError):
                MATRIX.positive_csv(value, "--nodes")

    def test_command_preserves_the_exact_case(self):
        args = types.SimpleNamespace(
            binary=pathlib.Path("/tmp/microllm_bench_hip_graph"),
            warmup=5,
            repetitions=20,
        )
        command = MATRIX.command(args, "graph", 128, 4096)
        self.assertEqual(command[command.index("--mode") + 1], "graph")
        self.assertEqual(command[command.index("--nodes") + 1], "128")
        self.assertEqual(command[command.index("--elements") + 1], "4096")

    def test_summary_requires_exact_nodes_transfers_and_large_case_speed(self):
        records = []
        for run in range(1, 4):
            for nodes in (1, 32):
                for mode, wall in (("eager", 2.0), ("graph", 1.0)):
                    records.append({
                        "process_run": run,
                        "mode": mode,
                        "nodes": nodes,
                        "elements": 1,
                        "event_median_ms": wall * 0.9,
                        "wall_median_ms": wall,
                        "wall_p95_ms": wall * 1.1,
                        "setup_ms": 3.0 if mode == "graph" else 0.0,
                        "captured_nodes": nodes + 1 if mode == "graph" else 0,
                        "maximum_absolute_error": 0.0,
                        "host_to_device_calls": 0,
                        "device_to_host_calls": 0,
                        "device_to_device_calls": 0,
                    })
        summary = MATRIX.summarize(records, [1, 32], [1], 3)
        self.assertEqual(
            summary["decision"], "keep caller-owned HIP Graph runtime primitive")
        self.assertTrue(all(summary["gate_results"].values()))
        self.assertEqual(summary["comparisons"][1]["wall_speedup"], 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
