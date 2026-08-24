#!/usr/bin/env python3
"""Contract tests for the caller-owned hipBLASLt Graph matrix."""

import argparse
import importlib.util
import pathlib
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "benchmarks/single_gpu/hip_graph_gemm_matrix.py"
SPEC = importlib.util.spec_from_file_location("hip_graph_gemm_matrix", MODULE_PATH)
MATRIX = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MATRIX)


class HipGraphGemmMatrixTest(unittest.TestCase):
    def test_shape_parser_is_named_positive_and_exact(self):
        self.assertEqual(
            MATRIX.parse_shape("qwen:512:896:896"),
            {"name": "qwen", "rows": 512, "inner": 896, "columns": 896})
        for value in ("qwen:1:2", ":1:2:3", "qwen:0:2:3", "qwen:x:2:3"):
            with self.assertRaises(argparse.ArgumentTypeError):
                MATRIX.parse_shape(value)

    def test_command_preserves_shape_calls_and_measurement(self):
        args = types.SimpleNamespace(
            binary=pathlib.Path("/tmp/microllm_bench_hip_graph_gemm"),
            warmup=3,
            repetitions=10,
        )
        shape = MATRIX.parse_shape("deep:512:1536:1536")
        command = MATRIX.command(args, "graph", 8, shape)
        self.assertEqual(command[command.index("--mode") + 1], "graph")
        self.assertEqual(command[command.index("--calls") + 1], "8")
        self.assertEqual(command[command.index("--inner") + 1], "1536")

    def test_summary_keeps_exact_repeated_vendor_graph(self):
        shape = MATRIX.parse_shape("fixture:4:4:4")
        records = []
        for run in range(1, 4):
            for calls in (1, 8):
                for mode, wall in (("eager", 2.0), ("graph", 1.0)):
                    records.append({
                        "process_run": run,
                        "shape_name": "fixture",
                        "mode": mode,
                        "calls": calls,
                        "rows": 4,
                        "inner": 4,
                        "columns": 4,
                        "event_median_ms": wall * 0.9,
                        "wall_median_ms": wall,
                        "wall_p95_ms": wall * 1.1,
                        "setup_ms": 1.0 if mode == "graph" else 0.0,
                        "captured_nodes": calls if mode == "graph" else 0,
                        "maximum_absolute_error": 0.0,
                        "rms_error": 0.0,
                        "output_address_stable": True,
                        "host_to_device_calls": 0,
                        "device_to_host_calls": 0,
                        "device_to_device_calls": 0,
                    })
        summary = MATRIX.summarize(records, [shape], [1, 8], 3)
        self.assertTrue(all(summary["gate_results"].values()))
        self.assertEqual(
            summary["decision"], "keep caller-owned hipBLASLt Graph boundary")


if __name__ == "__main__":
    unittest.main(verbosity=2)
