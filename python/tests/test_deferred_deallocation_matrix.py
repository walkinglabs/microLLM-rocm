#!/usr/bin/env python3
"""Contract tests for deferred HIP deallocation matrix."""

import importlib.util
import pathlib
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "benchmarks/single_gpu/deferred_deallocation_matrix.py"
SPEC = importlib.util.spec_from_file_location("deferred_deallocation_matrix", MODULE_PATH)
MATRIX = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MATRIX)


class DeferredDeallocationMatrixTest(unittest.TestCase):
    def test_command_preserves_mode_shape_and_measurement(self):
        args = types.SimpleNamespace(
            binary=pathlib.Path("/tmp/microllm_bench_deferred_deallocation"),
            warmup=3,
            repetitions=10,
        )
        command = MATRIX.command(args, "deferred", 32, 4096)
        self.assertEqual(command[command.index("--mode") + 1], "deferred")
        self.assertEqual(command[command.index("--nodes") + 1], "32")
        self.assertEqual(command[command.index("--elements") + 1], "4096")

    def test_summary_requires_exact_lifetime_bytes_and_speed(self):
        records = []
        for run in range(1, 4):
            for mode, wall in (("immediate_sync", 3.0), ("deferred", 1.0)):
                records.append({
                    "process_run": run,
                    "mode": mode,
                    "nodes": 8,
                    "elements": 4,
                    "wall_median_ms": wall,
                    "wall_p95_ms": wall * 1.1,
                    "deferred_blocks": 7 if mode == "deferred" else 0,
                    "deferred_bytes": 112 if mode == "deferred" else 0,
                    "overflow_flushes": 0,
                    "maximum_absolute_error": 0.0,
                    "host_to_device_calls": 0,
                    "device_to_host_calls": 0,
                    "device_to_device_calls": 0,
                })
        summary = MATRIX.summarize(records, [8], [4], 3)
        self.assertTrue(all(summary["gate_results"].values()))
        self.assertEqual(
            summary["decision"], "keep explicit deferred HIP deallocation scope")


if __name__ == "__main__":
    unittest.main(verbosity=2)
