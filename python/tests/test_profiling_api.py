import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from microllm.profiling import (export_perfetto, merge_rocprof_perfetto,
                                profile, profile_scope)


class ProfilingApiTest(unittest.TestCase):
    def test_decorator_scope_error_and_async_records(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.jsonl"

            @profile(output=output, name="decorated", run_id="fixed",
                     metadata={"batch": 2})
            def decorated(value):
                with profile_scope("nested", output=output, run_id="fixed"):
                    return value + 1

            @profile(output=output, name="failure", run_id="fixed")
            def failure():
                raise ValueError("expected")

            @profile(output=output, name="async", run_id="fixed")
            async def async_call():
                return 7

            self.assertEqual(decorated(4), 5)
            with self.assertRaisesRegex(ValueError, "expected"):
                failure()
            self.assertEqual(asyncio.run(async_call()), 7)

            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            by_name = {row["name"]: row for row in rows}
            self.assertEqual(by_name["nested"]["depth"], 1)
            self.assertEqual(by_name["decorated"]["depth"], 0)
            self.assertEqual(by_name["decorated"]["metadata"], {"batch": 2})
            self.assertEqual(by_name["failure"]["status"], "error")
            self.assertEqual(by_name["failure"]["exception_type"], "ValueError")
            self.assertEqual(by_name["async"]["status"], "pass")
            for row in rows:
                self.assertEqual(row["schema_version"], 1)
                self.assertEqual(row["framework"], "microllm-python")
                self.assertGreaterEqual(row["duration_ns"], 0)
                self.assertIsInstance(row["thread_id"], int)

            perfetto = Path(directory) / "trace.json"
            report = export_perfetto(output, perfetto)
            self.assertEqual(report["events"], 4)
            document = json.loads(perfetto.read_text())
            self.assertEqual(len(document["traceEvents"]), 4)
            self.assertTrue(all(event["ph"] == "X"
                                for event in document["traceEvents"]))
            self.assertEqual(document["metadata"]["source"], "microllm-python")

    def test_invalid_identity_and_metadata_fail_before_call(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.jsonl"
            with self.assertRaises(ValueError):
                profile_scope("", output=output)
            with self.assertRaises(ValueError):
                profile_scope("x", output=output, metadata={"bad": float("nan")})
            output.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                export_perfetto(output, Path(directory) / "empty.json")

    def test_rocprof_marker_kernel_merge_uses_correlation_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "marker.csv"
            kernel = root / "kernel.csv"
            marker.write_text(
                '"Function","Process_Id","Thread_Id","Correlation_Id",'
                '"Start_Timestamp","End_Timestamp"\n'
                '"span",7,8,2,1000,2000\n', encoding="utf-8")
            kernel.write_text(
                '"Agent_Id","Queue_Id","Kernel_Name","Correlation_Id",'
                '"Start_Timestamp","End_Timestamp"\n'
                '"Agent 2",1,"add",2,2010,2100\n', encoding="utf-8")
            output = root / "merged.json"
            report = merge_rocprof_perfetto(marker, kernel, output)
            self.assertEqual(report["correlated_ids"], 1)
            self.assertEqual(report["trace_events"], 4)
            events = json.loads(output.read_text())["traceEvents"]
            self.assertEqual([event["ph"] for event in events].count("s"), 1)
            self.assertEqual([event["ph"] for event in events].count("f"), 1)


if __name__ == "__main__":
    unittest.main()
