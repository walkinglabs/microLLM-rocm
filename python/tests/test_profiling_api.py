import asyncio
import enum
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import microllm.profiling as profiling_module
from microllm.profiling import (calibrate_python_rocprof_clock,
                                export_perfetto, hip_event_profile_scope,
                                merge_rocprof_perfetto, profile, profile_scope)


class ProfilingApiTest(unittest.TestCase):
    class FakeRoctx:
        available = True

        def __init__(self):
            self.pushed = []
            self.pops = 0

        def push(self, name):
            self.pushed.append(name)
            return True

        def pop(self):
            self.pops += 1
            return True

    class FakeDevice(enum.Enum):
        CPU = 0
        HIP = 1

    class FakeHipEvent:
        instances = []

        def __init__(self, device, *, enable_timing=True):
            self.device = (ProfilingApiTest.FakeDevice.HIP, 0)
            self.enable_timing = enable_timing
            self.recorded = False
            self.synchronized = False
            self.closed = False
            self.__class__.instances.append(self)

        def record_default_stream(self):
            self.recorded = True

        def record(self, stream):
            self.recorded = True

        def ready(self):
            return self.synchronized

        def synchronize(self):
            self.synchronized = True

        def elapsed_ms_since(self, start):
            if not start.recorded or not self.recorded:
                raise RuntimeError("events were not recorded")
            return 2.5

        def close(self):
            self.closed = True

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

    def test_optional_roctx_range_records_calibration_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.jsonl"
            fake = self.FakeRoctx()
            with mock.patch.object(profiling_module, "_ROCTX_RUNTIME", fake):
                self.assertTrue(profiling_module.roctx_available())
                with profile_scope("gpu.add", output=output, run_id="run",
                                   emit_roctx=True):
                    pass
            row = json.loads(output.read_text())
            self.assertTrue(row["roctx_requested"])
            self.assertTrue(row["roctx_emitted"])
            self.assertEqual(row["roctx_status"], "emitted")
            self.assertRegex(row["roctx_range_name"],
                             r"^microllm\.python\.[0-9a-f]{32}\.gpu\.add$")
            self.assertLessEqual(row["roctx_push_before_ns"],
                                 row["roctx_push_after_ns"])
            self.assertLessEqual(row["roctx_pop_before_ns"],
                                 row["roctx_pop_after_ns"])
            self.assertEqual(fake.pushed, [row["roctx_range_name"]])
            self.assertEqual(fake.pops, 1)
            self.assertEqual(row["process_id"], profiling_module.os.getpid())
            self.assertGreater(row["native_thread_id"], 0)

    def test_requested_roctx_is_explicitly_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.jsonl"
            unavailable = mock.Mock(available=False)
            with mock.patch.object(
                    profiling_module, "_ROCTX_RUNTIME", unavailable):
                self.assertFalse(profiling_module.roctx_available())
                with profile_scope("cpu", output=output, emit_roctx=True):
                    pass
            row = json.loads(output.read_text())
            self.assertTrue(row["roctx_requested"])
            self.assertFalse(row["roctx_emitted"])
            self.assertEqual(row["roctx_status"], "unavailable")
            unavailable.push.assert_not_called()
            unavailable.pop.assert_not_called()

    def test_hip_event_scope_observes_completion_on_background_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "event.jsonl"
            self.FakeHipEvent.instances = []
            fake_capi = types.SimpleNamespace(
                Device=self.FakeDevice, Event=self.FakeHipEvent)
            with mock.patch.dict(sys.modules, {"microllm._capi": fake_capi}):
                with hip_event_profile_scope(
                        "async.add", output=output, run_id="run",
                        metadata={"elements": 1024},
                        stream=object()) as completion:
                    pass
                self.assertFalse(completion.ready())
                with self.assertRaisesRegex(RuntimeError, "wait for HIP Event"):
                    completion.close()
                future = completion.observe_async()
                record = future.result(timeout=1.0)
                self.assertTrue(completion.ready())
                self.assertEqual(completion.wait(), record)
                completion.close()
                completion.close()
            self.assertEqual(len(output.read_text().splitlines()), 1)
            self.assertEqual(record["kind"], "hip_event_completion_span")
            self.assertEqual(record["device_elapsed_ns"], 2_500_000)
            self.assertFalse(record["event_ready_at_submit"])
            self.assertEqual(record["synchronization_scope"],
                             "hip_event_explicit_stream")
            self.assertEqual(record["metadata"], {"elements": 1024})
            self.assertTrue(all(event.closed
                                for event in self.FakeHipEvent.instances))
            perfetto = Path(directory) / "event.json"
            export_perfetto(output, perfetto)
            event = json.loads(perfetto.read_text())["traceEvents"][0]
            self.assertEqual(event["args"]["device_elapsed_ns"], 2_500_000)
            self.assertEqual(event["args"]["submission_duration_ns"],
                             record["submission_duration_ns"])

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
            hip_api = root / "hip-api.csv"
            marker.write_text(
                '"Function","Process_Id","Thread_Id","Correlation_Id",'
                '"Start_Timestamp","End_Timestamp"\n'
                '"span",7,8,2,1000,2000\n', encoding="utf-8")
            kernel.write_text(
                '"Agent_Id","Queue_Id","Kernel_Name","Correlation_Id",'
                '"Start_Timestamp","End_Timestamp"\n'
                '"Agent 2",1,"add",2,2010,2100\n'
                '"Agent 2",1,"copy",3,2110,2150\n', encoding="utf-8")
            hip_api.write_text(
                '"Domain","Function","Process_Id","Thread_Id","Correlation_Id",'
                '"Start_Timestamp","End_Timestamp"\n'
                '"HIP_RUNTIME_API","hipLaunchKernel",7,8,2,1200,1250\n'
                '"HIP_RUNTIME_API","hipMemcpy",7,8,3,1300,1350\n',
                encoding="utf-8")
            output = root / "merged.json"
            report = merge_rocprof_perfetto(
                marker, kernel, output, hip_api_csv=hip_api)
            self.assertEqual(report["correlated_ids"], 2)
            self.assertEqual(report["correlated_pairs"], 2)
            self.assertEqual(report["trace_events"], 7)
            events = json.loads(output.read_text())["traceEvents"]
            self.assertEqual([event["ph"] for event in events].count("s"), 2)
            self.assertEqual([event["ph"] for event in events].count("f"), 2)
            flows = [event["id"] for event in events if event["ph"] == "s"]
            self.assertEqual(len(flows), len(set(flows)))

    def test_python_clock_calibration_and_three_way_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.jsonl"
            marker = root / "marker.csv"
            kernel = root / "kernel.csv"
            hip_api = root / "hip-api.csv"
            rows = [
                {
                    "schema_version": 1, "record_type": "python_profile_span",
                    "name": "span-a", "phase": "python", "status": "pass",
                    "depth": 0, "run_id": "run", "span_id": "a",
                    "start_ns": 800, "duration_ns": 1400,
                    "process_id": 7, "thread_id": 80, "native_thread_id": 8,
                    "metadata": {}, "roctx_emitted": True,
                    "roctx_range_name": "range-a",
                    "roctx_push_before_ns": 900, "roctx_push_after_ns": 1100,
                    "roctx_pop_before_ns": 1900, "roctx_pop_after_ns": 2100,
                },
                {
                    "schema_version": 1, "record_type": "python_profile_span",
                    "name": "span-b", "phase": "python", "status": "pass",
                    "depth": 0, "run_id": "run", "span_id": "b",
                    "start_ns": 2800, "duration_ns": 2400,
                    "process_id": 7, "thread_id": 80, "native_thread_id": 8,
                    "metadata": {}, "roctx_emitted": True,
                    "roctx_range_name": "range-b",
                    "roctx_push_before_ns": 2900, "roctx_push_after_ns": 3100,
                    "roctx_pop_before_ns": 4900, "roctx_pop_after_ns": 5100,
                },
                {
                    "schema_version": 1, "record_type": "python_profile_span",
                    "name": "unmarked", "phase": "python", "status": "pass",
                    "depth": 1, "run_id": "run", "span_id": "c",
                    "start_ns": 2200, "duration_ns": 200,
                    "process_id": 7, "thread_id": 80, "native_thread_id": 8,
                    "metadata": {"batch": 2}, "roctx_emitted": False,
                },
            ]
            profile_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8")
            marker.write_text(
                '"Domain","Function","Process_Id","Thread_Id","Correlation_Id",'
                '"Start_Timestamp","End_Timestamp"\n'
                '"MARKER_CORE_RANGE_API","range-a",7,8,2,10000,11000\n'
                '"MARKER_CORE_RANGE_API","range-b",7,8,3,12000,14000\n',
                encoding="utf-8")
            kernel.write_text(
                '"Agent_Id","Queue_Id","Kernel_Name","Correlation_Id",'
                '"Start_Timestamp","End_Timestamp"\n'
                '"Agent 2",1,"add",2,10050,10090\n', encoding="utf-8")
            hip_api.write_text(
                '"Domain","Function","Process_Id","Thread_Id","Correlation_Id",'
                '"Start_Timestamp","End_Timestamp"\n'
                '"HIP_RUNTIME_API","hipLaunchKernel",7,8,2,10020,10040\n',
                encoding="utf-8")
            calibration_path = root / "calibration.json"
            calibration = calibrate_python_rocprof_clock(
                profile_path, marker, calibration_path)
            self.assertEqual(calibration["matched_spans"], 2)
            self.assertEqual(calibration["boundary_points"], 4)
            self.assertAlmostEqual(calibration["scale"], 1.0)
            self.assertEqual(calibration["max_abs_residual_ns"], 0.0)
            self.assertTrue(calibration_path.is_file())
            one_span = root / "one-span.jsonl"
            one_span.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least two"):
                calibrate_python_rocprof_clock(one_span, marker)

            output = root / "three-way.json"
            report = merge_rocprof_perfetto(
                marker, kernel, output, hip_api_csv=hip_api,
                python_jsonl=profile_path)
            self.assertEqual(report["python_events"], 3)
            self.assertEqual(report["trace_events"], 8)
            document = json.loads(output.read_text())
            self.assertEqual(
                document["metadata"]["python_clock_calibration"]["matched_spans"], 2)
            python_events = [event for event in document["traceEvents"]
                             if event["cat"] == "python"]
            self.assertEqual({event["name"] for event in python_events},
                             {"span-a", "span-b", "unmarked"})
            self.assertTrue(all(event["pid"] == 7 and event["tid"] == 8
                                for event in python_events))


if __name__ == "__main__":
    unittest.main()
