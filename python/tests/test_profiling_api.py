import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from microllm.profiling import profile, profile_scope


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

    def test_invalid_identity_and_metadata_fail_before_call(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.jsonl"
            with self.assertRaises(ValueError):
                profile_scope("", output=output)
            with self.assertRaises(ValueError):
                profile_scope("x", output=output, metadata={"bad": float("nan")})


if __name__ == "__main__":
    unittest.main()
