import json
import pathlib
import sys
import tempfile
import unittest


TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools" / "alignment"
sys.path.insert(0, str(TOOLS))

from common import load_jsonl, value_statistics, write_jsonl  # noqa: E402
from compare import compare_timings, compare_values  # noqa: E402


def value_record(framework, values, *, name="x", shape=None):
    shape = shape or [len(values)]
    return {
        "schema_version": 1,
        "framework": framework,
        "run_id": "unit",
        "phase": "values",
        "sequence": 0,
        "iteration": 0,
        "kind": "operator",
        "name": name,
        "shape": shape,
        "dtype": "float32",
        "device": "cpu",
        "wall_ms": 0.0,
        "statistics": value_statistics(values, len(values)),
        "values_truncated": False,
        "values": values,
    }


def timing_record(framework, iteration, duration, *, name="x"):
    record = value_record(framework, [], name=name, shape=[1])
    record.update(phase="operator_timing", iteration=iteration, wall_ms=duration)
    return record


class AlignmentToolsTest(unittest.TestCase):
    def test_jsonl_round_trip_requires_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "trace.jsonl"
            expected = [value_record("microllm", [1.0, 2.0])]
            write_jsonl(path, expected)
            self.assertEqual(load_jsonl(path), expected)
            path.write_text(json.dumps({"schema_version": 99}) + "\n")
            with self.assertRaises(ValueError):
                load_jsonl(path)

    def test_value_comparison_reports_pass_and_exact_failure_location(self):
        passed = compare_values(
            [value_record("microllm", [1.0, 2.000001])],
            [value_record("pytorch", [1.0, 2.0])],
            1.0e-5,
            1.0e-5,
            False,
        )
        self.assertEqual(passed[0]["status"], "pass")
        failed = compare_values(
            [value_record("microllm", [1.0, 3.0])],
            [value_record("pytorch", [1.0, 2.0])],
            1.0e-5,
            1.0e-5,
            False,
        )
        self.assertEqual(failed[0]["status"], "numeric_mismatch")
        self.assertEqual(failed[0]["maximum_error_index"], 1)
        self.assertEqual(failed[0]["max_abs"], 1.0)

    def test_comparison_rejects_shape_and_truncated_value_claims(self):
        shape = compare_values(
            [value_record("microllm", [1.0, 2.0], shape=[2])],
            [value_record("pytorch", [1.0, 2.0], shape=[1, 2])],
            0.0,
            0.0,
            False,
        )
        self.assertEqual(shape[0]["status"], "metadata_mismatch")
        truncated = value_record("microllm", [1.0])
        truncated["values_truncated"] = True
        result = compare_values(
            [truncated], [value_record("pytorch", [1.0])], 0.0, 0.0, False
        )
        self.assertEqual(result[0]["status"], "truncated")

    def test_non_finite_values_must_match_by_kind(self):
        same_left = value_record("microllm", [0.0, 0.0])
        same_left["values"] = ["nan", "inf"]
        same_right = value_record("pytorch", [0.0, 0.0])
        same_right["values"] = ["nan", "inf"]
        same = compare_values(
            [same_left], [same_right],
            0.0, 0.0, False,
        )
        self.assertEqual(same[0]["status"], "pass")
        different_left = value_record("microllm", [0.0])
        different_left["values"] = ["nan"]
        different_right = value_record("pytorch", [0.0])
        different_right["values"] = ["inf"]
        different = compare_values(
            [different_left], [different_right],
            0.0, 0.0, False,
        )
        self.assertEqual(different[0]["status"], "numeric_mismatch")

    def test_timing_comparison_aggregates_repetitions_and_speed_ratio(self):
        micro = [timing_record("microllm", index, value)
                 for index, value in enumerate((2.0, 1.0, 3.0))]
        torch = [timing_record("pytorch", index, value)
                 for index, value in enumerate((4.0, 2.0, 6.0))]
        result = compare_timings(micro, torch)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["microllm"]["median_ms"], 2.0)
        self.assertEqual(result[0]["pytorch"]["median_ms"], 4.0)
        self.assertEqual(result[0]["pytorch_over_microllm"], 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
