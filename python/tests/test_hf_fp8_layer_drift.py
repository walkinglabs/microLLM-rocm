import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks/single_gpu"
sys.path.insert(0, str(BENCHMARKS))
SPEC = importlib.util.spec_from_file_location(
    "hf_fp8_layer_drift", BENCHMARKS / "hf_fp8_layer_drift.py")
DRIFT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DRIFT)


class HfFp8LayerDriftTest(unittest.TestCase):
    def test_compare_stages_reports_complete_metrics(self):
        left = [{"name": "inference.blocks.0", "shape": [2], "values": [1.0, 2.0]}]
        right = [{"name": "inference.blocks.0", "shape": [2], "values": [1.0, 3.0]}]
        row = DRIFT.compare_stages(left, right)[0]
        self.assertEqual(row["max_abs"], 1.0)
        self.assertFalse(row["exact"])

    def test_selected_trace_rejects_truncation_and_orders_model_stages(self):
        records = [
            {"name": "inference.logits", "shape": [1], "values": [3.0],
             "values_truncated": False},
            {"name": "inference.blocks.1", "shape": [1], "values": [2.0],
             "values_truncated": False},
            {"name": "inference.blocks.0", "shape": [1], "values": [1.0],
             "values_truncated": False},
            {"name": "inference.final_norm", "shape": [1], "values": [2.5],
             "values_truncated": False},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("\n".join(map(__import__("json").dumps, records)) + "\n")
            selected = DRIFT.selected_trace(path)
            self.assertEqual([row["name"] for row in selected], [
                "inference.blocks.0", "inference.blocks.1",
                "inference.final_norm", "inference.logits"])
            records[0]["values_truncated"] = True
            path.write_text("\n".join(map(__import__("json").dumps, records)) + "\n")
            with self.assertRaises(RuntimeError):
                DRIFT.selected_trace(path)


if __name__ == "__main__":
    unittest.main()
