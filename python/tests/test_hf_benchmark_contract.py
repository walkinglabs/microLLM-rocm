import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "hf_model_matrix", ROOT / "benchmarks/single_gpu/hf_model_matrix.py")
HF_MATRIX = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(HF_MATRIX)


def model_entry(batch=2):
    return {
        "name": "fixture",
        "revision": "fixed",
        "parameter_count": 10,
        "loaded_tensors": 2,
        "config": "/missing/config.json",
        "weights": "/missing/model.safetensors",
        "vocab": "/missing/vocab.json",
        "merges": "/missing/merges.txt",
        "tokenizer_family": "qwen2",
        "inference": {"token_ids": [1], "new_tokens": 1},
        "training": {
            "tokens": "1,2,3,4",
            "batch": batch,
            "learning_rate": 1.0e-5,
            "warmup": 1,
            "steps": 2,
        },
    }


class HfBenchmarkBatchContractTest(unittest.TestCase):
    def test_micro_command_and_schema_keep_batch_context_and_token_count(self):
        captured = []

        def fake_run(command):
            captured.extend(command)
            return {
                "schema_version": 1,
                "status": "pass",
                "device": "hip:0",
                "compute_dtype": "float32",
                "parameter_count": 10,
                "loaded_tensors": 2,
                "fp32_weight_bytes": 40,
                "engine_current_bytes": 40,
                "engine_peak_bytes": 80,
                "engine_total_allocated_bytes": 120,
                "step_ms": 2.0,
                "tokens_per_second": 6000.0,
                "milliseconds_per_token": 1.0 / 6.0,
                "parameter_changed": True,
                "loss": 1.0,
                "optimizer_host_to_device_calls": 0,
                "optimizer_device_to_host_calls": 0,
                "warmup": 1,
                "steps": 2,
                "batch": 2,
                "context": 3,
                "trained_tokens": 12,
            }

        with mock.patch.object(HF_MATRIX, "run_json", side_effect=fake_run):
            record = HF_MATRIX.train(Path("fake-train"), model_entry(), "hip")
        batch_index = captured.index("--batch")
        self.assertEqual(captured[batch_index + 1], "2")
        self.assertEqual(record["batch"], 2)
        self.assertEqual(record["context"], 3)
        self.assertEqual(record["trained_tokens"], 12)

    def test_manifest_rejects_nonpositive_batch(self):
        document = {"schema_version": 1, "models": [model_entry(batch=0)]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "batch must be positive"):
                HF_MATRIX.load_manifest(path)


if __name__ == "__main__":
    unittest.main()
