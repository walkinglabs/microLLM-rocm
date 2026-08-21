import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "hf_inference_shape_matrix",
    ROOT / "benchmarks/single_gpu/hf_inference_shape_matrix.py")
MATRIX = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MATRIX)


class HfInferenceShapeMatrixTest(unittest.TestCase):
    def test_context_batch_and_kv_formulas_are_explicit(self):
        self.assertEqual(MATRIX.positive_int_list("8,128,512", "contexts"),
                         [8, 128, 512])
        self.assertEqual(MATRIX.expanded_tokens([10, 20, 30], 5),
                         [10, 20, 30, 10, 20])
        self.assertEqual(
            MATRIX.theoretical_kv_cache_bytes(
                layers=24, kv_heads=2, head_dimension=64,
                batch=4, tokens=512, element_bytes=2),
            2 * 24 * 2 * 64 * 4 * 512 * 2)
        with self.assertRaisesRegex(Exception, "unique positive"):
            MATRIX.positive_int_list("1,1", "batches")

    def test_validator_rejects_silent_batch_fallback(self):
        model = {"name": "qwen", "parameter_count": 10}
        record = {
            "status": "pass", "model": "qwen", "parameter_count": 10,
            "context": 128, "batch": 1, "workload": "decode",
            "cache_mode": "cached",
            "precision": "mixed_bf16_weights_fp32_activations", "warmup": 1,
            "steps": 2, "measured_tokens": 32, "peak_bytes": 100,
            "resident_weight_bytes": 80, "kv_cache_utilization": 1.0,
            "throughput_tokens_per_second": 10.0,
            "kv_cache_theoretical_bytes": 40, "kv_cache_actual_bytes": 40,
        }
        with self.assertRaisesRegex(RuntimeError, "shape contract"):
            MATRIX.validate_measurement(
                record, model, "microllm", 128, 2, "decode", "cached", 1, 2, 8)

    def test_summary_preserves_limits_and_computes_paired_medians(self):
        model = {"name": "qwen", "revision": "fixed"}
        records = []
        for framework, throughputs, peaks in (
            ("microllm", [100.0, 120.0, 110.0], [1000, 1200, 1100]),
            ("pytorch", [200.0, 220.0, 210.0], [800, 1000, 900]),
        ):
            for throughput, peak in zip(throughputs, peaks):
                records.append({
                    "model": "qwen", "context": 8, "batch": 1,
                    "workload": "prefill", "cache_mode": "uncached",
                    "framework": framework, "status": "pass",
                    "throughput_tokens_per_second": throughput,
                    "latency_ms": 1.0, "peak_bytes": peak,
                    "resident_weight_bytes": 700,
                    "kv_cache_actual_bytes": 0, "kv_cache_theoretical_bytes": 0,
                    "kv_cache_allocation_efficiency": 0.0,
                    "kv_cache_utilization": 0.0,
                    "kv_cache_share_of_peak": 0.0,
                })
        # The two decode cases remain explicitly limited because no fake samples exist.
        summary = MATRIX.summarize(records, [model], [8], [1], 3)
        prefill = summary["rows"][0]
        self.assertEqual(prefill["status"], "pass")
        self.assertEqual(prefill["microllm_throughput_tokens_per_second"], 110.0)
        self.assertEqual(prefill["pytorch_throughput_tokens_per_second"], 210.0)
        self.assertAlmostEqual(prefill["throughput_ratio_microllm_over_pytorch"],
                               110.0 / 210.0)
        self.assertAlmostEqual(prefill["peak_memory_ratio_microllm_over_pytorch"],
                               1100.0 / 900.0)
        self.assertEqual(summary["status"], "complete_with_recorded_limits")

    def test_failure_classification_keeps_unsupported_and_oom(self):
        self.assertEqual(MATRIX.classify_failure(
            "cached decode currently supports batch 1"), "unsupported")
        self.assertEqual(MATRIX.classify_failure("HIP out of memory"), "oom")
        self.assertEqual(MATRIX.classify_failure("wrong answer"), "failed")

    def test_case_filter_rejects_unknown_and_omits_unrequested_rows(self):
        self.assertEqual(MATRIX.case_list("prefill,cached"), ["prefill", "cached"])
        with self.assertRaisesRegex(Exception, "cases must contain"):
            MATRIX.case_list("prefill,magic")
        model = {"name": "tiny", "revision": "fixed"}
        records = []
        for framework in ("microllm", "pytorch"):
            records.append({
                "model": "tiny", "context": 8, "batch": 1,
                "workload": "prefill", "cache_mode": "uncached",
                "framework": framework, "status": "pass",
                "throughput_tokens_per_second": 1.0, "latency_ms": 1.0,
                "peak_bytes": 10, "resident_weight_bytes": 8,
            })
        summary = MATRIX.summarize(
            records, [model], [8], [1], 1, cases=["prefill"])
        self.assertEqual(len(summary["rows"]), 1)
        self.assertEqual(summary["status"], "pass")

    def test_micro_normalization_separates_storage_and_active_cache(self):
        args = type("Args", (), {"warmup": 1, "steps": 2, "decode_tokens": 4,
                                  "micro_kv_cache_dtype": "bf16"})()
        raw = {
            "decode_tokens_per_second": 50.0, "mean_generation_ms": 8.0,
            "engine_peak_bytes": 2000, "measured_tokens": 8,
            "resident_weight_bytes": 1000,
            "inference_weight_policy": "single_representation_bf16_ffn_attention",
            "kv_cache_actual_bytes": 320, "kv_cache_active_bytes": 224,
            "kv_cache_capacity_tokens": 10, "kv_cache_active_tokens": 7,
            "kv_cache_layers": 2, "kv_cache_heads": 1,
            "kv_cache_head_dimension": 4, "kv_cache_element_bytes": 2,
            "kv_cache_utilization": 0.7,
        }
        record = MATRIX.normalize_micro(
            raw, {"name": "tiny", "revision": "fixed"}, 6, 1,
            "decode", "cached", args)
        self.assertEqual(record["kv_cache_actual_bytes"], 320)
        self.assertEqual(record["kv_cache_theoretical_bytes"], 320)
        self.assertEqual(record["kv_cache_utilization"], 0.7)
        self.assertEqual(record["kv_cache_dtype"], "bf16")
        self.assertEqual(record["precision"],
                         "mixed_bf16_weights_fp32_activations")

    def test_micro_command_propagates_explicit_bf16_cache_policy(self):
        args = type("Args", (), {
            "micro_binary": Path("micro"), "decode_tokens": 4,
            "warmup": 1, "steps": 2, "micro_batch_argmax_mode": "device",
            "micro_kv_cache_dtype": "bf16"})()
        model = {"config": "config.json", "weights": "weights.bin",
                 "inference": {"token_ids": [1, 2]}}
        command = MATRIX.micro_command(
            args, model, context=8, batch=4, workload="decode", cache="cached")
        policy = command.index("--kv-cache-dtype")
        self.assertEqual(command[policy + 1], "bf16")
        batch = command.index("--batch")
        self.assertEqual(command[batch + 1], "4")


if __name__ == "__main__":
    unittest.main()
