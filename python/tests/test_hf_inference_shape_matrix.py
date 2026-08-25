import importlib.util
import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "hf_inference_shape_matrix",
    ROOT / "benchmarks/single_gpu/hf_inference_shape_matrix.py")
MATRIX = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MATRIX)
RENDER_SPEC = importlib.util.spec_from_file_location(
    "render_serving_batch_scale",
    ROOT / "benchmarks/single_gpu/render_serving_batch_scale.py")
RENDER = importlib.util.module_from_spec(RENDER_SPEC)
assert RENDER_SPEC.loader is not None
RENDER_SPEC.loader.exec_module(RENDER)


class HfInferenceShapeMatrixTest(unittest.TestCase):
    def test_current_serving_batch_scale_is_complete_and_rendered(self):
        root = ROOT / "benchmarks/results/2026-08-25-serving-batch-scale"
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        raw = [json.loads(line) for line in (root / "raw.jsonl").read_text(
            encoding="utf-8").splitlines() if line]
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(len(summary["rows"]), 8)
        self.assertEqual(len(raw), 48)
        self.assertEqual(sum(row["framework"] == "microllm" and
                             row["cached_attention_materialized_policy"] ==
                             "auto-enabled" for row in raw), 24)
        self.assertEqual(sum(row["framework"] == "pytorch" and
                             row["device_discovery_workaround"] ==
                             "amdsmi_zero_fallback_to_hip_runtime"
                             for row in raw), 24)
        self.assertEqual(analysis["qwen_b8_scaling"], 6.5851601491695755)
        self.assertEqual(analysis["deepseek_b8_scaling"], 6.282360624661898)
        self.assertEqual(analysis["deepseek_divergent_batches"], [1, 8])
        self.assertFalse(analysis["scheduler_default_admitted"])
        self.assertEqual(verification["measurement_commit"],
                         "b4138b9be073ae51407ab86263957c47d82a6dda")
        ET.parse(root / "batch-scale.svg")

    def test_current_deepseek_t2048_baseline_is_complete_and_exact(self):
        root = ROOT / "benchmarks/results/2026-08-25-current-deepseek-t2048"
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        verification = json.loads((root / "verification.json").read_text(
            encoding="utf-8"))
        attempts = json.loads((root / "attempts.json").read_text(encoding="utf-8"))
        raw = [json.loads(line) for line in (root / "raw.jsonl").read_text(
            encoding="utf-8").splitlines() if line]
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["runs_per_framework"], 3)
        self.assertEqual(len(summary["rows"]), 1)
        row = summary["rows"][0]
        self.assertEqual((row["context"], row["batch"], row["decode_tokens"]),
                         (2048, 2, 64))
        self.assertTrue(row["cross_framework_tokens_equal"])
        self.assertEqual(row["cross_framework_first_token_difference"], -1)
        self.assertEqual(row["cross_framework_matching_prefix_tokens"], 64)
        self.assertEqual(row["microllm_throughput_tokens_per_second"],
                         133.501430352)
        self.assertEqual(row["pytorch_throughput_tokens_per_second"],
                         163.64307901695054)
        self.assertEqual(row["throughput_ratio_microllm_over_pytorch"],
                         0.8158085948637742)
        self.assertEqual(row["microllm_kv_cache_actual_bytes"], 121110528.0)
        self.assertEqual(row["pytorch_kv_cache_actual_bytes"], 121110528.0)
        self.assertEqual(row["microllm_kv_cache_utilization"], 1.0)
        self.assertEqual(row["pytorch_kv_cache_utilization"], 1.0)
        self.assertEqual(len(raw), 6)
        self.assertTrue(all(record["status"] == "pass" for record in raw))
        self.assertEqual({record["framework"] for record in raw},
                         {"microllm", "pytorch"})
        self.assertEqual(verification["measurement_commit"],
                         "4ac239384a8c4f3f26b49e4884efeb32e0f02cfa")
        self.assertTrue(verification["performance_gap_reproduced"])
        self.assertEqual(attempts["attempts"][0]["status"],
                         "invalid_for_cross_framework_comparison")
        self.assertEqual(attempts["attempts"][1]["status"], "pass")

    def test_named_suites_cover_short_long_context_and_batch_scaling(self):
        self.assertEqual(MATRIX.MATRIX_SUITES["smoke"]["contexts"], [8, 128])
        standard = MATRIX.MATRIX_SUITES["standard"]
        self.assertIn(8, standard["contexts"])
        self.assertIn(2048, standard["contexts"])
        self.assertEqual(standard["batches"], [1, 2, 4, 8])
        serving = MATRIX.MATRIX_SUITES["serving"]
        self.assertEqual(serving["contexts"], [1, 8, 32, 128, 512, 2048])
        self.assertEqual(serving["batches"], [1, 2, 4, 8])
        self.assertEqual(serving["decode_lengths"], [1, 8, 32, 64])
        extended = MATRIX.MATRIX_SUITES["extended"]
        self.assertIn(1, extended["contexts"])
        self.assertIn(4096, extended["contexts"])
        self.assertIn(16, extended["batches"])
        self.assertEqual(extended["decode_lengths"], [1, 8, 32])
        boundary = MATRIX.MATRIX_SUITES["boundary"]
        self.assertTrue({31, 32, 33, 127, 128, 129, 511, 512, 513} <=
                        set(boundary["contexts"]))
        self.assertIn(3, boundary["batches"])

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
            "device_total_bytes": 1000, "resident_weight_bytes": 80,
            "kv_cache_active_bytes": 40, "kv_cache_utilization": 1.0,
            "throughput_tokens_per_second": 10.0,
            "kv_cache_theoretical_bytes": 40, "kv_cache_actual_bytes": 40,
        }
        with self.assertRaisesRegex(RuntimeError, "shape contract"):
            MATRIX.validate_measurement(
                record, model, "microllm", 128, 2, "decode", "cached", 1, 2, 8)

    def test_validator_rejects_prefill_semantic_drift(self):
        model = {"name": "qwen", "parameter_count": 10}
        record = {
            "status": "pass", "model": "qwen", "parameter_count": 10,
            "context": 128, "batch": 1, "workload": "prefill",
            "cache_mode": "uncached", "prefill_logits_mode": "full",
            "precision": "mixed_bf16_weights_fp32_activations",
            "warmup": 1, "steps": 2, "peak_bytes": 100,
            "device_total_bytes": 1000, "resident_weight_bytes": 80,
            "throughput_tokens_per_second": 10.0,
            "top_logits": [{"token": 1, "logit": 2.0}],
        }
        with self.assertRaisesRegex(RuntimeError, "prefill logits semantics"):
            MATRIX.validate_measurement(
                record, model, "microllm", 128, 1, "prefill", "uncached",
                1, 2, 8, "last")

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

    def test_first_sequence_difference_reports_exact_divergence(self):
        self.assertEqual(MATRIX.first_sequence_difference([1, 2], [1, 2]), -1)
        self.assertEqual(MATRIX.first_sequence_difference([1, 2], [1, 3]), 1)
        self.assertEqual(MATRIX.first_sequence_difference([1], [1, 2]), 1)

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
            "device_total_bytes": 10000, "engine_peak_share_of_device": 0.2,
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
        self.assertEqual(record["kv_cache_reservation_policy"],
                         "fixed_exact_capacity")
        self.assertEqual(record["peak_memory_share_of_device"], 0.2)
        self.assertEqual(record["precision"],
                         "mixed_bf16_weights_fp32_activations")

    def test_micro_command_propagates_explicit_bf16_cache_policy(self):
        args = type("Args", (), {
            "micro_binary": Path("micro"), "decode_tokens": 4,
            "warmup": 1, "steps": 2, "micro_batch_argmax_mode": "device",
            "micro_kv_cache_dtype": "bf16",
            "micro_kv_cache_fp32_layers": "0,2",
            "prefill_logits_mode": "last"})()
        model = {"config": "config.json", "weights": "weights.bin",
                 "inference": {"token_ids": [1, 2]}}
        command = MATRIX.micro_command(
            args, model, context=8, batch=4, workload="decode", cache="cached")
        policy = command.index("--kv-cache-dtype")
        self.assertEqual(command[policy + 1], "bf16")
        batch = command.index("--batch")
        self.assertEqual(command[batch + 1], "4")
        layers = command.index("--kv-cache-fp32-layers")
        self.assertEqual(command[layers + 1], "0,2")
        logits = command.index("--prefill-logits")
        self.assertEqual(command[logits + 1], "last")
        decode_mode = command.index("--decode-mode")
        self.assertEqual(command[decode_mode + 1], "steady")
        capacity = command.index("--cache-capacity")
        self.assertEqual(command[capacity + 1], "12")

    def test_mixed_layer_cache_theoretical_bytes_sum_each_dtype(self):
        args = type("Args", (), {"warmup": 1, "steps": 1,
                                  "decode_tokens": 4,
                                  "micro_kv_cache_dtype": "bf16"})()
        raw = {
            "decode_tokens_per_second": 10.0, "mean_generation_ms": 1.0,
            "engine_peak_bytes": 1000, "measured_tokens": 4,
            "device_total_bytes": 10000, "engine_peak_share_of_device": 0.1,
            "resident_weight_bytes": 800,
            "inference_weight_policy": "single_representation_bf16_ffn_attention",
            "kv_cache_actual_bytes": 192, "kv_cache_active_bytes": 192,
            "kv_cache_capacity_tokens": 4, "kv_cache_active_tokens": 4,
            "kv_cache_layers": 2, "kv_cache_heads": 1,
            "kv_cache_head_dimension": 4, "kv_cache_element_bytes": 0,
            "kv_cache_fp32_layers": 1, "kv_cache_bf16_layers": 1,
            "kv_cache_utilization": 1.0,
        }
        record = MATRIX.normalize_micro(
            raw, {"name": "tiny", "revision": "fixed"}, 4, 1,
            "decode", "cached", args)
        self.assertEqual(record["kv_cache_theoretical_bytes"], 192)
        self.assertEqual(record["kv_cache_fp32_layers"], 1)
        self.assertEqual(record["kv_cache_bf16_layers"], 1)

    def test_sweep_capacity_exposes_unused_short_decode_pages(self):
        args = type("Args", (), {
            "micro_binary": Path("micro"), "decode_tokens": 1,
            "decode_lengths": [1, 8], "micro_cache_capacity": "sweep-max",
            "warmup": 1, "steps": 2, "micro_batch_argmax_mode": "device",
            "micro_kv_cache_dtype": "bf16", "micro_kv_cache_fp32_layers": "",
            "prefill_logits_mode": "last"})()
        model = {"config": "config.json", "weights": "weights.bin",
                 "inference": {"token_ids": [1, 2]}}
        command = MATRIX.micro_command(
            args, model, context=32, batch=4, workload="decode", cache="cached",
            decode_tokens=1)
        capacity = command.index("--cache-capacity")
        self.assertEqual(command[capacity + 1], "40")

    def test_summary_computes_batch_and_memory_efficiency(self):
        model = {"name": "tiny", "revision": "fixed"}
        records = []
        for framework in ("microllm", "pytorch"):
            for batch, throughput, peak, cache in (
                    (1, 100.0, 1000.0, 100.0),
                    (4, 320.0, 2500.0, 400.0)):
                records.append({
                    "model": "tiny", "context": 128, "batch": batch,
                    "workload": "decode", "cache_mode": "cached",
                    "framework": framework, "status": "pass",
                    "throughput_tokens_per_second": throughput,
                    "latency_ms": 1.0, "peak_bytes": peak,
                    "device_total_bytes": 10000.0,
                    "peak_memory_share_of_device": peak / 10000.0,
                    "resident_weight_bytes": 800.0,
                    "kv_cache_actual_bytes": cache,
                    "kv_cache_theoretical_bytes": cache,
                    "kv_cache_allocation_efficiency": 1.0,
                    "kv_cache_utilization": 1.0,
                    "kv_cache_share_of_peak": cache / peak,
                    "generated_tokens": [1, 2],
                })
        summary = MATRIX.summarize(
            records, [model], [128], [1, 4], 1, cases=["cached"])
        row = summary["rows"][1]
        self.assertAlmostEqual(row["microllm_batch_throughput_scaling"], 3.2)
        self.assertAlmostEqual(row["microllm_batch_efficiency"], 0.8)
        self.assertAlmostEqual(row["microllm_peak_memory_scaling"], 2.5)
        self.assertEqual(row["microllm_kv_cache_bytes_per_request"], 100.0)
        self.assertEqual(row["microllm_peak_bytes_per_request"], 625.0)
        self.assertEqual(summary["axes"]["batches"], [1, 4])

    def test_validator_checks_active_capacity_and_utilization(self):
        model = {"name": "tiny", "parameter_count": 10}
        record = {
            "status": "pass", "model": "tiny", "parameter_count": 10,
            "context": 8, "batch": 2, "workload": "decode",
            "cache_mode": "cached",
            "precision": "mixed_bf16_weights_fp32_activations",
            "warmup": 1, "steps": 2, "measured_tokens": 16,
            "measured_forward_steps": 16,
            "peak_bytes": 1000, "device_total_bytes": 10000,
            "resident_weight_bytes": 800,
            "throughput_tokens_per_second": 20.0,
            "decode_step_semantics": "one_model_forward_per_measured_token",
            "kv_cache_theoretical_bytes": 120,
            "kv_cache_actual_bytes": 120, "kv_cache_active_bytes": 120,
            "kv_cache_capacity_tokens": 12, "kv_cache_active_tokens": 12,
            "kv_cache_utilization": 1.0,
        }
        MATRIX.validate_measurement(
            record, model, "microllm", 8, 2, "decode", "cached", 1, 2, 4)
        record["kv_cache_active_tokens"] = 11
        with self.assertRaisesRegex(RuntimeError, "KV token accounting"):
            MATRIX.validate_measurement(
                record, model, "microllm", 8, 2, "decode", "cached", 1, 2, 4)

    def test_decode_length_axis_and_process_tail_metrics_are_preserved(self):
        model = {"name": "tiny", "revision": "fixed"}
        records = []
        for decode_tokens in (1, 8):
            for framework in ("microllm", "pytorch"):
                for run, latency in enumerate((10.0, 12.0, 20.0), start=1):
                    records.append({
                        "model": "tiny", "context": 128, "batch": 2,
                        "decode_tokens": decode_tokens,
                        "workload": "decode", "cache_mode": "cached",
                        "framework": framework, "status": "pass",
                        "process_run": run,
                        "throughput_tokens_per_second":
                            decode_tokens * 2 * 1000.0 / latency,
                        "latency_ms": latency, "peak_bytes": 1200,
                        "device_total_bytes": 10000,
                        "peak_memory_share_of_device": 0.12,
                        "resident_weight_bytes": 800,
                        "kv_cache_actual_bytes": 240,
                        "kv_cache_active_bytes": 220,
                        "kv_cache_theoretical_bytes": 240,
                        "kv_cache_capacity_tokens": 136,
                        "kv_cache_active_tokens": 135,
                        "kv_cache_element_bytes": 2,
                        "kv_cache_allocation_efficiency": 1.0,
                        "kv_cache_utilization": 220 / 240,
                        "kv_cache_share_of_peak": 0.2,
                        "generated_tokens": list(range(decode_tokens)),
                    })
        summary = MATRIX.summarize(
            records, [model], [128], [2], 3, cases=["cached"],
            decode_lengths=[1, 8])
        self.assertEqual(len(summary["rows"]), 2)
        self.assertEqual(summary["axes"]["decode_lengths"], [1, 8])
        long_row = summary["rows"][1]
        self.assertEqual(long_row["decode_tokens"], 8)
        self.assertEqual(long_row["microllm_process_latency_ms_p50"], 12.0)
        self.assertAlmostEqual(long_row["microllm_process_latency_ms_p95"], 19.2)
        self.assertEqual(long_row["microllm_peak_incremental_bytes"], 400.0)
        self.assertEqual(long_row[
            "microllm_peak_incremental_bytes_per_request"], 200.0)
        self.assertAlmostEqual(long_row[
            "microllm_kv_cache_share_of_incremental_peak"], 0.6)
        self.assertEqual(long_row["microllm_kv_cache_waste_bytes"], 20.0)
        self.assertEqual(long_row[
            "microllm_kv_cache_waste_bytes_per_request"], 10.0)
        self.assertAlmostEqual(long_row["microllm_kv_cache_waste_ratio"], 1.0 / 12.0)
        self.assertAlmostEqual(long_row[
            "microllm_kv_cache_active_share_of_incremental_peak"], 0.55)
        self.assertEqual(long_row["microllm_non_kv_incremental_bytes"], 160.0)
        self.assertGreater(long_row[
            "microllm_decode_length_throughput_vs_shortest"], 1.0)


if __name__ == "__main__":
    unittest.main()
