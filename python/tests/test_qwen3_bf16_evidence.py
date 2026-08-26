#!/usr/bin/env python3
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "benchmarks/results/2026-08-26-qwen3-bf16-inference/summary.json"
DIVERGENCE_ROOT = (ROOT / "benchmarks/results" /
                   "2026-08-26-qwen3-bf16-first-divergence")
SWEEP_ROOT = (ROOT / "benchmarks/results" /
              "2026-08-26-qwen3-bf16-oracle-sweep")
ISLAND_ROOT = (ROOT / "benchmarks/results" /
               "2026-08-26-qwen3-bf16-t128-weight-islands")
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "audit_qwen3_bf16_divergence",
    ROOT / "benchmarks/single_gpu/audit_qwen3_bf16_divergence.py")
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(RUNNER)
SWEEP_SPEC = importlib.util.spec_from_file_location(
    "qwen3_bf16_oracle_sweep",
    ROOT / "benchmarks/single_gpu/qwen3_bf16_oracle_sweep.py")
SWEEP = importlib.util.module_from_spec(SWEEP_SPEC)
assert SWEEP_SPEC.loader is not None
SWEEP_SPEC.loader.exec_module(SWEEP)

def main():
    row = json.loads(SUMMARY.read_text())
    assert row["status"] == "pass_explicit_policy"
    assert row["bf16_ffn_linears"] == 84
    assert row["bf16_attention_linears"] == 112
    assert row["qk_norm_dtype"] == "float32"
    assert row["tokens"] == [14582, 25, 16246, 264]
    assert row["fp32_oracle_elements"] == 151_936
    assert row["microllm_bf16_vs_fp32_max"] < row["pytorch_bf16_vs_fp32_max"]
    assert row["microllm_bf16_vs_fp32_rms"] < row["pytorch_bf16_vs_fp32_rms"]
    assert row["microllm_warmup"] == row["pytorch_warmup"] == 2
    assert row["microllm_repetitions"] == row["pytorch_repetitions"] == 5
    assert row["microllm_over_pytorch"] > 1.0
    assert row["bf16_resident_weight_bytes"] < row["fp32_resident_weight_bytes"]
    divergence = json.loads((DIVERGENCE_ROOT / "summary.json").read_text())
    raw = [json.loads(line) for line in
           (DIVERGENCE_ROOT / "raw.jsonl").read_text().splitlines() if line]
    assert divergence["status"] == "pass_diagnosed_precision_policy"
    assert all(divergence["gates"].values())
    assert divergence["context"] == 32
    assert divergence["capture_step"] == 1
    assert divergence["vocabulary_size"] == 151_936
    assert len(raw) == len(divergence["policy_rows"]) == 6
    assert all(item["status"] == "pass" for item in raw)
    policies = {item["policy"]: item for item in divergence["policy_rows"]}
    assert policies["torch-fp32"]["argmax_token"] == 374
    assert policies["micro-fp32-fp32"]["argmax_token"] == 374
    assert policies["micro-bf16-bf16"]["argmax_token"] == 374
    assert policies["torch-bf16"]["argmax_token"] == 323
    assert policies["torch-bf16"]["top1_top2_margin"] == 0.0
    assert policies["torch-bf16"]["top3"][0]["logit"] == \
        policies["torch-bf16"]["top3"][1]["logit"] == 14.1875
    assert policies["micro-bf16-bf16"]["top1_top2_margin"] > 0.03
    assert policies["micro-fp32-fp32"]["versus_torch_fp32_maximum_error"] < 6e-5
    assert policies["micro-fp32-fp32"]["versus_torch_fp32_rms_error"] < 1.3e-5
    assert policies["micro-bf16-bf16"]["versus_torch_fp32_maximum_error"] < \
        policies["torch-bf16"]["versus_torch_fp32_maximum_error"]
    maximum, rms, bitwise = RUNNER.error([1.0, 2.0], [1.0, 3.0])
    assert maximum == 1.0 and math.isclose(rms, math.sqrt(0.5)) and not bitwise
    assert RUNNER.top_tokens([0.0, 3.0, 2.0]) == [1, 2, 0]
    sweep = json.loads((SWEEP_ROOT / "summary.json").read_text())
    sweep_raw = [json.loads(line) for line in
                 (SWEEP_ROOT / "raw.jsonl").read_text().splitlines() if line]
    assert sweep["status"] == "pass_all_mismatches_attributed"
    assert sweep["unique_oracle_cases"] == len(SWEEP.CASES) == 5
    assert sweep["matrix_mismatch_rows"] == 8
    assert sweep["micro_oracle_case_wins"] == 4
    assert sweep["torch_oracle_case_wins"] == 1
    assert sweep["micro_oracle_matrix_rows"] == 7
    assert sweep["torch_oracle_matrix_rows"] == 1
    case_rows = {item["name"]: item for item in sweep["case_rows"]}
    expected = {
        "t32-b1-step1": (374, 374, 323),
        "t32-b2-step1": (374, 374, 323),
        "t128-b2-step8": (320, 25, 320),
        "t512-b1-step2": (2955, 2955, 1096),
        "t512-b2-step8-forced": (1273, 1273, 4285),
    }
    assert {name: (item["oracle_argmax"], item["micro_mixed_argmax"],
                   item["torch_bf16_argmax"])
            for name, item in case_rows.items()} == expected
    forced = case_rows["t512-b2-step8-forced"]["forced_inputs"]
    assert forced == [14582, 198, 262, 1096, 374, 279, 2038, 374, 264]
    assert len(sweep_raw) == 28 and all(item["status"] == "pass"
                                        for item in sweep_raw)
    forced_raw = [item for item in sweep_raw
                  if item["oracle_case"] == "t512-b2-step8-forced"]
    assert len(forced_raw) == 4
    assert all(item["forced_decode_inputs"] is True and
               item["forced_decode_input_count"] == 9 for item in forced_raw)
    islands = json.loads((ISLAND_ROOT / "summary.json").read_text())
    island_raw = [json.loads(line) for line in
                  (ISLAND_ROOT / "raw.jsonl").read_text().splitlines() if line]
    assert islands["status"] == "pass_diagnosed_precision_policy"
    assert all(islands["gates"].values())
    assert islands["context"] == 128 and islands["batch"] == 2
    assert islands["forced_inputs"] == \
        [14582, 1, 374, 264, 3491, 429, 374, 537, 264]
    policies = {item["policy"]: item for item in islands["policy_rows"]}
    assert len(policies) == len(island_raw) == 7
    assert policies["micro-fp32-fp32"]["argmax_token"] == 320
    assert policies["micro-ffn-bf16-fp32"]["argmax_token"] == 25
    assert policies["micro-attention-bf16-fp32"]["argmax_token"] == 320
    assert policies["micro-bf16-fp32"]["argmax_token"] == 25
    assert policies["micro-bf16-bf16"]["argmax_token"] == 25
    assert policies["torch-bf16"]["argmax_token"] == 320
    assert policies["micro-ffn-bf16-fp32"]["captured_rows_maximum_error"] > 0.14
    assert policies["micro-attention-bf16-fp32"]["captured_rows_maximum_error"] < 0.033
    raw_by_policy = {item["framework_policy"]: item for item in island_raw}
    assert raw_by_policy["micro-ffn-bf16-fp32"]["bf16_ffn_weights"] is True
    assert raw_by_policy["micro-ffn-bf16-fp32"]["bf16_attention_weights"] is False
    assert raw_by_policy["micro-attention-bf16-fp32"]["bf16_ffn_weights"] is False
    assert raw_by_policy["micro-attention-bf16-fp32"]["bf16_attention_weights"] is True
    print("qwen3 bf16 evidence: pass")

if __name__ == "__main__":
    main()
