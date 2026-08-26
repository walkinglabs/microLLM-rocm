#!/usr/bin/env python3
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "benchmarks/results/2026-08-26-qwen3-bf16-inference/summary.json"
DIVERGENCE_ROOT = (ROOT / "benchmarks/results" /
                   "2026-08-26-qwen3-bf16-first-divergence")
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "audit_qwen3_bf16_divergence",
    ROOT / "benchmarks/single_gpu/audit_qwen3_bf16_divergence.py")
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(RUNNER)

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
    print("qwen3 bf16 evidence: pass")

if __name__ == "__main__":
    main()
