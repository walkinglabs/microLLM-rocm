#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "benchmarks/results/2026-08-26-qwen3-bf16-inference/summary.json"

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
    print("qwen3 bf16 evidence: pass")

if __name__ == "__main__":
    main()
