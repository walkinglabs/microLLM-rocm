#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "benchmarks/results/2026-08-26-qwen3-official-alignment/summary.json"


def main():
    row = json.loads(SUMMARY.read_text())
    assert row["status"] == "pass"
    assert row["strict_streaming_loaded_tensors"] == 310
    assert row["stored_tensors"] == 311
    assert row["stored_parameter_values"] == 751_632_384
    assert row["unique_runtime_parameters"] == 596_049_920
    assert row["tied_alias_payload_exact"] is True
    assert row["tied_alias_bytes"] == 311_164_928
    assert row["logit_elements"] == 151_936
    assert row["logit_maximum_absolute_error"] <= 4.0e-5
    assert row["logit_rms_error"] <= 9.0e-6
    assert row["microllm_argmax"] == row["pytorch_argmax"] == 14_582
    assert row["microllm_generated_tokens"] == row["pytorch_generated_tokens"]
    assert row["microllm_generated_tokens"] == [14_582, 25, 16_246, 264]
    assert row["decode_tokens_per_second"] > 0
    print("qwen3 official evidence: pass")


if __name__ == "__main__":
    main()
