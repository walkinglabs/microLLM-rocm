#!/usr/bin/env python3
"""Reject a stale hf_infer binary whose embedded CLI contract lags the source."""

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    args = parser.parse_args()
    if not args.binary.is_file():
        raise RuntimeError("hf_infer binary is missing")
    payload = args.binary.read_bytes()
    required = (
        b"device-tensor-amax",
        b"ffn-outer-row",
        b"--fp8-activation-minimum-scale",
        b"fp8_device_weight_bytes_scanned",
        b"fp8_dynamic_tensor_calls",
        b"--fp8-fp32-layers",
        b"--fp8-diagnostic-mode",
        b"fp8_linears_covered",
        b"both-roundtrip",
        b"fp8_output_column_scale_calls",
        b"fp8_dynamic_column_calls",
        b"output-channel-amax",
        b"fp8_output_column_native_status",
        b"--fp8-weight-scale-scope",
    )
    missing = [value.decode() for value in required if value not in payload]
    if missing:
        raise RuntimeError(f"hf_infer binary has a stale CLI contract: {missing}")
    print("hf_infer binary contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
