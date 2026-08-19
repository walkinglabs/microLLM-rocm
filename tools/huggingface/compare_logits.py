#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compare complete microLLM/PyTorch logits")
    parser.add_argument("--microllm", required=True, type=Path)
    parser.add_argument("--pytorch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=3.0e-4)
    args = parser.parse_args()
    actual = np.fromfile(args.microllm, dtype=np.float32)
    expected = np.fromfile(args.pytorch, dtype=np.float32)
    if actual.shape != expected.shape or actual.size == 0:
        raise ValueError(f"logit shape mismatch: {actual.shape} vs {expected.shape}")
    difference = np.abs(actual - expected)
    maximum_index = int(difference.argmax())
    denominator = np.maximum(np.abs(expected), 1.0e-12)
    report = {
        "schema_version": 1,
        "status": "pass" if float(difference.max()) <= args.atol else "fail",
        "count": int(actual.size),
        "atol": args.atol,
        "max_abs": float(difference.max()),
        "max_rel": float((difference / denominator).max()),
        "mse": float(np.mean((actual - expected) ** 2)),
        "cosine": float(np.dot(actual, expected) /
                        (np.linalg.norm(actual) * np.linalg.norm(expected))),
        "maximum_error_index": maximum_index,
        "microllm_at_max": float(actual[maximum_index]),
        "pytorch_at_max": float(expected[maximum_index]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
