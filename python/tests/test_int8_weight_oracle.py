#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the C++ symmetric INT8 weight contract with PyTorch"
    )
    parser.add_argument("--helper", required=True, type=Path)
    return parser.parse_args()


def parse_output(text):
    result = {}
    for line in text.splitlines():
        key, value = line.split("=", maxsplit=1)
        result[key] = value
    return result


def main():
    args = parse_args()
    output = subprocess.run(
        [args.helper], check=True, capture_output=True, text=True
    ).stdout
    actual = parse_output(output)
    shape = tuple(int(value) for value in actual["shape"].split(","))
    scale = float(actual["scale"])
    source = torch.tensor(
        [-31.75, -3.0, -1.625, -0.375, 0.0,
          0.375, 1.625, 3.0, 31.75, 100.0],
        dtype=torch.float32,
    ).reshape(shape)
    expected_quantized = torch.clamp(torch.round(source / scale), -127, 127).to(
        torch.int8
    )
    actual_quantized = torch.tensor(
        [int(value) for value in actual["quantized"].split(",")],
        dtype=torch.int8,
    ).reshape(shape)
    torch.testing.assert_close(actual_quantized, expected_quantized, rtol=0, atol=0)

    actual_restored = torch.tensor(
        [float(value) for value in actual["restored"].split(",")],
        dtype=torch.float32,
    ).reshape(shape)
    torch.testing.assert_close(
        actual_restored, expected_quantized.float() * scale, rtol=0, atol=0
    )
    activation = torch.tensor(
        [1.0, -2.0, -1.0, 0.5, 2.0, 3.0], dtype=torch.float32
    ).reshape(3, 2)
    actual_matmul = torch.tensor(
        [float(value) for value in actual["matmul"].split(",")],
        dtype=torch.float32,
    ).reshape(3, 5)
    torch.testing.assert_close(
        actual_matmul,
        activation @ (expected_quantized.float() * scale),
        rtol=0,
        atol=0,
    )
    print("oracle=pytorch")
    print("shape=2x5")
    print("rounding=nearest-even")
    print("matmul=explicit-dequantize-baseline")
    print("status=pass")


if __name__ == "__main__":
    main()
