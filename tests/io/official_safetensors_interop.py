#!/usr/bin/env python3
import argparse
import subprocess
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


EXPECTED = {
    "layer.weight": torch.tensor(
        [-3.25, -1.0, -0.125, 0.0, 1.5, 7.75], dtype=torch.float32
    ).reshape(2, 3),
    "norm.weight": torch.tensor([1.0, 0.5, 2.0], dtype=torch.float32),
    "scalar": torch.tensor(0.333251953125, dtype=torch.float32),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prove bidirectional interoperability with official safetensors"
    )
    parser.add_argument("--helper", required=True, type=Path)
    return parser.parse_args()


def assert_fixture(path, dtype, tolerance):
    actual = load_file(path)
    if set(actual) != set(EXPECTED):
        raise AssertionError(f"unexpected keys in {path}: {sorted(actual)}")
    for name, expected in EXPECTED.items():
        tensor = actual[name]
        if tensor.dtype != dtype:
            raise AssertionError(f"{name}: expected {dtype}, got {tensor.dtype}")
        if tensor.shape != expected.shape:
            raise AssertionError(f"{name}: expected {expected.shape}, got {tensor.shape}")
        torch.testing.assert_close(
            tensor.float(), expected, atol=tolerance, rtol=0.0,
            msg=lambda message, tensor_name=name: f"{tensor_name}: {message}",
        )


def main():
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="microllm-official-safetensors-") as temporary:
        directory = Path(temporary)
        subprocess.run([args.helper, "write", directory], check=True)
        for suffix, dtype, tolerance in (
            ("f32", torch.float32, 0.0),
            ("bf16", torch.bfloat16, 2.0e-2),
            ("f16", torch.float16, 2.0e-3),
        ):
            assert_fixture(directory / f"cpp_{suffix}.safetensors", dtype, tolerance)
            save_file(
                {name: tensor.to(dtype).contiguous() for name, tensor in EXPECTED.items()},
                directory / f"python_{suffix}.safetensors",
                metadata={"producer": "official-safetensors-python"},
            )
        subprocess.run([args.helper, "verify", directory], check=True)
    print("directions=cpp-to-python,python-to-cpp")
    print("dtypes=float32,bfloat16,float16")
    print("status=pass")


if __name__ == "__main__":
    main()
