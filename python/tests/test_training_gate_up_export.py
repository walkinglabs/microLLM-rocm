#!/usr/bin/env python3
"""Exercise diagnostic gate/up gradient and parameter safetensors on CPU."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path


GATE_UP_SHAPES = {
    "blocks.0.feed_forward.gate_proj.weight",
    "blocks.0.feed_forward.up_proj.weight",
}
GATE_UP_SHAPES = {name: [8, 16] for name in GATE_UP_SHAPES}
ALL_SHAPES = {
    "token_embedding.weight": [8, 8],
    "blocks.0.attention_norm.weight": [8],
    "blocks.0.attention.q_proj.weight": [8, 8],
    "blocks.0.attention.q_proj.bias": [8],
    "blocks.0.attention.k_proj.weight": [8, 4],
    "blocks.0.attention.k_proj.bias": [4],
    "blocks.0.attention.v_proj.weight": [8, 4],
    "blocks.0.attention.v_proj.bias": [4],
    "blocks.0.attention.o_proj.weight": [8, 8],
    "blocks.0.ffn_norm.weight": [8],
    "blocks.0.feed_forward.gate_proj.weight": [8, 16],
    "blocks.0.feed_forward.up_proj.weight": [8, 16],
    "blocks.0.feed_forward.down_proj.weight": [16, 8],
    "final_norm.weight": [8],
}


def header(path: Path, expected: dict[str, list[int]]) -> dict:
    data = path.read_bytes()
    if len(data) < 8:
        raise AssertionError("safetensors output is truncated")
    size = struct.unpack("<Q", data[:8])[0]
    document = json.loads(data[8:8 + size])
    document.pop("__metadata__", None)
    assert set(document) == set(expected)
    ranges = []
    total_bytes = 0
    for name, value in document.items():
        assert value["dtype"] == "F32"
        assert value["shape"] == expected[name]
        begin, end = value["data_offsets"]
        elements = 1
        for dimension in expected[name]:
            elements *= dimension
        assert end - begin == elements * 4
        ranges.append((begin, end))
        total_bytes += elements * 4
    cursor = 0
    for begin, end in sorted(ranges):
        assert begin == cursor
        cursor = end
    assert cursor == total_bytes
    assert len(data) >= 8 + size + total_bytes
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixture-helper", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.binary, args.fixture_helper):
        if not path.is_file():
            raise AssertionError(f"required binary is missing: {path}")
    with tempfile.TemporaryDirectory(prefix="microllm-gate-up-export-") as temporary:
        root = Path(temporary)
        fixture = root / "fixture"
        subprocess.run([str(args.fixture_helper), str(fixture)], check=True,
                       capture_output=True, text=True)
        gradients = root / "gradients.safetensors"
        parameters = root / "parameters.safetensors"
        base = [
            str(args.binary), "--config", str(fixture / "config.json"),
            "--weights", str(fixture / "model.safetensors"),
            "--tokens", "1,2,3", "--device", "cpu",
            "--learning-rate", "0.00001", "--warmup", "0", "--steps", "1",
            "--batch", "1", "--linear-precision", "fp32",
        ]
        completed = subprocess.run(base + [
            "--gate-up-gradients-output", str(gradients),
            "--gate-up-parameters-output", str(parameters),
        ], check=True, capture_output=True, text=True)
        record = json.loads(completed.stdout)
        assert record["measurement_profile"] == "diagnostic"
        assert record["gate_up_gradient_tensors"] == 2
        assert record["gate_up_parameter_tensors"] == 2
        assert record["gate_up_gradient_elements"] == 256
        assert record["gate_up_parameter_elements"] == 256
        header(gradients, GATE_UP_SHAPES)
        header(parameters, GATE_UP_SHAPES)

        all_gradients = root / "all-gradients.safetensors"
        all_parameters = root / "all-parameters.safetensors"
        completed = subprocess.run(base + [
            "--all-gradients-output", str(all_gradients),
            "--all-parameters-output", str(all_parameters),
        ], check=True, capture_output=True, text=True)
        record = json.loads(completed.stdout)
        assert record["measurement_profile"] == "diagnostic"
        assert record["all_gradient_tensors"] == 14
        assert record["all_parameter_tensors"] == 14
        assert record["all_gradient_elements"] == 680
        assert record["all_parameter_elements"] == 680
        header(all_gradients, ALL_SHAPES)
        header(all_parameters, ALL_SHAPES)

        rejected = subprocess.run(base + [
            "--warmup", "1", "--gate-up-gradients-output",
            str(root / "rejected.safetensors"),
        ], capture_output=True, text=True)
        assert rejected.returncode != 0
        assert "requires warmup 0 and steps 1" in rejected.stderr

        rejected = subprocess.run(base + [
            "--gate-up-gradients-output", str(root / "gate.safetensors"),
            "--all-parameters-output", str(root / "all.safetensors"),
        ], capture_output=True, text=True)
        assert rejected.returncode != 0
        assert "mutually exclusive" in rejected.stderr
    print("training gate/up export: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
