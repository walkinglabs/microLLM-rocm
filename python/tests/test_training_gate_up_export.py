#!/usr/bin/env python3
"""Exercise diagnostic gate/up gradient and parameter safetensors on CPU."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path


NAMES = {
    "blocks.0.feed_forward.gate_proj.weight",
    "blocks.0.feed_forward.up_proj.weight",
}


def header(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) < 8:
        raise AssertionError("safetensors output is truncated")
    size = struct.unpack("<Q", data[:8])[0]
    document = json.loads(data[8:8 + size])
    document.pop("__metadata__", None)
    assert set(document) == NAMES
    ranges = []
    for value in document.values():
        assert value["dtype"] == "F32"
        assert value["shape"] == [8, 16]
        begin, end = value["data_offsets"]
        assert end - begin == 8 * 16 * 4
        ranges.append((begin, end))
    assert sorted(ranges) == [(0, 512), (512, 1024)]
    assert len(data) >= 8 + size + 1024
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
        header(gradients)
        header(parameters)

        rejected = subprocess.run(base + [
            "--warmup", "1", "--gate-up-gradients-output",
            str(root / "rejected.safetensors"),
        ], capture_output=True, text=True)
        assert rejected.returncode != 0
        assert "requires warmup 0 and steps 1" in rejected.stderr
    print("training gate/up export: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
