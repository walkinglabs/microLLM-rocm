#!/usr/bin/env python3
"""Verify that hf_infer exports one last-logit row for every batch item."""

from __future__ import annotations

import argparse
import array
import subprocess
import tempfile
from pathlib import Path


def floats(path: Path) -> array.array:
    values = array.array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // values.itemsize)
    return values


def run(binary: Path, fixture: Path, batch: int,
        mode: str, output: Path) -> array.array:
    completed = subprocess.run([
        str(binary), "--config", str(fixture / "config.json"),
        "--weights", str(fixture / "model.safetensors"),
        "--tokens", "1,2", "--device", "cpu", "--top-k", "2",
        "--batch", str(batch), "--workload", "prefill",
        "--new-tokens", "0", "--warmup", "0", "--steps", "1",
        "--prefill-warmup", "0", "--prefill-steps", "1",
        "--prefill-logits", mode, "--logits-output", str(output),
    ], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise AssertionError(completed.stdout + completed.stderr)
    return floats(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--fixture-helper", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture = root / "fixture"
        completed = subprocess.run([
            str(args.fixture_helper), str(fixture),
        ], text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        batch_one = run(
            args.binary, fixture, 1, "last", root / "b1.bin")
        batch_last = run(
            args.binary, fixture, 2, "last", root / "b2-last.bin")
        batch_full = run(
            args.binary, fixture, 2, "full", root / "b2-full.bin")
        assert len(batch_one) == 8
        assert len(batch_last) == 16
        assert len(batch_full) == 16
        assert batch_last == batch_full
        assert batch_last[:8] == batch_one
        assert batch_last[8:] == batch_one
    print("hf_infer batch logits export: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
