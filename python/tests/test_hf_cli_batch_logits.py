#!/usr/bin/env python3
"""Verify that hf_infer exports one last-logit row for every batch item."""

from __future__ import annotations

import argparse
import array
import json
import subprocess
import tempfile
from pathlib import Path


def floats(path: Path) -> array.array:
    values = array.array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // values.itemsize)
    return values


def run(binary: Path, fixture: Path, batch: int,
        mode: str, output: Path) -> tuple[array.array, dict]:
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
    record = json.loads(completed.stdout.splitlines()[-1])
    assert record["cached_attention_materialized_policy"] == "auto-bypass"
    assert record["cached_attention_materialized_auto_eligible"] is False
    assert record["cached_attention_materialized_scores"] is False
    assert record["cached_attention_materialized_minimum_sequence"] == 2048
    return floats(output), record


def cache_export(binary: Path, fixture: Path, output: Path) -> tuple[dict, bytes]:
    completed = subprocess.run([
        str(binary), "--config", str(fixture / "config.json"),
        "--weights", str(fixture / "model.safetensors"),
        "--tokens", "1,2", "--device", "cpu", "--top-k", "1",
        "--batch", "2", "--workload", "decode", "--new-tokens", "1",
        "--use-cache", "true", "--cache-prefill-mode", "full",
        "--kv-cache-dtype", "bf16", "--cache-capacity", "3",
        "--decode-mode", "steady", "--warmup", "0", "--steps", "1",
        "--prefill-warmup", "0", "--prefill-steps", "1",
        "--prefill-cache-output", str(output), "--prefill-cache-layer", "0",
    ], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise AssertionError(completed.stdout + completed.stderr)
    record = json.loads(completed.stdout.splitlines()[-1])
    with output.open("rb") as stream:
        header = json.loads(stream.readline())
        payload = stream.read()
    assert header == {
        "schema_version": 1, "record_type": "prefill_kv_cache",
        "layer": 0, "dtype": "bfloat16", "shape": [2, 1, 2, 4],
        "key_bytes": 32, "value_bytes": 32,
    }
    assert record["prefill_cache_exported"] is True
    assert record["prefill_cache_layer"] == 0
    assert record["prefill_cache_dtype"] == "bfloat16"
    assert record["prefill_cache_shape"] == [2, 1, 2, 4]
    assert record["prefill_cache_key_bytes"] == 32
    assert record["prefill_cache_value_bytes"] == 32
    return header, payload


def forced_decode(binary: Path, fixture: Path, output: Path) -> None:
    base = [
        str(binary), "--config", str(fixture / "config.json"),
        "--weights", str(fixture / "model.safetensors"),
        "--tokens", "1,2", "--device", "cpu", "--top-k", "1",
        "--batch", "2", "--workload", "decode", "--new-tokens", "2",
        "--use-cache", "true", "--cache-prefill-mode", "full",
        "--kv-cache-dtype", "fp32", "--cache-capacity", "4",
        "--decode-mode", "steady", "--warmup", "0", "--steps", "1",
        "--prefill-warmup", "0", "--prefill-steps", "1",
        "--cache-logits-output", str(output), "--cache-logits-step", "1",
    ]
    completed = subprocess.run(
        base + ["--forced-decode-inputs", "3,4"],
        text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise AssertionError(completed.stdout + completed.stderr)
    record = json.loads(completed.stdout.splitlines()[-1])
    assert record["forced_decode_inputs"] is True
    assert record["forced_decode_input_count"] == 2
    assert record["decode_mode"] == "steady"
    assert len(record["generated_tokens"]) == 2
    assert len(floats(output)) == 16

    rejected = subprocess.run(
        base + ["--forced-decode-inputs", "3,8"],
        text=True, capture_output=True, check=False)
    assert rejected.returncode != 0
    assert "must contain in-vocabulary IDs" in rejected.stderr

    scoped_output = output.with_name("forced-gate-only-logits.bin")
    scoped_base = list(base)
    scoped_base[scoped_base.index(str(output))] = str(scoped_output)
    scoped = subprocess.run(
        scoped_base + [
            "--forced-decode-inputs", "3,4", "--bf16-ffn", "true",
            "--bf16-ffn-weight-scope", "gate-only",
        ], text=True, capture_output=True, check=False)
    if scoped.returncode != 0:
        raise AssertionError(scoped.stdout + scoped.stderr)
    scoped_record = json.loads(scoped.stdout.splitlines()[-1])
    assert scoped_record["bf16_ffn_weight_scope"] == "gate-only"
    assert scoped_record["bf16_ffn_converted_tensors"] == 1
    assert len(floats(scoped_output)) == 16

    phase_output = output.with_name("forced-decode-up-fp32-logits.bin")
    phase_base = list(base)
    phase_base[phase_base.index(str(output))] = str(phase_output)
    phase = subprocess.run(
        phase_base + [
            "--forced-decode-inputs", "3,4", "--bf16-ffn", "true",
            "--bf16-ffn-decode-up-fp32", "true",
        ], text=True, capture_output=True, check=False)
    if phase.returncode != 0:
        raise AssertionError(phase.stdout + phase.stderr)
    phase_record = json.loads(phase.stdout.splitlines()[-1])
    assert phase_record["bf16_ffn_weight_scope"] == "all"
    assert phase_record["bf16_ffn_decode_up_fp32"] is True
    assert phase_record["bf16_ffn_converted_tensors"] == 2
    assert phase_record["bf16_ffn_fp32_decode_tensors_retained"] == 1
    assert phase_record["bf16_ffn_fp32_decode_bytes_retained"] == 512
    assert phase_record["bf16_ffn_bf16_prefill_mirror_tensors"] == 1
    assert phase_record["bf16_ffn_bf16_prefill_mirror_bytes_retained"] == 256
    assert phase_record["bf16_weight_bytes_retained"] == 768
    assert phase_record["inference_weight_policy"] == \
        "dual_representation_bf16_prefill_decode_up_fp32"
    assert len(floats(phase_output)) == 16


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
        batch_one, _ = run(
            args.binary, fixture, 1, "last", root / "b1.bin")
        batch_last, _ = run(
            args.binary, fixture, 2, "last", root / "b2-last.bin")
        batch_full, _ = run(
            args.binary, fixture, 2, "full", root / "b2-full.bin")
        assert len(batch_one) == 8
        assert len(batch_last) == 16
        assert len(batch_full) == 16
        assert batch_last == batch_full
        assert batch_last[:8] == batch_one
        assert batch_last[8:] == batch_one
        header, payload = cache_export(
            args.binary, fixture, root / "prefill-cache.bin")
        row_bytes = header["key_bytes"] // header["shape"][0]
        key = payload[:header["key_bytes"]]
        value = payload[header["key_bytes"]:]
        assert len(payload) == header["key_bytes"] + header["value_bytes"]
        assert key[:row_bytes] == key[row_bytes:]
        assert value[:row_bytes] == value[row_bytes:]
        forced_decode(args.binary, fixture, root / "forced-logits.bin")
    print("hf_infer batch logits export: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
