#!/usr/bin/env python3

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/prepare_hf_fixture.py"


def write_safetensors(path: Path) -> None:
    header = {
        "first": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
        "second": {"dtype": "F16", "shape": [2], "data_offsets": [16, 20]},
    }
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + bytes(20))


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        (source / "config.json").write_text("{}\n", encoding="utf-8")
        (source / "vocab.json").write_text('{"a":0}\n', encoding="utf-8")
        (source / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
        write_safetensors(source / "model.safetensors")
        registry = root / "registry.toml"
        registry.write_text(
            """schema_version = 1
[[model]]
id = "fixture"
repo_id = "owner/model"
source = "https://example.invalid/owner/model"
revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
license = "MIT"
license_url = "https://example.invalid/license"
parameter_count = 6
tensor_count = 2
tokenizer_family = "test"
inference_token_ids = [1, 2]
expected_generated_tokens = [3]
training_tokens = "1,2"
""", encoding="utf-8")
        manifest = root / "manifest.json"
        evidence = root / "evidence.json"
        prepared = subprocess.run([
            sys.executable, str(TOOL), "prepare", "--registry", str(registry),
            "--manifest", str(manifest), "--evidence", str(evidence),
            "--model-source", f"fixture={source}",
        ], capture_output=True, text=True)
        assert prepared.returncode == 0, prepared.stderr
        document = json.loads(manifest.read_text(encoding="utf-8"))
        assert document["models"][0]["parameter_count"] == 6
        assert document["models"][0]["loaded_tensors"] == 2
        assert document["models"][0]["state"] == "fixture-ready"
        report = json.loads(evidence.read_text(encoding="utf-8"))
        assert report["status"] == "pass"
        assert report["models"][0]["weight_dtypes"] == ["F16", "F32"]

        validated = subprocess.run([
            sys.executable, str(TOOL), "validate", "--registry", str(registry),
            "--manifest", str(manifest),
        ], capture_output=True, text=True)
        assert validated.returncode == 0, validated.stderr

        (source / "merges.txt").unlink()
        rejected = subprocess.run([
            sys.executable, str(TOOL), "validate", "--registry", str(registry),
            "--manifest", str(manifest),
        ], capture_output=True, text=True)
        assert rejected.returncode != 0
        assert "required fixture file is missing" in rejected.stderr
    print("Hugging Face fixture tool contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
