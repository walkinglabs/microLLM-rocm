#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/compare_bf16_grouped_qkv_prewarm.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json,struct,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2]));grouped='--bf16-grouped-qkv-algorithm-index' in a;prewarm=a.get('--bf16-grouped-qkv-prewarm')=='true';blocks=24
with open(a['--logits-output'],'wb') as f:f.write(struct.pack('4f',1,2,3,4))
print(json.dumps({'status':'pass','forward_ms':5 if prewarm else 205 if grouped else 6,
'bf16_grouped_qkv_plan_hits':blocks if prewarm else 0,
'bf16_grouped_qkv_dispatches':2*blocks if prewarm else blocks if grouped else 0,
'bf16_grouped_qkv_prewarm_ms':200 if prewarm else 0,
'bf16_grouped_qkv_prewarm_kernel_ms':199 if prewarm else 0,
'bf16_grouped_qkv_prewarm_arguments_ms':1 if prewarm else 0}))
""", encoding="utf-8")
        os.chmod(fake, 0o755)
        config = root / "config.json"
        weights = root / "model.safetensors"
        config.write_text("{}", encoding="utf-8")
        weights.write_bytes(b"fixture")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "models": [
            {"name": "qwen2.5-0.5b", "revision": "q", "config": str(config),
             "weights": str(weights), "inference": {"token_ids": [1]}},
            {"name": "deepseek-r1-distill-qwen-1.5b", "revision": "d",
             "config": str(config), "weights": str(weights),
             "inference": {"token_ids": [2]}},
        ]}), encoding="utf-8")
        # The fake uses 24 blocks for both; use a wrapper manifest model name rewrite
        # is deliberately avoided by accepting the DeepSeek 28-block contract below.
        fake_text = fake.read_text(encoding="utf-8").replace(
            "blocks=24", "blocks=28 if 'deepseek' in a['--config'] else 24")
        # Both fixtures share one path, so create model-specific config paths.
        qconfig = root / "qwen-config.json"
        dconfig = root / "deepseek-config.json"
        qconfig.write_text("{}", encoding="utf-8")
        dconfig.write_text("{}", encoding="utf-8")
        document = json.loads(manifest.read_text(encoding="utf-8"))
        document["models"][0]["config"] = str(qconfig)
        document["models"][1]["config"] = str(dconfig)
        manifest.write_text(json.dumps(document), encoding="utf-8")
        fake.write_text(fake_text, encoding="utf-8")
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--manifest", str(manifest),
            "--binary", str(fake), "--output-directory", str(root / "out"),
            "--runs", "2"], text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((root / "out/summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "pass"
        assert summary["raw_processes"] == 12
        assert summary["correctness_gate"] is True
        assert summary["setup_moved_before_request"] is True
        assert all(row["prewarmed_first_ms"] == 5
                   for row in summary["comparisons"])
    print("BF16 grouped QKV prewarm contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
