#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/hf_strided_copy_sources.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2]));deep='deepseek' in a['--config'];layers=28 if deep else 24
r=[{'source':'attention.layout','device':'hip:0','element_bytes':4,'calls':3*layers,'elements':10,'bytes':30,'shape':[1,2,3,4],'strides':[24,4,8,1]},{'source':'attention.core','device':'hip:0','element_bytes':4,'calls':layers,'elements':5,'bytes':20,'shape':[1,3,2,4],'strides':[24,4,12,1]}]
print(json.dumps({'status':'pass','strided_copy_diagnostics':True,'strided_copy_calls':4*layers,'strided_copy_bytes':50,'strided_copy_records':r,'bf16_grouped_qkv_dispatches':layers,'bf16_grouped_gate_up_dispatches':layers}))
""", encoding="utf-8")
        os.chmod(fake, 0o755)
        config = root / "config.json"
        weights = root / "model.safetensors"
        config.write_text("{}", encoding="utf-8")
        weights.write_bytes(b"fixture")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "models": [
            {"name": "qwen2.5-0.5b", "revision": "q",
             "config": str(config), "weights": str(weights),
             "inference": {"token_ids": [1]}},
            {"name": "deepseek-r1-distill-qwen-1.5b", "revision": "d",
             "config": str(config), "weights": str(weights),
             "inference": {"token_ids": [2]}},
        ]}), encoding="utf-8")
        output = root / "output"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--manifest", str(manifest),
            "--binary", str(fake), "--output-directory", str(output),
            "--runs", "2",
        ], text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text(
            encoding="utf-8"))
        assert summary["status"] == "pass"
        assert summary["raw_processes"] == 4
        assert summary["attribution_gate"] is True
        assert all(set(row["source_totals"]) ==
                   {"attention.core", "attention.layout"}
                   for row in summary["comparisons"])
    print("HF strided-copy source contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
