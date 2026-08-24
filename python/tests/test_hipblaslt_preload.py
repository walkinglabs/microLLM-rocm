#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/compare_hipblaslt_preload.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json,os,struct,sys,time
a=dict(zip(sys.argv[1::2],sys.argv[2::2]));bf16=a.get('--bf16-ffn')=='true';preload=os.environ.get('HIPBLASLT_PRELOAD_KERNELS')=='1'
if preload: time.sleep(.04)
with open(a['--logits-output'],'wb') as f:f.write(struct.pack('4f',1,2,3,4))
print(json.dumps({'status':'pass','forward_ms':40 if preload else 10 if bf16 else 8,
'engine_peak_bytes':200 if bf16 else 300,'load_ms':1,'weight_preparation_ms':1}))
""", encoding="utf-8")
        os.chmod(fake, 0o755)
        weights = root / "model.safetensors"
        weights.write_bytes(b"fixture")
        qconfig = root / "qwen-config.json"
        dconfig = root / "deepseek-config.json"
        qconfig.write_text("{}", encoding="utf-8")
        dconfig.write_text("{}", encoding="utf-8")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "models": [
            {"name": "qwen2.5-0.5b", "revision": "q",
             "config": str(qconfig), "weights": str(weights),
             "inference": {"token_ids": [1]}},
            {"name": "deepseek-r1-distill-qwen-1.5b", "revision": "d",
             "config": str(dconfig), "weights": str(weights),
             "inference": {"token_ids": [2]}},
        ]}), encoding="utf-8")
        output = root / "out"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--manifest", str(manifest),
            "--binary", str(fake), "--output-directory", str(output),
            "--runs", "2", "--minimum-preload-slowdown", "1.2",
        ], text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "pass"
        assert summary["raw_processes"] == 12
        assert summary["correctness_gate"] is True
        assert summary["preload_counterexample_gate"] is True
        assert all(row["preload_forward_slowdown"] == 4
                   for row in summary["comparisons"])
        raw = [json.loads(line) for line in
               (output / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
        assert {row["hipblaslt_preload_kernels"] for row in raw} == {0, 1}
    print("hipBLASLt preload contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
