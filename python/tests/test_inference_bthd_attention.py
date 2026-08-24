#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/compare_inference_bthd_attention.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json,struct,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2]));b=a.get('--inference-bthd-attention')=='true';diag=a.get('--strided-copy-diagnostics')=='true';deep='deepseek' in a['--config'];blocks=28 if deep else 24;warm=int(a['--prefill-warmup']);steps=int(a['--prefill-steps'])
if '--logits-output' in a:
 with open(a['--logits-output'],'wb') as o:o.write(struct.pack('4f',1,2,3,4))
print(json.dumps({'status':'pass','inference_bthd_attention':b,'prefill_tokens_per_second':110 if b else 100,'engine_peak_bytes':900 if b else 1000,'bf16_grouped_qkv_dispatches':blocks*(warm+steps),'bf16_grouped_gate_up_dispatches':blocks*(warm+steps),'strided_copy_calls':0 if b else blocks*4,'strided_copy_bytes':0 if b else blocks*100,'strided_copy_records':[] if b else [{'source':'attention.layout'}]}))
""", encoding="utf-8")
        os.chmod(fake, 0o755)
        qconfig = root / "qwen-config.json"
        dconfig = root / "deepseek-config.json"
        weights = root / "model.safetensors"
        qconfig.write_text("{}", encoding="utf-8")
        dconfig.write_text("{}", encoding="utf-8")
        weights.write_bytes(b"fixture")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "models": [
            {"name": "qwen2.5-0.5b", "revision": "q",
             "config": str(qconfig), "weights": str(weights),
             "inference": {"token_ids": [1]}},
            {"name": "deepseek-r1-distill-qwen-1.5b", "revision": "d",
             "config": str(dconfig), "weights": str(weights),
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
        assert summary["performance_processes"] == 8
        assert summary["diagnostic_processes"] == 8
        assert summary["correctness_gate"] is True
        assert summary["copy_elimination_gate"] is True
        assert summary["performance_gate"] is True
        assert summary["memory_gate"] is True
        assert all(row["bthd_strided_calls"] == 0
                   for row in summary["comparisons"])
    print("inference BTHD Attention contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
