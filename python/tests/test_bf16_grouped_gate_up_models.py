#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/compare_bf16_grouped_gate_up_models.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json,struct,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2]));grouped='--bf16-grouped-gate-up-algorithm-index' in a;deep='deepseek' in a['--config'];blocks=28 if deep else 24;f=7
with open(a['--logits-output'],'wb') as o:o.write(struct.pack('4f',1,2,3,4))
p='bf16_grouped_gate_up_';r={'status':'pass','prefill_tokens_per_second':110 if grouped else 100,'engine_peak_bytes':1000}
r.update({p+'registered_entries':1 if grouped else 0,p+'kernel_entries':1 if grouped else 0,p+'kernel_misses':1 if grouped else 0,p+'kernel_hits':blocks-1 if grouped else 0,p+'plan_entries':blocks if grouped else 0,p+'plan_misses':blocks if grouped else 0,p+'plan_hits':blocks*(f-1) if grouped else 0,p+'dispatches':blocks*f if grouped else 0,p+'kernel_setup_ms':50 if grouped else 0,p+'argument_setup_ms':1 if grouped else 0})
print(json.dumps(r))
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
        assert summary["raw_processes"] == 8
        assert summary["correctness_gate"] is True
        assert summary["performance_gate"] is True
        assert summary["memory_gate"] is True
        assert summary["setup_gate"] is True
        assert all(row["grouped_speedup"] == 1.1
                   for row in summary["comparisons"])
    print("BF16 grouped gate/up model contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
