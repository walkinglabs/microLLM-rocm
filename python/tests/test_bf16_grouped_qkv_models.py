#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/compare_bf16_grouped_qkv_models.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json,struct,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2]));grouped='--bf16-grouped-qkv-algorithm-index' in a
values=[1.0,2.0,3.0,4.0] if not grouped else [1.001,2.0,3.0,4.0]
with open(a['--logits-output'],'wb') as f:f.write(struct.pack('4f',*values))
print(json.dumps({'status':'pass','prefill_tokens_per_second':102.0 if grouped else 100.0,
'engine_peak_bytes':1003 if grouped else 1000,'engine_allocation_calls':8 if grouped else 10,
'bf16_grouped_qkv_registered_entries':1 if grouped else 0,
'bf16_grouped_qkv_algorithm_entries':1 if grouped else 0,
'bf16_grouped_qkv_kernel_entries':1 if grouped else 0,
'bf16_grouped_qkv_plan_entries':2 if grouped else 0,'bf16_grouped_qkv_plan_hits':4 if grouped else 0,
'bf16_grouped_qkv_plan_misses':2 if grouped else 0,'bf16_grouped_qkv_dispatches':6 if grouped else 0,
'bf16_grouped_qkv_kernel_setup_ms':0.5 if grouped else 0,
'bf16_grouped_qkv_argument_setup_ms':0.1 if grouped else 0,
'top_logits':[{'token':7,'logit':4.0}]}))
""", encoding="utf-8")
        os.chmod(fake, 0o755)
        config = root / "config.json"
        weights = root / "model.safetensors"
        config.write_text("{}\n", encoding="utf-8")
        weights.write_bytes(b"fixture")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "models": [
                {"name": "qwen2.5-0.5b", "revision": "qwen",
                 "config": str(config), "weights": str(weights),
                 "inference": {"token_ids": [1, 2]}},
                {"name": "deepseek-r1-distill-qwen-1.5b", "revision": "deep",
                 "config": str(config), "weights": str(weights),
                 "inference": {"token_ids": [3, 4]}},
            ],
        }), encoding="utf-8")
        output = root / "output"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--manifest", str(manifest),
            "--binary", str(fake), "--output-directory", str(output),
            "--runs", "2", "--warmup", "0", "--steps", "1",
            "--maximum-absolute-tolerance", "0.01", "--rms-tolerance", "0.01"],
            text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "pass"
        assert summary["raw_processes"] == 8
        assert summary["correctness_gate"] is True
        assert summary["performance_gate"] is True
        assert summary["memory_gate"] is True
        assert summary["setup_gate"] is True
        assert summary["keep_steady_policy"] is True
        assert summary["keep_default"] is True
        assert len(summary["comparisons"]) == 2
        assert all(row["grouped_speedup"] == 1.02
                   for row in summary["comparisons"])
    print("BF16 grouped QKV official-model contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
