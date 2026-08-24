#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/compare_fp32_attention_solutions.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json,struct,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2]));qk='--fp32-attention-qk-solution-index' in a;pv='--fp32-attention-pv-solution-index' in a;candidate=qk or pv;registered=int(qk)+int(pv)
with open(a['--logits-output'],'wb') as f:f.write(struct.pack('4f',1.0,2.0,3.0,4.0))
print(json.dumps({'status':'pass','prefill_tokens_per_second':105.0 if candidate else 100.0,
'engine_peak_bytes':1000,'engine_allocation_calls':20,
'fp32_solution_registered_entries':registered,
'fp32_solution_cached_algorithms':registered,
'fp32_solution_dispatches':10 if candidate else 0}))
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
                {"name": "qwen2.5-0.5b", "revision": "qwen-revision",
                 "config": str(config), "weights": str(weights),
                 "inference": {"token_ids": [1, 2]}},
                {"name": "deepseek-r1-distill-qwen-1.5b",
                 "revision": "deepseek-revision", "config": str(config),
                 "weights": str(weights), "inference": {"token_ids": [3, 4]}},
            ],
        }), encoding="utf-8")
        output = root / "output"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--manifest", str(manifest),
            "--binary", str(fake), "--output-directory", str(output),
            "--runs", "2", "--warmup", "0", "--steps", "1",
            "--sequence", "256"], text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "pass"
        assert summary["raw_processes"] == 16
        assert summary["correctness_gate"] is True
        assert summary["performance_gate"] is True
        assert summary["memory_gate"] is True
        assert summary["keep_default"] is True
        assert summary["keep_policies"] == ["qk", "pv", "both"]
        assert len(summary["comparisons"]) == 6
        assert all(row["candidate_speedup"] == 1.05
                   for row in summary["comparisons"])
        assert all(row["maximum_absolute_logit_difference"] == 0
                   for row in summary["comparisons"])
    print("FP32 Attention official-model solution contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
