#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/compare_bf16_exact_startup.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json,struct,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2]))
if '--config' not in a:
 i=int(a['--inner']);base=20 if i==8 else 30
 c=[{'index':base,'correctness_passed':True,'event_ms_p50':2.0},
    {'index':base+1,'correctness_passed':True,'event_ms_p50':1.0}]
 print(json.dumps({'status':'pass','default_event_ms_p50':2.0,
  'candidates':c,'passing_candidates':2,'candidate_count':2}))
else:
 exact='--bf16-algorithm-index' in a;cold=a['--prefill-warmup']=='0'
 with open(a['--logits-output'],'wb') as f:f.write(struct.pack('4f',1,2,3,4))
 print(json.dumps({'status':'pass','forward_ms':12 if exact else 10,
  'prefill_tokens_per_second':90 if exact else 100,
  'engine_peak_bytes':200,'load_ms':1,'weight_preparation_ms':1}))
""", encoding="utf-8")
        os.chmod(fake, 0o755)
        weights = root / "model.safetensors"
        weights.write_bytes(b"fixture")
        models = []
        for name, revision, hidden, intermediate, token in (
                ("qwen2.5-0.5b", "q", 8, 16, 1),
                ("deepseek-r1-distill-qwen-1.5b", "d", 12, 24, 2)):
            config = root / f"{name}.json"
            config.write_text(json.dumps({
                "hidden_size": hidden,
                "intermediate_size": intermediate,
            }), encoding="utf-8")
            models.append({
                "name": name, "revision": revision,
                "config": str(config), "weights": str(weights),
                "inference": {"token_ids": [token]},
            })
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": 1, "models": models,
        }), encoding="utf-8")
        output = root / "out"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--manifest", str(manifest),
            "--binary", str(fake), "--tuner", str(fake),
            "--output-directory", str(output), "--runs", "2",
        ], text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text(
            encoding="utf-8"))
        assert summary["status"] == "pass"
        assert summary["tuner_processes"] == 4
        assert summary["model_processes"] == 16
        assert summary["correctness_gate"] is True
        assert summary["memory_gate"] is True
        assert summary["performance_gate"] is False
        assert summary["decision"].startswith("reject")
        assert {row["selected_index"] for row in summary["comparisons"]} == {
            21, 31}
        assert all(row["operator_speedup"] == 2
                   for row in summary["comparisons"])
    print("BF16 exact startup contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
