#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/compare_bf16_ffn_arena_models.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config = root / "config.json"
        weights = root / "weights.bin"
        config.write_text("{}", encoding="utf-8")
        weights.write_bytes(b"weights")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "models": [
            {"name": "qwen", "revision": "a", "config": str(config),
             "weights": str(weights), "inference": {"token_ids": [1, 2],
             "new_tokens": 2, "expected_generated_tokens": [3, 4]}},
            {"name": "deepseek", "revision": "b", "config": str(config),
             "weights": str(weights), "inference": {"token_ids": [5, 6],
             "new_tokens": 2, "expected_generated_tokens": [3, 4]}},
        ]}), encoding="utf-8")
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import array,json,pathlib,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2]));arena=a['--bf16-ffn-arena']=='true';batch=int(a['--batch']);tokens=len(a['--tokens'].split(','));rows=tokens*batch if a['--workload']=='prefill' else batch;minimum=int(a.get('--bf16-ffn-arena-minimum-rows','1'));eligible=arena and rows>=minimum
pathlib.Path(a['--logits-output']).write_bytes(array.array('f',[1.0,2.0]).tobytes())
print(json.dumps({'status':'pass','bf16_ffn_arena_enabled':arena,
'generated_tokens':[3,4] if int(a['--new-tokens']) else [],
'prefill_tokens_per_second':110.0 if eligible else 100.0,
'decode_tokens_per_second':110.0 if eligible else 100.0,
'engine_allocation_calls':80 if eligible else 100,'engine_peak_bytes':1000,
'bf16_ffn_arena_capacity_bytes':256 if eligible else 0,
'bf16_ffn_arena_entries':1 if eligible else 0,'bf16_ffn_arena_hits':9 if eligible else 0,
'bf16_ffn_arena_misses':1 if eligible else 0,'bf16_ffn_arena_eligible_calls':10 if eligible else 0,
'bf16_ffn_arena_bypassed_calls':0 if eligible else (10 if arena else 0)}))
""", encoding="utf-8")
        os.chmod(fake, 0o755)
        output = root / "output"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--manifest", str(manifest),
            "--binary", str(fake), "--output-directory", str(output),
            "--runs", "1", "--warmup", "0", "--steps", "1"],
            text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "pass"
        assert summary["raw_processes"] == 20
        assert len(summary["comparisons"]) == 10
        assert summary["keep_rows"] == 10
        assert summary["regression_rows"] == 0
        assert summary["decision"] == "keep universal model Arena"
        selective = root / "selective"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--manifest", str(manifest),
            "--binary", str(fake), "--output-directory", str(selective),
            "--runs", "1", "--warmup", "0", "--steps", "1",
            "--arena-minimum-rows", "512"],
            text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        selective_summary = json.loads(
            (selective / "summary.json").read_text(encoding="utf-8"))
        assert selective_summary["eligible_rows"] == 2
        assert selective_summary["bypassed_rows"] == 8
        assert selective_summary["keep_rows"] == 2
        assert selective_summary["decision"] == \
            "keep rows>=512 selective model Arena"
    print("BF16 FFN Arena model runner contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
