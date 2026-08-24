#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/compare_bf16_grouped_shape_models.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json,struct,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2]));both='--bf16-grouped-qkv-algorithm-index' in a;batch=int(a['--batch']);deep='deepseek' in a['--config'];blocks=28 if deep else 24;f=7
vals=[float(i) for _ in range(batch) for i in range(4)]
with open(a['--logits-output'],'wb') as o:o.write(struct.pack(f'{len(vals)}f',*vals))
print(json.dumps({'status':'pass','prefill_tokens_per_second':110 if both else 100,'engine_peak_bytes':1005 if both else 1000,'bf16_grouped_qkv_dispatches':blocks*f if both else 0,'bf16_grouped_gate_up_dispatches':blocks*f if both else 0,'bf16_grouped_qkv_kernel_setup_ms':200 if both else 0,'bf16_grouped_gate_up_kernel_setup_ms':1 if both else 0}))
""", encoding="utf-8")
        os.chmod(fake, 0o755)
        weights = root / "model.safetensors"
        weights.write_bytes(b"fixture")
        models = []
        for name, revision, token in (
                ("qwen2.5-0.5b", "q", 1),
                ("deepseek-r1-distill-qwen-1.5b", "d", 2)):
            config = root / f"{name}.json"
            config.write_text(json.dumps({"vocab_size": 4}),
                              encoding="utf-8")
            models.append({
                "name": name, "revision": revision,
                "config": str(config), "weights": str(weights),
                "inference": {"token_ids": [token]},
            })
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": 1, "models": models,
        }), encoding="utf-8")
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
        assert summary["raw_processes"] == 24
        assert summary["correctness_gate"] is True
        assert summary["performance_gate"] is True
        assert summary["memory_gate"] is True
        assert summary["setup_gate"] is True
        assert len(summary["comparisons"]) == 6
        assert all(row["top_rows_equal"] is True
                   for row in summary["comparisons"])
    print("BF16 grouped shape model contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
