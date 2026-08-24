#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/compare_inference_bthd_bf16_qk.py"
PROFILE_SUMMARIZER = ROOT / (
    "benchmarks/single_gpu/summarize_inference_bthd_bf16_qk_profile.py")


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json,struct,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2]));q=a['--inference-bthd-bf16-qk']=='true';deep='deepseek' in a['--config'];blocks=28 if deep else 24;n=blocks*(int(a['--prefill-warmup'])+int(a['--prefill-steps']))
with open(a['--logits-output'],'wb') as o:o.write(struct.pack('4f',1,2,3,4))
print(json.dumps({'status':'pass','inference_bthd_attention':True,'inference_bthd_bf16_qk':q,'bf16_grouped_qkv_dispatches':n,'bf16_grouped_qkv_retained_query_key_dispatches':n if q else 0,'prefill_tokens_per_second':102 if q else 100,'engine_peak_bytes':900 if q else 1000}))
""", encoding="utf-8")
        os.chmod(fake, 0o755)
        weights = root / "model.safetensors"
        weights.write_bytes(b"fixture")
        manifest = root / "manifest.json"
        models = []
        for name, revision in (("qwen2.5-0.5b", "q"),
                               ("deepseek-r1-distill-qwen-1.5b", "d")):
            config = root / f"{name}-config.json"
            config.write_text("{}", encoding="utf-8")
            models.append({"name": name, "revision": revision,
                           "config": str(config), "weights": str(weights),
                           "inference": {"token_ids": [1, 2]}})
        manifest.write_text(json.dumps({"schema_version": 1, "models": models}),
                            encoding="utf-8")
        output = root / "output"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--manifest", str(manifest),
            "--binary", str(fake), "--output-directory", str(output),
            "--runs", "2",
        ], text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "pass"
        assert summary["processes"] == 8
        assert summary["correctness_gate"] is True
        assert summary["routing_gate"] is True
        assert summary["performance_gate"] is True
        assert summary["memory_gate"] is True
        profile_root = root / "profile"
        for model, blocks in (("qwen2.5-0.5b", 24),
                              ("deepseek-r1-distill-qwen-1.5b", 28)):
            for policy, calls, total in (
                    ("fp32-boundary", blocks * 3, 1000),
                    ("bf16-qk", blocks, 900)):
                directory = profile_root / model / policy
                directory.mkdir(parents=True)
                document = {
                    "total_kernel_ns_per_step": total,
                    "categories": [{
                        "category": "FP32/BF16 cast",
                        "calls_per_step": calls,
                        "duration_ns_per_step": calls * 10,
                    }],
                    "top_kernels": [{
                        "name": "rope_split_half_bias_bthd_kernel",
                        "duration_ns_per_step": 50,
                    }],
                }
                (directory / "profile-delta.json").write_text(
                    json.dumps(document), encoding="utf-8")
        profile_summary = root / "profile-summary.json"
        profiled = subprocess.run([
            sys.executable, str(PROFILE_SUMMARIZER),
            "--performance-summary", str(output / "summary.json"),
            "--profile-root", str(profile_root),
            "--output", str(profile_summary),
        ], text=True, capture_output=True, check=False)
        if profiled.returncode != 0:
            raise AssertionError(profiled.stdout + profiled.stderr)
        profile = json.loads(profile_summary.read_text(encoding="utf-8"))
        assert profile["status"] == "pass"
        assert profile["cast_elimination_gate"] is True
        assert profile["kernel_performance_gate"] is True
    print("inference BTHD BF16 Q/K contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
