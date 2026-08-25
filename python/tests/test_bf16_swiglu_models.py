#!/usr/bin/env python3
"""Contract test for the BF16 SwiGLU full-model gate runner."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/compare_bf16_swiglu_models.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config = root / "config.json"
        weights = root / "weights.bin"
        config.write_text("{}", encoding="utf-8")
        weights.write_bytes(b"x")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "models": [{
                "name": name, "revision": f"r{index}",
                "config": str(config), "weights": str(weights),
                "inference": {"token_ids": [index + 1, index + 2]},
            } for index, name in enumerate((
                "qwen2.5-0.5b", "deepseek-r1-distill-qwen-1.5b"))],
        }), encoding="utf-8")
        source = """#!/usr/bin/env python3
import array,json,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2])); candidate='candidate' in sys.argv[0]
v=array.array('f',[1.0,2.0,3.0])
with open(a['--logits-output'],'wb') as f: v.tofile(f)
print(json.dumps({'status':'pass','inference_bthd_attention':True,
'inference_bthd_bf16_qk':True,'inference_bthd_online_attention':False,
'bf16_grouped_gate_up_swish':a.get('--bf16-grouped-gate-up-swish','false')=='true',
'bf16_ffn_norm_fusion_enabled':a.get('--bf16-ffn-norm-fusion','false')=='true',
'prefill_tokens_per_second':101.0 if candidate else 100.0,
'engine_peak_bytes':1000,'engine_allocation_calls':20}))
"""
        baseline = root / "baseline.py"
        candidate = root / "candidate.py"
        baseline.write_text(source, encoding="utf-8")
        candidate.write_text(source, encoding="utf-8")
        baseline.chmod(0o755)
        candidate.chmod(0o755)
        output = root / "out"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--manifest", str(manifest),
            "--baseline-binary", str(baseline), "--candidate-binary", str(candidate),
            "--output-directory", str(output), "--runs", "2",
            "--warmup", "0", "--steps", "1"],
            text=True, capture_output=True, check=False)
        assert completed.returncode == 0, completed.stderr
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["raw_processes"] == 8
        assert summary["keep_default"] is True
        assert all(row["candidate_speedup"] == 1.01
                   for row in summary["comparisons"])
        assert all(row["maximum_absolute_logit_difference"] == 0.0
                   for row in summary["comparisons"])
        swish_output = root / "swish"
        swish = subprocess.run([
            sys.executable, str(RUNNER), "--manifest", str(manifest),
            "--baseline-binary", str(baseline), "--candidate-binary", str(candidate),
            "--output-directory", str(swish_output), "--runs", "1",
            "--warmup", "0", "--steps", "1", "--candidate-swish"],
            text=True, capture_output=True, check=False)
        assert swish.returncode == 0, swish.stderr
        swish_summary = json.loads(
            (swish_output / "summary.json").read_text(encoding="utf-8"))
        assert swish_summary["candidate_swish"] is True
        assert swish_summary["record_type"] == "bf16_grouped_swiglu_model_summary"
        norm_output = root / "norm"
        norm = subprocess.run([
            sys.executable, str(RUNNER), "--manifest", str(manifest),
            "--baseline-binary", str(baseline), "--candidate-binary", str(candidate),
            "--output-directory", str(norm_output), "--runs", "1",
            "--warmup", "0", "--steps", "1", "--candidate-bf16-norm"],
            text=True, capture_output=True, check=False)
        assert norm.returncode == 0, norm.stderr
        norm_summary = json.loads(
            (norm_output / "summary.json").read_text(encoding="utf-8"))
        assert norm_summary["candidate_bf16_norm"] is True
        assert norm_summary["record_type"] == "bf16_ffn_norm_fusion_model_summary"
    print("BF16 SwiGLU model gate contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
