#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/rocwmma_online_attention_matrix.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake_online.py"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import json,sys\n"
            "a=dict(zip(sys.argv[1::2],sys.argv[2::2]));t=int(a['--sequence']);h=int(a['--heads']);kv=int(a['--kv-heads']);d=int(a['--width']);o=.5;c=o*1.5;s=o*(2 if t>=1024 else .5)\n"
            "print(json.dumps({'schema_version':1,'status':'pass','accuracy_passed':True,'architecture':'gfx942','rocwmma_version':'test','sequence':t,'heads':h,'kv_heads':kv,'width':d,'worker_threads':512,'pv_path':'rocwmma_bf16','global_score_bytes':0,'current_score_bytes':h*t*t*4,'complete_output_elements':h*t*d,'online_max_error':5e-4,'online_rms_error':5e-5,'scalar_max_error':0,'scalar_rms_error':0,'current_max_error':1e-7,'current_rms_error':1e-8,'online_event_ms_p50':o,'online_event_ms_p95':o,'scalar_event_ms_p50':s,'scalar_event_ms_p95':s,'current_event_ms_p50':c,'current_event_ms_p95':c}))\n",
            encoding="utf-8")
        os.chmod(fake, 0o755)
        output = root / "output"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--binary", str(fake),
            "--output-directory", str(output), "--runs", "2",
            "--warmup", "0", "--repetitions", "2",
        ], text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text())
        verification = json.loads((output / "verification.json").read_text())
        raw = [json.loads(line) for line in (output / "raw.jsonl").read_text().splitlines()]
        assert summary["status"] == "pass"
        assert summary["processes"] == 28
        assert len(summary["comparisons"]) == 14
        assert summary["correctness_gate"] is True
        assert summary["current_performance_gate"] is True
        assert summary["long_scalar_gate"] is True
        assert summary["short_scalar_counterexample"] is True
        assert summary["memory_gate"] is True
        assert summary["operator_integration_admitted"] is True
        assert summary["model_route_accepted"] is False
        assert verification["all_complete_outputs"] is True
        assert len(raw) == 28
    print("rocWMMA online Attention matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
