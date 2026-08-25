#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/rocwmma_online_operator_matrix.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake_operator.py"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import json,sys\n"
            "a=dict(zip(sys.argv[1::2],sys.argv[2::2]));b=int(a['--batch']);h=int(a['--heads']);kv=int(a['--kv-heads']);t=int(a['--sequence']);d=int(a['--width']);w=int(a['--warmup']);r=int(a['--repetitions']);native=t>=32 and t%32==0 and d in (64,128);c=.5;o=c/1.5 if native else c*1.5\n"
            "print(json.dumps({'schema_version':1,'status':'pass','accuracy_passed':True,'architecture':'gfx942','batch':b,'heads':h,'kv_heads':kv,'sequence':t,'width':d,'native_expected':native,'native_calls':w+r if native else 0,'fallback_calls':0 if native else w+r,'complete_output_elements':b*h*t*d,'candidate_global_score_bytes':0,'current_score_bytes':b*h*t*t*4,'candidate_max_error':5e-4 if native else 1e-7,'candidate_rms_error':5e-5 if native else 1e-8,'current_max_error':1e-7,'current_rms_error':1e-8,'candidate_event_ms_p50':o,'candidate_event_ms_p95':o,'candidate_wall_ms_p50':o,'current_event_ms_p50':c,'current_event_ms_p95':c,'current_wall_ms_p50':c,'candidate_over_current':c/o,'candidate_h2d_calls':0,'candidate_d2h_calls':0}))\n",
            encoding="utf-8")
        os.chmod(fake, 0o755)
        output = root / "output"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--binary", str(fake),
            "--output-directory", str(output), "--runs", "2",
            "--warmup", "1", "--repetitions", "2",
        ], text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text())
        verification = json.loads((output / "verification.json").read_text())
        assert summary["status"] == "pass"
        assert summary["processes"] == 28
        assert summary["native_cases"] == 10
        assert summary["fallback_cases"] == 4
        assert summary["native_performance_gate"] is True
        assert summary["routing_gate"] is True
        assert summary["fallback_counterexample"] is True
        assert summary["model_gate_admitted"] is True
        assert summary["model_route_accepted"] is False
        assert verification["all_complete_outputs"] is True
    print("rocWMMA public online operator matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
