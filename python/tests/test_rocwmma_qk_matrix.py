#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/rocwmma_qk_matrix.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake_rocwmma.py"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import json,sys\n"
            "a=dict(zip(sys.argv[1::2],sys.argv[2::2]));s=int(a['--rows']);d=int(a['--inner']);tile=int(a['--tile'])\n"
            "r=.4 if s==512 else (.8 if s>=2048 else .5);scalar=r*10;blas=r*(1.25 if s==512 else (.625 if s>=2048 else 2))\n"
            "print(json.dumps({'status':'pass','accuracy_passed':True,'architecture':'gfx942','rocwmma_version':'test','rows':s,'columns':s,'inner':d,'tile':[tile,tile,16],'waves_per_block':1,'complete_output_elements':s*s,'rocwmma_max_error':0,'rocwmma_rms_error':0,'scalar_max_error':0,'scalar_rms_error':0,'hipblaslt_max_error':0,'hipblaslt_rms_error':0,'rocwmma_event_ms_p50':r,'rocwmma_event_ms_p95':r,'scalar_event_ms_p50':scalar,'scalar_event_ms_p95':scalar,'hipblaslt_event_ms_p50':blas,'hipblaslt_event_ms_p95':blas}))\n",
            encoding="utf-8")
        os.chmod(fake, 0o755)
        output = root / "output"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--binary", str(fake),
            "--output-directory", str(output), "--sequences", "16,512,2048",
            "--inners", "64,128", "--runs", "2", "--warmup", "0",
            "--repetitions", "2",
        ], text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text())
        verification = json.loads((output / "verification.json").read_text())
        raw = [json.loads(line) for line in (output / "raw.jsonl").read_text().splitlines()]
        assert summary["status"] == "pass"
        assert summary["processes"] == 12
        assert len(summary["comparisons"]) == 6
        assert summary["correctness_gate"] is True
        assert summary["long_scalar_gate"] is True
        assert summary["t512_blas_gate"] is True
        assert summary["long_blas_counterexample"] is True
        assert summary["online_prototype_admitted"] is True
        assert summary["model_route_accepted"] is False
        assert verification["all_complete_outputs"] is True
        assert verification["raw_records"] == 12
        assert len(raw) == 12 and all(row["record_type"] == "rocwmma_qk_measurement"
                                      for row in raw)
    print("rocWMMA QK matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
