#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/compare_causal_softmax_threads.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake_op.py"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import json,sys\n"
            "a=dict(zip(sys.argv[1::2],sys.argv[2::2]));t=a['--threads-128']=='true';seq=int(a['--sequence']);heads=int(a['--heads']);candidate=0.9 if seq==512 else 1.02\n"
            "print(json.dumps({'status':'pass','threads':128 if t else 256,'sequence':seq,'heads':heads,'maximum_absolute_error':1e-9 if t else 0,'rms_error':1e-10 if t else 0,'event_ms_p50':candidate if t else 1.0,'wall_ms_p50':candidate if t else 1.0}))\n",
            encoding="utf-8")
        os.chmod(fake, 0o755)
        output = root / "output"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--binary", str(fake),
            "--output-directory", str(output), "--runs", "2",
        ], text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text())
        assert summary["status"] == "pass"
        assert summary["processes"] == 24
        assert summary["correctness_gate"] is True
        assert summary["universal_performance_gate"] is False
        assert summary["t512_performance_gate"] is True
        assert len(summary["comparisons"]) == 6
    print("causal-softmax thread comparison contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
