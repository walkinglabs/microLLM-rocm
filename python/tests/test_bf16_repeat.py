#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/compare_bf16_repeat.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import json,sys\n"
            "a=dict(zip(sys.argv[1::2],sys.argv[2::2]));f=a['--fused']=='true'\n"
            "print(json.dumps({'status':'pass','fused':f,'event_ms_p50':0.5 if f else 1.0}))\n",
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
        assert summary["processes"] == 32
        assert summary["correctness_gate"] is True
        assert summary["performance_gate"] is True
        assert len(summary["comparisons"]) == 8
    print("BF16 repeat comparison contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
