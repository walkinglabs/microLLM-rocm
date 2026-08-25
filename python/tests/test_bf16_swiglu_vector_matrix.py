#!/usr/bin/env python3
"""Contract test for the repeated BF16 SwiGLU runner."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/bf16_swiglu_vector_matrix.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        output = root / "out"
        fake.write_text(
            """#!/usr/bin/env python3
import json,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2])); impl=a['--implementation']
print(json.dumps({'op':'swiglu','implementation':impl,'size':int(a['--size']),
'kernel_ms_mean':1.0 if impl=='scalar' else 0.8,'maximum_absolute_error':0.0}))
""", encoding="utf-8")
        fake.chmod(0o755)
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--binary", str(fake),
            "--output-directory", str(output), "--runs", "2",
            "--warmup", "0", "--repetitions", "1"],
            text=True, capture_output=True, check=False)
        assert completed.returncode == 0, completed.stderr
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["raw_processes"] == 8
        assert summary["operator_gate_passed"] is True
        assert len(summary["comparisons"]) == 2
        assert all(row["speedup"] == 1.25 for row in summary["comparisons"])
        assert all(row["maximum_absolute_error"] == 0.0
                   for row in summary["comparisons"])
    print("BF16 SwiGLU vector matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
