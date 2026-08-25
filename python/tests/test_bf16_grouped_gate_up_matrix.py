#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/bf16_grouped_gate_up_matrix.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2]));model=a['--model']
print(json.dumps({'schema_version':1,'status':'pass',
'record_type':'bf16_grouped_gate_up_probe','projection':'gate-up',
'model':model,'groups':2,'gate_swish':a['--gate-swish']=='true',
'grouped_supported':True,'algorithm_count':100,
'passing_candidates':2,'solution_index':7,'maximum_absolute_error':0.0,
'maximum_rms_error':0.0,'event_speedup':1.3,'wall_speedup':1.2,
'reinitialized_event_speedup':0.9,'reinitialized_wall_speedup':0.8,
'user_arguments_setup_ms':0.1,'user_arguments_event_speedup':1.2,
'user_arguments_wall_speedup':1.1,'workspace_bytes':64}))
""", encoding="utf-8")
        os.chmod(fake, 0o755)
        output = root / "output"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--binary", str(fake),
            "--output-directory", str(output), "--runs", "2",
            "--warmup", "0", "--repetitions", "1",
            "--maximum-algorithms", "2",
        ], text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text(
            encoding="utf-8"))
        assert summary["status"] == "pass"
        assert summary["raw_processes"] == 4
        assert summary["capability_gate"] is True
        assert summary["reinitialization_counterexample_gate"] is True
        assert len(summary["comparisons"]) == 2
        assert all(row["user_arguments_event_speedup_median"] == 1.2
                   for row in summary["comparisons"])
        swish_output = root / "swish"
        swish = subprocess.run([
            sys.executable, str(RUNNER), "--binary", str(fake),
            "--output-directory", str(swish_output), "--runs", "1",
            "--warmup", "0", "--repetitions", "1",
            "--maximum-algorithms", "2", "--rows", "1024",
            "--gate-swish",
        ], text=True, capture_output=True, check=False)
        if swish.returncode != 0:
            raise AssertionError(swish.stdout + swish.stderr)
        swish_summary = json.loads((swish_output / "summary.json").read_text(
            encoding="utf-8"))
        assert swish_summary["gate_swish"] is True
        assert swish_summary["record_type"] == \
            "bf16_grouped_gate_up_swish_matrix_summary"
        assert all(row["rows"] == 1024 for row in swish_summary["comparisons"])
    print("BF16 grouped gate/up matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
