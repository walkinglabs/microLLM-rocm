#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/fp32_attention_solution_matrix.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2]));model=a['--model'];op=a['--operation'];seq=int(a['--sequence'])
print(json.dumps({'status':'pass','record_type':'fp32_attention_algorithm_tune',
'model':model,'operation':op,'sequence':seq,'candidate_count':2,
'default_event_ms_p50':1.0,'default_wall_ms_p50':1.1,'candidates':[
{'index':7,'workspace_bytes':0,'correctness_passed':True,
'maximum_absolute_error':0,'rms_error':0,'event_ms_p50':0.5,'wall_ms_p50':0.6},
{'index':9,'workspace_bytes':1024,'correctness_passed':True,
'maximum_absolute_error':1e-6,'rms_error':1e-7,'event_ms_p50':0.8,'wall_ms_p50':0.9}]}))
""", encoding="utf-8")
        os.chmod(fake, 0o755)
        output = root / "output"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--binary", str(fake),
            "--output-directory", str(output), "--runs", "2",
            "--sequence", "1024",
            "--maximum-algorithms", "2", "--warmup", "0",
            "--repetitions", "1"], text=True, capture_output=True,
            check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "pass"
        assert summary["raw_processes"] == 8
        assert len(summary["comparisons"]) == 4
        assert summary["keep_rows"] == 4
        assert summary["decision"] == "register exact FP32 Attention candidates"
        assert all(row["recommended_index"] == 7 for row in summary["comparisons"])
        assert all(row["sequence"] == 1024 for row in summary["comparisons"])
        assert all(row["recommended_event_speedup"] == 2.0
                   for row in summary["comparisons"])
    print("FP32 Attention solution matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
