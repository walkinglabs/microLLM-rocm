#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/bf16_grouped_shape_matrix.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2]));p=a['--projection'];groups=3 if p=='qkv' else 2
print(json.dumps({'schema_version':1,'status':'pass','record_type':'bf16_grouped_qkv_probe' if p=='qkv' else 'bf16_grouped_gate_up_probe','projection':p,'model':a['--model'],'rows':int(a['--rows']),'groups':groups,'grouped_supported':True,'algorithm_count':100,'passing_candidates':2,'solution_index':7,'maximum_absolute_error':0.0001,'maximum_rms_error':0.00001,'event_speedup':1.3,'user_arguments_event_speedup':1.2,'user_arguments_wall_speedup':1.1,'reinitialized_event_speedup':1.1 if a['--model']=='deepseek' and a['--rows']=='256' and p=='qkv' else .9,'user_arguments_setup_ms':.1}))
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
        assert summary["raw_processes"] == 16
        assert summary["capability_gate"] is True
        assert summary["reinitialization_faster_cases"] == 1
        assert len(summary["comparisons"]) == 8
        assert all(row["user_arguments_event_speedup_median"] == 1.2
                   for row in summary["comparisons"])
    print("BF16 grouped shape matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
