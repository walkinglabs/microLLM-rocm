#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/bf16_grouped_qkv_matrix.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2]));mode=a['--output-dtype'];model=a['--model'];supported=mode=='model'
r={'schema_version':1,'status':'pass','record_type':'bf16_grouped_qkv_probe','model':model,'output_dtype':mode,'grouped_supported':supported}
if supported:r.update({'passing_candidates':2,'solution_index':7,'maximum_absolute_error':0.0001,'maximum_rms_error':0.00001,'event_speedup':1.5,'wall_speedup':1.3,'reinitialized_event_speedup':0.9,'reinitialized_wall_speedup':0.8,'workspace_bytes':720})
print(json.dumps(r))
""", encoding="utf-8")
        os.chmod(fake, 0o755)
        output = root / "output"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--binary", str(fake),
            "--output-directory", str(output), "--runs", "2",
            "--warmup", "0", "--repetitions", "1",
            "--maximum-algorithms", "2"],
            text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "pass"
        assert summary["raw_processes"] == 8
        assert summary["direct_fp32_unsupported_rows"] == 4
        assert summary["operator_keep"] is True
        assert len(summary["comparisons"]) == 2
        assert all(row["event_speedup_median"] == 1.5
                   for row in summary["comparisons"])
    print("BF16 grouped QKV matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
