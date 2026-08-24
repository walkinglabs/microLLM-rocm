#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/arena_ffn_matrix.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2]));m=a['--mode'];model=a['--model'];rows=int(a['--rows'])
wall={'deferred':1.0,'arena':0.6,'arena_graph':0.5}[m]
print(json.dumps({'status':'pass','mode':m,'model':model,'rows':rows,
'hidden':8,'intermediate':16,'wall_p50_ms':wall,'graph_setup_ms':1.0,
'maximum_absolute_error':0,'rms_error':0,'arena_capacity_bytes':192,
'graph_node_count':4 if m=='arena_graph' else 0}))
""", encoding="utf-8")
        os.chmod(fake, 0o755)
        output = root / "output"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--binary", str(fake),
            "--output-directory", str(output), "--repetitions", "1",
            "--warmup", "0", "--timed-repetitions", "1"],
            text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "pass"
        assert summary["raw_processes"] == 12
        assert len(summary["comparisons"]) == 4
        assert summary["arena_keep_rows"] == 4
        assert summary["arena_graph_keep_rows"] == 4
        assert summary["decision"] == "keep shape-selective FFN arena and Graph candidate"
    print("arena FFN matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
