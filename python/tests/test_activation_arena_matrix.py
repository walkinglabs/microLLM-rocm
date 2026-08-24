#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/activation_arena_matrix.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2]))
n=int(a['--nodes']);e=int(a['--elements']);m=a['--mode']
cap=((e*4+255)//256)*512
wall={'deferred':1.0,'arena':0.5,'arena_graph':0.4}[m]
print(json.dumps({'status':'pass','mode':m,'nodes':n,'elements':e,
'wall_p50_ms':wall,'maximum_absolute_error':0,
'arena_capacity_bytes':cap if m!='deferred' else 0,
'maximum_unique_addresses':2 if m!='deferred' else 0,
'graph_node_count':n+1 if m=='arena_graph' else 0,'graph_setup_ms':1.0}))
""", encoding="utf-8")
        os.chmod(fake, 0o755)
        output = root / "output"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--binary", str(fake),
            "--output-directory", str(output), "--nodes", "8,32",
            "--elements", "1", "--repetitions", "1", "--warmup", "0",
            "--timed-repetitions", "1"],
            text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "pass"
        assert summary["raw_processes"] == 6
        assert summary["layout_contract"] is True
        assert summary["arena_performance_gate"] is True
        assert summary["arena_graph_performance_gate"] is True
        assert summary["decision"] == "keep arena and arena Graph candidate"
        assert all(row["arena_graph_break_even_replays"] == 2
                   for row in summary["comparisons"])
    print("activation arena matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
