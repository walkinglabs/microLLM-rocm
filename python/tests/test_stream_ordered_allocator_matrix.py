#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/stream_ordered_allocator_matrix.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2])); n=int(a['--nodes']); mode=a['--mode']
wall={'deferred':1.0,'async':0.8,'graph':0.7}[mode]
print(json.dumps({'status':'pass','mode':mode,'nodes':n,'elements':int(a['--elements']),
'wall_p50_ms':wall,'maximum_absolute_error':0,
'maximum_unique_addresses':2 if mode=='async' else n if mode=='graph' else 0,
'maximum_deferred_bytes':n*4 if mode=='deferred' else 0,
'pool_reserved_high_bytes':128 if mode=='async' else 0,
'graph_node_count':n*3+1 if mode=='graph' else 0}))
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
        assert len(summary["comparisons"]) == 2
        assert summary["address_contract"] is True
        assert summary["async_performance_gate"] is True
        assert summary["graph_performance_gate"] is True
        assert summary["decision"] == "enable async allocator"
    print("stream ordered allocator matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
