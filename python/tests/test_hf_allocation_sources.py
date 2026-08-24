#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/hf_allocation_sources.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config = root / "config.json"
        weights = root / "weights.bin"
        config.write_text("{}", encoding="utf-8")
        weights.write_bytes(b"weights")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "models": [
            {"name": "qwen", "revision": "a", "config": str(config),
             "weights": str(weights), "inference": {"token_ids": [1, 2]}},
            {"name": "deepseek", "revision": "b", "config": str(config),
             "weights": str(weights), "inference": {"token_ids": [3, 4]}},
        ]}), encoding="utf-8")
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json
records=[{'source':'attention.core','device':'hip:0','allocation_bytes':64,
'calls':4,'total_bytes':256},{'source':'ffn','device':'hip:0',
'allocation_bytes':32,'calls':2,'total_bytes':64}]
print(json.dumps({'status':'pass','allocation_source_diagnostics':True,
'allocation_source_calls':6,'allocation_source_bytes':320,
'allocation_source_records':records}))
""", encoding="utf-8")
        os.chmod(fake, 0o755)
        output = root / "output"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--manifest", str(manifest),
            "--binary", str(fake), "--output-directory", str(output),
            "--runs", "2", "--context", "8"],
            text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "pass"
        assert summary["raw_processes"] == 4
        assert summary["deterministic_distributions"] is True
        assert summary["common_top_source"] == "attention.core"
        assert summary["decision"] == "profile and optimize attention.core"
        assert all(row["allocation_calls"] == 6 for row in summary["models"])
    print("HF allocation source runner contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
