#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/scoped_deferred_model_matrix.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake_benchmark.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json, pathlib, struct, sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
if '--logits-output' in args:
    path = pathlib.Path(args['--logits-output'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack('<4f', 1.0, 2.0, 3.0, 4.0))
deferred = args['--policy'] == 'deferred'
print(json.dumps({
    'schema_version': 1, 'status': 'pass',
    'record_type': 'scoped_deferred_model_measurement',
    'model': args['--model'], 'mode': args['--mode'],
    'policy': args['--policy'], 'precision': args['--precision'],
    'context': int(args['--context']), 'tokens_per_second': 105.0 if deferred else 100.0,
    'loss': 2.0, 'observed_parameter_after': 0.5, 'parameter_changed': True,
    'engine_peak_bytes': 1000, 'maximum_deferred_bytes': 200 if deferred else 0,
    'engine_backend_allocation_calls': 20 if deferred else 2,
    'deferred_overflow_flushes': 0
}))
""",
            encoding="utf-8")
        os.chmod(fake, 0o755)
        for name in ("qwen-config", "qwen-weights", "deep-config", "deep-weights"):
            (root / name).write_text("fixture", encoding="utf-8")
        output = root / "output"
        command = [
            sys.executable, str(RUNNER),
            "--benchmark", str(fake),
            "--output-directory", str(output),
            "--qwen-config", str(root / "qwen-config"),
            "--qwen-weights", str(root / "qwen-weights"),
            "--deepseek-config", str(root / "deep-config"),
            "--deepseek-weights", str(root / "deep-weights"),
            "--contexts", "4", "--repetitions", "1",
            "--warmup", "0", "--steps", "1",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "pass"
        assert summary["raw_processes"] == 8
        assert len(summary["paired_checks"]) == 4
        assert len(summary["comparisons"]) == 4
        assert summary["correctness_gate"] is True
        assert summary["performance_gate"] is True
        assert summary["decision"] == "enable candidate"
        assert len((output / "raw.jsonl").read_text().splitlines()) == 8
        assert len((output / "pairs.jsonl").read_text().splitlines()) == 4
    print("scoped deferred model matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
