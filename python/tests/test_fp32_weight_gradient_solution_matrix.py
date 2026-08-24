#!/usr/bin/env python3
"""Contract test for cross-process FP32 weight-gradient solution selection."""

from __future__ import annotations

import json
import pathlib
import stat
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/fp32_weight_gradient_solution_matrix.py"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="microllm-wgrad-solution-") as temporary:
        root = pathlib.Path(temporary)
        binary = root / "fake_tuner.py"
        binary.write_text(
            """#!/usr/bin/env python3
import json, sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
count = int(args['--maximum-algorithms'])
candidates = []
for index in range(count):
  candidates.append({
    'index': 100 + index, 'correctness_passed': True, 'finite': True,
    'event_speedup_vs_default': 1.08 if index == 2 else 1.0,
  })
print(json.dumps({
  'schema_version': 1, 'status': 'pass',
  'record_type': 'fp32_weight_gradient_algorithm_tune',
  'model': args['--model'], 'operation': args['--operation'],
  'rows': int(args['--rows']), 'candidates': candidates,
}))
""", encoding="utf-8")
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        output = root / "results"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--binary", str(binary),
            "--output-directory", str(output), "--runs", "3",
            "--rows", "8", "--maximum-algorithms", "4",
            "--warmup", "0", "--repetitions", "1",
        ], capture_output=True, text=True)
        if completed.returncode != 0:
            raise AssertionError(completed.stderr or completed.stdout)
        summary = json.loads((output / "summary.json").read_text(
            encoding="utf-8"))
        assert summary["processes"] == 6
        assert summary["candidate_evaluations"] == 24
        assert summary["model_gate_ready"] is True
        assert all(row["selected_index"] == 102 for row in summary["summaries"])
        assert all(row["performance_gate"] is True for row in summary["summaries"])
    print("FP32 weight-gradient solution matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
