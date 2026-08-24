#!/usr/bin/env python3
"""Contract test for exact weight-gradient solution model gating."""

from __future__ import annotations

import json
import pathlib
import stat
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/compare_fp32_weight_gradient_solutions.py"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="microllm-wgrad-model-") as temporary:
        root = pathlib.Path(temporary)
        binary = root / "fake_train.py"
        binary.write_text(
            """#!/usr/bin/env python3
import json, sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
qwen = 'qwen' in args['--config']
candidate = '--fp32-gate-up-weight-gradient-solution-index' in args
layers = 24 if qwen else 28
steps = int(args['--steps']); warmup = int(args['--warmup'])
dispatches = (steps + warmup) * layers * 2 if candidate else 0
print(json.dumps({
  'status': 'pass', 'parameter_count': 494032768 if qwen else 1777088000,
  'warmup': warmup, 'steps': steps, 'batch': int(args['--batch']),
  'context': len(args['--tokens'].split(',')) - 1, 'parameter_changed': True,
  'fp32_solution_registered_entries': 1 if candidate else 0,
  'fp32_solution_registry_hits': dispatches,
  'fp32_solution_dispatches': dispatches,
  'tokens_per_second': 102.0 if candidate else 100.0,
  'engine_peak_bytes': 1000, 'first_loss': 2.0, 'final_loss': 1.0,
}))
""", encoding="utf-8")
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        paths = {}
        for name in ("qwen-config", "qwen-weights", "deepseek-config",
                     "deepseek-weights"):
            path = root / name
            path.write_text("fixture", encoding="utf-8")
            paths[name] = path
        output = root / "results"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--binary", str(binary),
            "--qwen-config", str(paths["qwen-config"]),
            "--qwen-weights", str(paths["qwen-weights"]),
            "--deepseek-config", str(paths["deepseek-config"]),
            "--deepseek-weights", str(paths["deepseek-weights"]),
            "--output-directory", str(output), "--runs", "3",
            "--warmup", "1", "--steps", "2", "--context", "8",
        ], capture_output=True, text=True)
        if completed.returncode != 0:
            raise AssertionError(completed.stderr or completed.stdout)
        summary = json.loads((output / "summary.json").read_text(
            encoding="utf-8"))
        assert summary["status"] == "pass"
        assert summary["processes"] == 12
        assert all(row["throughput_speedup"] == 1.02
                   for row in summary["comparisons"])
        assert all(all(row["gates"].values())
                   for row in summary["comparisons"])
    print("FP32 weight-gradient solution model contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
