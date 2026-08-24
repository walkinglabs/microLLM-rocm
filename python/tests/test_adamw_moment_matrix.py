#!/usr/bin/env python3
"""Contract test for the reproducible BF16 AdamW moment matrix runner."""

from __future__ import annotations

import json
import pathlib
import stat
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/adamw_moment_matrix.py"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="microllm-adamw-moment-") as temporary:
        root = pathlib.Path(temporary)
        fake = root / "fake_train.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json, sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
policy = args['--adamw-moment-precision']
deep = 'deepseek' in args['--config']
parameters = 1777088000 if deep else 494032768
record = {
  'schema_version': 1, 'status': 'pass', 'parameter_count': parameters,
  'adamw_moment_precision': policy,
  'adamw_moment_state_bytes': parameters * (4 if policy == 'bf16' else 8),
  'optimizer_host_to_device_calls': 0, 'optimizer_device_to_host_calls': 0,
  'optimizer_host_to_device_bytes': 0, 'optimizer_device_to_host_bytes': 0,
  'adamw_multi_tensor_update': False,
  'warmup': int(args['--warmup']), 'steps': int(args['--steps']),
  'batch': int(args['--batch']),
  'context': len(args['--tokens'].split(',')) - 1,
  'tokens_per_second': 110.0 if policy == 'bf16' else 100.0,
  'mean_optimizer_ms': 8.0 if policy == 'bf16' else 10.0,
  'engine_peak_bytes': parameters * (18 if policy == 'bf16' else 22),
  'first_loss': 2.0, 'final_loss': 1.0, 'parameter_changed': True,
}
print(json.dumps(record))
""",
            encoding="utf-8")
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        inputs = {}
        for name in ("qwen-config", "qwen-weights", "deepseek-config",
                     "deepseek-weights"):
            path = root / name
            path.write_text("fixture", encoding="utf-8")
            inputs[name] = path
        output = root / "results"
        command = [
            sys.executable, str(RUNNER), "--binary", str(fake),
            "--qwen-config", str(inputs["qwen-config"]),
            "--qwen-weights", str(inputs["qwen-weights"]),
            "--deepseek-config", str(inputs["deepseek-config"]),
            "--deepseek-weights", str(inputs["deepseek-weights"]),
            "--output-directory", str(output), "--runs", "3",
            "--warmup", "1", "--steps", "2", "--context", "8",
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise AssertionError(completed.stderr or completed.stdout)
        rows = [json.loads(line) for line in
                (output / "training.jsonl").read_text(encoding="utf-8").splitlines()]
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert len(rows) == 12
        assert summary["status"] == "pass"
        assert all(all(model["gates"].values()) for model in summary["models"])
        assert {row["policy"] for row in rows} == {"fp32", "bf16"}
    print("adamw moment matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
