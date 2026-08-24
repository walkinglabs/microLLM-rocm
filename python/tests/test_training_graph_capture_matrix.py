#!/usr/bin/env python3
"""Contract test for the staged training HIP Graph matrix."""

from __future__ import annotations

import json
import pathlib
import stat
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/training_graph_capture_matrix.py"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="microllm-training-graph-") as temporary:
        root = pathlib.Path(temporary)
        binary = root / "fake_probe.py"
        binary.write_text(
            """#!/usr/bin/env python3
import json, sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
stage = args['--stage']
supported = stage == 'optimizer'
print(json.dumps({
  'schema_version': 1, 'status': 'pass', 'stage': stage,
  'precision': args['--precision'], 'capture_supported': supported,
  'captured_nodes': 21 if supported else 0,
  'capture_error': '' if supported else
      'HIP graph capture forbids dynamic Tensor allocation',
  'capture_status_after_failure': 0, 'recovery_status': 0,
  'capture_status_after_recovery': 0, 'capture_recovery_failed': False,
  'optimizer_step_after_capture': 1 if supported else 0,
  'optimizer_step_after_replay': 1 if supported else 0,
  'optimizer_replay_advances_host_step': False,
  'deferred_blocks': 10, 'deferred_bytes': 4096,
}))
""", encoding="utf-8")
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        output = root / "results"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--binary", str(binary),
            "--output-directory", str(output), "--runs", "3",
            "--maximum-blocks", "64",
        ], capture_output=True, text=True)
        if completed.returncode != 0:
            raise AssertionError(completed.stderr or completed.stdout)
        raw = [json.loads(line) for line in (
            output / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
        summary = json.loads((output / "summary.json").read_text(
            encoding="utf-8"))
        assert len(raw) == 24
        assert len({(row["precision"], row["stage"], row["process_run"])
                    for row in raw}) == 24
        assert len(summary["cases"]) == 8
        assert summary["status"] == "pass"
        assert all(summary["gates"].values())
        assert "stable workspaces" in summary["decision"]
    print("training graph capture matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
