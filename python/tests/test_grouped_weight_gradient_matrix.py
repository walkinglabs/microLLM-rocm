#!/usr/bin/env python3
"""Schema and coverage contract for grouped weight-gradient probing."""

from __future__ import annotations

import json
import pathlib
import stat
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/grouped_weight_gradient_matrix.py"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="microllm-grouped-wgrad-") as temporary:
        root = pathlib.Path(temporary)
        binary = root / "fake_probe.py"
        binary.write_text(
            """#!/usr/bin/env python3
import json, sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
print(json.dumps({
  'schema_version': 1, 'status': 'pass',
  'record_type': 'grouped_weight_gradient_probe',
  'model': args['--model'], 'projection': args['--projection'],
  'input_layout': args['--input-layout'], 'rows': int(args['--rows']),
  'algorithm_count': 17, 'supported_candidates': 0,
  'passing_candidates': 0, 'grouped_supported': False,
  'baseline_event_ms_p50': 0.25,
}))
""",
            encoding="utf-8")
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        output = root / "results"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--binary", str(binary),
            "--output-directory", str(output), "--rows", "8",
            "--warmup", "0", "--repetitions", "1",
        ], capture_output=True, text=True)
        if completed.returncode != 0:
            raise AssertionError(completed.stderr or completed.stdout)
        rows = [json.loads(line) for line in (
            output / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
        summary = json.loads((output / "summary.json").read_text(
            encoding="utf-8"))
        assert len(rows) == 8
        assert len({(row["input_layout"], row["model"], row["projection"])
                    for row in rows}) == 8
        assert summary == {
            "schema_version": 1,
            "status": "pass",
            "experiment": "grouped_weight_gradient_capability",
            "cases": 8,
            "supported_cases": 0,
            "unsupported_cases": 8,
            "decision": "discard grouped weight-gradient route",
        }
    print("grouped weight-gradient matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
