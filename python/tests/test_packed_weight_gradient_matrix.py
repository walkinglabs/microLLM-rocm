#!/usr/bin/env python3
"""Contract test for packed weight-gradient repeated measurements."""

from __future__ import annotations

import json
import pathlib
import stat
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/packed_weight_gradient_matrix.py"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="microllm-packed-wgrad-") as temporary:
        root = pathlib.Path(temporary)
        binary = root / "fake_probe.py"
        binary.write_text(
            """#!/usr/bin/env python3
import json, sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
groups = 3 if args['--projection'] == 'qkv' else 2
print(json.dumps({
  'schema_version': 1, 'status': 'pass',
  'record_type': 'packed_weight_gradient_probe',
  'model': args['--model'], 'projection': args['--projection'],
  'rows': int(args['--rows']), 'groups': groups,
  'pack_copies_per_step': groups, 'packed_gradient_bytes': 1024,
  'packed_output_bytes': 2048, 'event_speedup': 0.9,
  'maximum_absolute_error': 1e-7,
}))
""", encoding="utf-8")
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        output = root / "results"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--binary", str(binary),
            "--output-directory", str(output), "--runs", "3",
            "--rows", "8", "--warmup", "0", "--repetitions", "1",
        ], capture_output=True, text=True)
        if completed.returncode != 0:
            raise AssertionError(completed.stderr or completed.stdout)
        raw = [json.loads(line) for line in (
            output / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
        summary = json.loads((output / "summary.json").read_text(
            encoding="utf-8"))
        assert len(raw) == 12
        assert len({(row["model"], row["projection"], row["process_run"])
                    for row in raw}) == 12
        assert summary["performance_cases_passed"] == 0
        assert summary["performance_cases_total"] == 4
        assert summary["decision"] == "discard packed weight-gradient route"
    print("packed weight-gradient matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
