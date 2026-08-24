#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/per_device_handle_regression.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        baseline = root / "baseline.jsonl"
        baseline_rows = []
        for key, model in (("qwen", "qwen2.5-0.5b"),
                           ("deepseek", "deepseek-r1-distill-qwen-1.5b")):
            for mode in ("inference", "training"):
                baseline_rows.append({
                    "model_key": key, "model": model, "mode": mode,
                    "policy": "legacy", "context": 4,
                    "tokens_per_second": 100.0, "top_index": 3,
                    "top_value": 4.0, "logits_sum": 10.0,
                    "logits_square_sum": 30.0, "loss": 2.0,
                    "observed_parameter_after": 0.5,
                })
        baseline.write_text("".join(json.dumps(row) + "\n" for row in baseline_rows),
                            encoding="utf-8")
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json, sys
args=dict(zip(sys.argv[1::2],sys.argv[2::2]))
print(json.dumps({'status':'pass','model':args['--model'],'mode':args['--mode'],
'policy':'legacy','context':int(args['--context']),'tokens_per_second':101.0,
'top_index':3,'top_value':4.0,'logits_sum':10.0,'logits_square_sum':30.0,
'loss':2.0,'observed_parameter_after':0.5}))
""", encoding="utf-8")
        os.chmod(fake, 0o755)
        fixtures = []
        for name in ("qc", "qw", "dc", "dw"):
            path = root / name
            path.write_text("fixture", encoding="utf-8")
            fixtures.append(path)
        output = root / "output"
        command = [
            sys.executable, str(RUNNER), "--benchmark", str(fake),
            "--baseline", str(baseline), "--output-directory", str(output),
            "--qwen-config", str(fixtures[0]), "--qwen-weights", str(fixtures[1]),
            "--deepseek-config", str(fixtures[2]),
            "--deepseek-weights", str(fixtures[3]), "--context", "4",
            "--repetitions", "1", "--warmup", "0", "--steps", "1",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "pass"
        assert summary["raw_processes"] == 4
        assert len(summary["comparisons"]) == 4
        assert summary["minimum_throughput_ratio"] == 1.01
        assert summary["decision"] == "keep per-device handles"
    print("per-device handle regression contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
