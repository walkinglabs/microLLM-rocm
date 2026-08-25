#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/bf16_rms_norm_output_matrix.py"


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fake = root / "fake.py"
        fake.write_text(
            """#!/usr/bin/env python3
import json,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2])); model=a['--model']
print(json.dumps({'status':'pass','record_type':'bf16_rms_norm_output_probe',
'model':model,'rows':int(a['--rows']),'width':896 if model=='qwen' else 1536,
'complete_output_equal':True,'event_speedup':2.0,'wall_speedup':1.5,
'host_to_device_calls':0,'device_to_host_calls':0}))
""", encoding="utf-8")
        fake.chmod(0o755)
        output = root / "out"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--binary", str(fake),
            "--output-directory", str(output), "--runs", "2",
            "--rows", "1024", "--warmup", "0", "--repetitions", "1"],
            text=True, capture_output=True, check=False)
        assert completed.returncode == 0, completed.stderr
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert summary["raw_processes"] == 4
        assert summary["operator_gate_passed"] is True
        assert all(row["event_speedup_median"] == 2.0
                   for row in summary["comparisons"])
        assert all(row["complete_output_equal"] is True
                   for row in summary["comparisons"])
    print("BF16 RMSNorm output matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
