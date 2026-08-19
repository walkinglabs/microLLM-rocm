import json
import subprocess
import sys


def run(command):
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    record = json.loads(completed.stdout)
    if record.get("schema_version") != 1:
        raise RuntimeError(f"unexpected benchmark schema: {record!r}")
    return record


micro = run(
    [sys.argv[1], "--op", "add", "--size", "16", "--warmup", "1", "--repetitions", "2", "--device", "cpu"]
)
model = run(
    [sys.argv[2], "--mode", "train", "--model", "tiny", "--device", "cpu", "--steps", "1", "--warmup", "0", "--batch", "1", "--context", "2", "--new-tokens", "2"]
)
required_micro = {"kernel_ms_mean", "wall_ms_mean", "maximum_absolute_error"}
required_model = {"measured_wall_seconds", "tokens_per_second", "device_peak_engine_bytes"}
if not required_micro.issubset(micro):
    raise RuntimeError("micro benchmark record is missing required fields")
if not required_model.issubset(model):
    raise RuntimeError("model benchmark record is missing required fields")
