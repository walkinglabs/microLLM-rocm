import json
import math
import time
from pathlib import Path


TRACE_SCHEMA_VERSION = 1


def load_jsonl(path):
    records = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != TRACE_SCHEMA_VERSION:
                raise ValueError(f"{path}:{line_number}: unsupported trace schema")
            records.append(record)
    return records


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")


def tensor_dtype_name(tensor):
    text = str(tensor.dtype)
    return text.removeprefix("torch.")


def tensor_values(tensor):
    return [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def value_statistics(values, declared_numel):
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {
            "numel": declared_numel,
            "finite_count": 0,
            "minimum": 0.0,
            "maximum": 0.0,
            "mean": 0.0,
            "l2_norm": 0.0,
        }
    return {
        "numel": declared_numel,
        "finite_count": len(finite),
        "minimum": min(finite),
        "maximum": max(finite),
        "mean": sum(finite) / len(finite),
        "l2_norm": math.sqrt(sum(value * value for value in finite)),
    }


class TorchTraceSession:
    def __init__(self, framework, run_id, phase, device, *, record_operators=True,
                 record_layers=True, record_model=True, capture_values=True,
                 max_captured_elements=4096):
        self.framework = framework
        self.run_id = run_id
        self.phase = phase
        self.device = device
        self.record_operators = record_operators
        self.record_layers = record_layers
        self.record_model = record_model
        self.capture_values = capture_values
        self.max_captured_elements = max_captured_elements
        self.iteration = 0
        self.records = []

    def enabled(self, kind):
        if kind == "operator":
            return self.record_operators
        if kind == "layer":
            return self.record_layers
        if kind == "model":
            return self.record_model
        return self.capture_values

    def synchronize(self):
        if self.device.type == "cuda":
            import torch
            torch.cuda.synchronize(self.device)

    def record(self, kind, name, tensor, wall_ms=0.0):
        if not self.enabled(kind):
            return tensor
        values = tensor_values(tensor) if self.capture_values else []
        captured = []
        for value in values[:self.max_captured_elements]:
            if math.isnan(value):
                captured.append("nan")
            elif value == math.inf:
                captured.append("inf")
            elif value == -math.inf:
                captured.append("-inf")
            else:
                captured.append(value)
        self.records.append({
            "schema_version": TRACE_SCHEMA_VERSION,
            "framework": self.framework,
            "run_id": self.run_id,
            "phase": self.phase,
            "sequence": len(self.records),
            "iteration": self.iteration,
            "kind": kind,
            "name": name,
            "shape": list(tensor.shape),
            "dtype": tensor_dtype_name(tensor),
            "device": str(tensor.device),
            "wall_ms": wall_ms,
            "statistics": value_statistics(values, tensor.numel()),
            "values_truncated": len(captured) != len(values),
            "values": captured,
        })
        return tensor

    def measure(self, kind, name, function):
        if not self.enabled(kind):
            return function()
        self.synchronize()
        start = time.perf_counter_ns()
        output = function()
        self.synchronize()
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        return self.record(kind, name, output, elapsed_ms)
