#!/usr/bin/env python3
import json
import statistics
import time

import torch


def measure(function):
    for _ in range(2):
        function()
        torch.cuda.synchronize()
    events = []
    walls = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        finish = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        output = function()
        finish.record()
        finish.synchronize()
        events.append(start.elapsed_time(finish))
        walls.append((time.perf_counter() - wall_start) * 1000.0)
        if output.numel() == 0:
            raise RuntimeError("empty benchmark output")
    return statistics.median(events), statistics.median(walls)


def run(model, inner, columns):
    indices = torch.arange(inner, device="cuda")
    activation = ((indices.remainder(29).float() - 14) / 64).reshape(1, inner)
    weight_indices = torch.arange(inner * columns, device="cuda")
    weight_int8 = (
        weight_indices.remainder(31).to(torch.int16) - 15
    ).to(torch.int8).reshape(inner, columns)
    scale = 0.03125
    resident_fp32 = weight_int8.float() * scale
    dequant = measure(lambda: activation @ (weight_int8.float() * scale))
    resident = measure(lambda: activation @ resident_fp32)
    print(json.dumps({
        "model": model,
        "m": 1,
        "k": inner,
        "n": columns,
        "warmup": 2,
        "repetitions": 5,
        "torch_dequant_event_ms": dequant[0],
        "torch_dequant_wall_ms": dequant[1],
        "torch_resident_fp32_event_ms": resident[0],
        "torch_resident_fp32_wall_ms": resident[1],
    }, sort_keys=True))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch ROCm device is unavailable")
    run("qwen2.5-0.5b", 896, 4864)
    run("deepseek-distill-1.5b", 1536, 8960)
