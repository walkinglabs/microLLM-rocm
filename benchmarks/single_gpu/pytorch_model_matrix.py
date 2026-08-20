#!/usr/bin/env python3
"""PyTorch eager baseline for the built-in microLLM single-device model matrix."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    vocabulary_size: int
    dimension: int
    layers: int
    heads: int
    kv_heads: int
    ffn_dimension: int
    max_sequence_length: int = 512
    rope_base: float = 10000.0

    @property
    def head_dimension(self) -> int:
        return self.dimension // self.heads


CONFIGS = {
    "tiny": Config(32, 16, 2, 4, 2, 32, 128),
    "model-s": Config(8192, 384, 6, 6, 6, 832),
    "model-m": Config(8192, 512, 8, 8, 8, 1184),
}

PROFILES = {
    "tiny": {
        "parameter_count": 5_712,
        "train": {"batch": 1, "context": 8, "steps": 3, "warmup": 1, "new_tokens": 8},
        "generate": {"batch": 1, "context": 8, "steps": 3, "warmup": 1, "new_tokens": 8},
    },
    "model-s": {
        "parameter_count": 15_586_176,
        "train": {"batch": 1, "context": 2, "steps": 1, "warmup": 0, "new_tokens": 2},
        "generate": {"batch": 1, "context": 4, "steps": 1, "warmup": 0, "new_tokens": 2},
    },
    "model-m": {
        "parameter_count": 31_334_912,
        "train": {"batch": 1, "context": 1, "steps": 1, "warmup": 0, "new_tokens": 2},
        "generate": {"batch": 1, "context": 4, "steps": 1, "warmup": 0, "new_tokens": 2},
    },
}

COMPARISON_SETTINGS = {
    "tiny": {
        "train": {"batch": 1, "context": 8, "steps": 10, "warmup": 3, "new_tokens": 8},
        "generate": {"batch": 1, "context": 8, "steps": 10, "warmup": 3, "new_tokens": 8},
    },
    "model-s": {
        "train": {"batch": 1, "context": 2, "steps": 3, "warmup": 1, "new_tokens": 2},
        "generate": {"batch": 1, "context": 4, "steps": 3, "warmup": 1, "new_tokens": 2},
    },
    "model-m": {
        "train": {"batch": 1, "context": 1, "steps": 3, "warmup": 1, "new_tokens": 2},
        "generate": {"batch": 1, "context": 4, "steps": 3, "warmup": 1, "new_tokens": 2},
    },
}


def comma_list(text: str, allowed: set[str], name: str) -> list[str]:
    values = text.split(",")
    if not values or any(value not in allowed for value in values):
        raise argparse.ArgumentTypeError(
            f"{name} must be a comma-separated subset of {','.join(sorted(allowed))}"
        )
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(f"{name} cannot contain duplicates")
    return values


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--profiles", default="tiny,model-s,model-m")
    parser.add_argument("--modes", default="train,generate")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--measurement-profile", choices=("smoke", "comparison"),
                        default="smoke")
    parser.add_argument("--allow-amdsmi-fallback", action="store_true")
    parser.add_argument("--worker-profile", choices=tuple(PROFILES), help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=("train", "generate"), help=argparse.SUPPRESS)
    result = parser.parse_args()
    result.profiles = comma_list(result.profiles, set(PROFILES), "profiles")
    result.modes = comma_list(result.modes, {"train", "generate"}, "modes")
    return result


def prepare_device(name: str, allow_fallback: bool) -> tuple[torch.device, str]:
    workaround = "none"
    if name == "cuda" and torch.version.hip:
        public_count = torch.cuda.device_count()
        runtime_count = torch._C._cuda_getDeviceCount()
        if public_count == 0 and runtime_count > 0:
            if not allow_fallback:
                raise RuntimeError(
                    "AMDSMI reports zero devices while HIP runtime sees devices; "
                    "pass --allow-amdsmi-fallback to use the explicit runtime fallback"
                )
            torch.cuda._device_count_amdsmi = lambda: -1
            workaround = "amdsmi_zero_fallback_to_hip_runtime"
    if name == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() == 0):
        raise RuntimeError("PyTorch CUDA/ROCm device requested but unavailable")
    return torch.device(name), workaround


def rms_norm(value: torch.Tensor, weight: torch.Tensor, epsilon: float = 1.0e-5) -> torch.Tensor:
    variance = value.float().square().mean(dim=-1, keepdim=True)
    normalized = value.float() * torch.rsqrt(variance + epsilon)
    return (normalized * weight.float()).to(value.dtype)


def rope(value: torch.Tensor, offset: int, base: float) -> torch.Tensor:
    width = value.shape[-1]
    positions = torch.arange(offset, offset + value.shape[-2], device=value.device,
                             dtype=torch.float32)
    frequencies = base ** (-torch.arange(0, width, 2, device=value.device,
                                          dtype=torch.float32) / width)
    angles = positions[:, None] * frequencies[None, :]
    cosine = torch.cos(angles).to(value.dtype).view(1, 1, value.shape[-2], width // 2)
    sine = torch.sin(angles).to(value.dtype).view(1, 1, value.shape[-2], width // 2)
    even = value[..., 0::2]
    odd = value[..., 1::2]
    return torch.stack((even * cosine - odd * sine,
                        even * sine + odd * cosine), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        dimension = config.dimension
        kv_dimension = config.kv_heads * config.head_dimension
        self.attention_norm = nn.Parameter(torch.ones(dimension))
        self.q_proj = nn.Linear(dimension, dimension, bias=False)
        self.k_proj = nn.Linear(dimension, kv_dimension, bias=False)
        self.v_proj = nn.Linear(dimension, kv_dimension, bias=False)
        self.o_proj = nn.Linear(dimension, dimension, bias=False)
        self.ffn_norm = nn.Parameter(torch.ones(dimension))
        self.gate_proj = nn.Linear(dimension, config.ffn_dimension, bias=False)
        self.up_proj = nn.Linear(dimension, config.ffn_dimension, bias=False)
        self.down_proj = nn.Linear(config.ffn_dimension, dimension, bias=False)

    def forward(self, hidden: torch.Tensor, position_offset: int = 0,
                state: tuple[torch.Tensor, torch.Tensor] | None = None,
                return_cache: bool = False):
        batch, sequence, _ = hidden.shape
        normalized = rms_norm(hidden, self.attention_norm)
        query = self.q_proj(normalized).view(
            batch, sequence, self.config.heads, self.config.head_dimension
        ).transpose(1, 2)
        key = self.k_proj(normalized).view(
            batch, sequence, self.config.kv_heads, self.config.head_dimension
        ).transpose(1, 2)
        value = self.v_proj(normalized).view(
            batch, sequence, self.config.kv_heads, self.config.head_dimension
        ).transpose(1, 2)
        query = rope(query, position_offset, self.config.rope_base)
        key = rope(key, position_offset, self.config.rope_base)
        if state is not None:
            key = torch.cat((state[0], key), dim=2)
            value = torch.cat((state[1], value), dim=2)
        new_state = (key, value)
        repeats = self.config.heads // self.config.kv_heads
        if repeats != 1:
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        context = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=state is None and sequence > 1
        )
        attention = self.o_proj(context.transpose(1, 2).contiguous().view(
            batch, sequence, self.config.dimension
        ))
        residual = hidden + attention
        normalized_ffn = rms_norm(residual, self.ffn_norm)
        ffn = self.down_proj(F.silu(self.gate_proj(normalized_ffn)) *
                             self.up_proj(normalized_ffn))
        output = residual + ffn
        return (output, new_state) if return_cache else output


class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocabulary_size, config.dimension)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.final_norm = nn.Parameter(torch.ones(config.dimension))
        self.output_head = nn.Linear(config.dimension, config.vocabulary_size, bias=False)

    def forward(self, tokens: torch.Tensor, position_offset: int = 0,
                states: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
                return_cache: bool = False):
        hidden = self.embedding(tokens)
        new_states = []
        for index, block in enumerate(self.blocks):
            state = None if states is None else states[index]
            if return_cache:
                hidden, new_state = block(hidden, position_offset, state, True)
                new_states.append(new_state)
            else:
                hidden = block(hidden, position_offset, state, False)
        logits = self.output_head(rms_norm(hidden, self.final_norm))
        return (logits, new_states) if return_cache else logits


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def current_rss_bytes() -> int:
    with open("/proc/self/statm", encoding="utf-8") as stream:
        resident_pages = int(stream.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def clear_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def fixed_batch(config: Config, batch: int, context: int, device: torch.device):
    values = torch.arange(batch * context, device=device).reshape(batch, context)
    inputs = (values % config.vocabulary_size).long()
    targets = ((inputs + 1) % config.vocabulary_size).long()
    return inputs, targets


@torch.inference_mode()
def generate(model: Decoder, prompt: torch.Tensor, new_tokens: int) -> list[int]:
    logits, states = model(prompt, return_cache=True)
    generated: list[int] = []
    next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
    generated.append(int(next_token.item()))
    for index in range(1, new_tokens):
        logits, states = model(next_token, position_offset=prompt.shape[1] + index - 1,
                               states=states, return_cache=True)
        next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        generated.append(int(next_token.item()))
    return generated


def memory_fields(device: torch.device) -> dict:
    current_rss = current_rss_bytes()
    peak_rss = max(current_rss, peak_rss_bytes())
    fields = {
        "process_current_rss_bytes": current_rss,
        "process_peak_rss_bytes": peak_rss,
    }
    if device.type == "cuda":
        fields.update({
            "device_current_allocated_bytes": torch.cuda.memory_allocated(device),
            "device_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "device_current_reserved_bytes": torch.cuda.memory_reserved(device),
            "device_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        })
    else:
        fields.update({
            "device_current_allocated_bytes": fields["process_current_rss_bytes"],
            "device_peak_allocated_bytes": fields["process_peak_rss_bytes"],
            "device_current_reserved_bytes": fields["process_current_rss_bytes"],
            "device_peak_reserved_bytes": fields["process_peak_rss_bytes"],
        })
    return fields


def measure(profile: str, mode: str, device: torch.device, workaround: str,
            measurement_profile: str) -> dict:
    config = CONFIGS[profile]
    settings = (PROFILES[profile][mode] if measurement_profile == "smoke"
                else COMPARISON_SETTINGS[profile][mode])
    clear_device(device)
    torch.manual_seed(20260819)
    wall_start = time.perf_counter()
    model = Decoder(config).to(device=device, dtype=torch.float32)
    synchronize(device)
    construction_finish = time.perf_counter()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != PROFILES[profile]["parameter_count"]:
        raise RuntimeError(f"{profile} PyTorch parameter count changed: {parameter_count}")

    first_loss = 0.0
    final_loss = 0.0
    output_guard = 0.0
    if mode == "train":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8,
            weight_decay=0.01
        )
        inputs, targets = fixed_batch(config, settings["batch"], settings["context"], device)

        def step() -> float:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, config.vocabulary_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            return float(loss.detach())

        warmup_start = time.perf_counter()
        for _ in range(settings["warmup"]):
            step()
        synchronize(device)
        warmup_finish = time.perf_counter()
        measured_start = time.perf_counter()
        for iteration in range(settings["steps"]):
            loss_value = step()
            if iteration == 0:
                first_loss = loss_value
            final_loss = loss_value
        synchronize(device)
        measured_finish = time.perf_counter()
        measured_tokens = settings["steps"] * settings["batch"] * settings["context"]
        output_guard = final_loss
    else:
        prompt = torch.ones((1, settings["context"]), dtype=torch.long, device=device)
        warmup_start = time.perf_counter()
        for _ in range(settings["warmup"]):
            generate(model, prompt, settings["new_tokens"])
        synchronize(device)
        warmup_finish = time.perf_counter()
        measured_start = time.perf_counter()
        for _ in range(settings["steps"]):
            output_guard += sum(generate(model, prompt, settings["new_tokens"]))
        synchronize(device)
        measured_finish = time.perf_counter()
        measured_tokens = settings["steps"] * settings["new_tokens"]

    wall_finish = time.perf_counter()
    measured_seconds = measured_finish - measured_start
    total_seconds = wall_finish - wall_start
    record = {
        "schema_version": 1,
        "record_type": "single_gpu_pytorch_model_measurement",
        "status": "pass",
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else "host CPU"),
        "architecture": (torch.cuda.get_device_properties(device).gcnArchName
                         if device.type == "cuda" else "host"),
        "device_discovery_workaround": workaround,
        "dtype": "float32",
        "model": profile,
        "mode": mode,
        "measurement_profile": measurement_profile,
        "parameter_count": parameter_count,
        "fp32_weight_bytes": parameter_count * 4,
        "batch": settings["batch"],
        "context": settings["context"],
        "steps": settings["steps"],
        "warmup": settings["warmup"],
        "new_tokens": settings["new_tokens"],
        "measured_tokens": measured_tokens,
        "measured_wall_seconds": measured_seconds,
        "tokens_per_second": measured_tokens / measured_seconds,
        "milliseconds_per_token": measured_seconds * 1000.0 / measured_tokens,
        "model_construction_seconds": construction_finish - wall_start,
        "warmup_seconds": warmup_finish - warmup_start,
        "wall_seconds_with_setup": total_seconds,
        "tokens_per_second_with_setup": measured_tokens / total_seconds,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "output_guard": output_guard,
        **memory_fields(device),
    }
    numeric = ("tokens_per_second", "milliseconds_per_token", "device_peak_allocated_bytes")
    if any(not math.isfinite(float(record[field])) or float(record[field]) <= 0 for field in numeric):
        raise RuntimeError(f"{profile}/{mode} emitted invalid metrics")
    return record


def main() -> int:
    args = options()
    if (args.worker_profile is None) != (args.worker_mode is None):
        raise RuntimeError("worker profile and mode must be provided together")
    device, workaround = prepare_device(args.device, args.allow_amdsmi_fallback)
    if args.worker_profile is not None:
        print(json.dumps(measure(args.worker_profile, args.worker_mode, device, workaround,
                                 args.measurement_profile),
                         sort_keys=True))
        return 0

    records = []
    for profile in args.profiles:
        for mode in args.modes:
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--device", args.device,
                "--worker-profile", profile,
                "--worker-mode", mode,
                "--measurement-profile", args.measurement_profile,
            ]
            if args.allow_amdsmi_fallback:
                command.append("--allow-amdsmi-fallback")
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            records.append(json.loads(completed.stdout))
    summary = {
        "schema_version": 1,
        "record_type": "single_gpu_pytorch_model_matrix_summary",
        "status": "pass",
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "device": str(device),
        "profiles": args.profiles,
        "modes": args.modes,
        "measurement_count": len(records),
        "measurement_profile": args.measurement_profile,
        "device_discovery_workaround": workaround,
    }
    lines = [*(json.dumps(record, sort_keys=True) for record in records),
             json.dumps(summary, sort_keys=True)]
    output = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
