#!/usr/bin/env python3
"""Sequential-request PyTorch reference for an official continuous workload.

This is intentionally named sequential: it validates tokens and supplies a
framework baseline without pretending PyTorch executes microLLM's slot scheduler.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def positive_list(text: str, name: str) -> list[int]:
    try:
        values = [int(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must contain integers") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{name} must contain positive integers")
    return values


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-lengths", required=True)
    parser.add_argument("--new-token-lengths", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    result = parser.parse_args()
    result.prompt_lengths = positive_list(result.prompt_lengths, "prompt lengths")
    result.new_token_lengths = positive_list(
        result.new_token_lengths, "new-token lengths")
    if len(result.prompt_lengths) != len(result.new_token_lengths):
        parser.error("prompt and new-token lists must have equal length")
    if result.warmup < 0 or result.steps <= 0:
        parser.error("warmup must be nonnegative and steps positive")
    return result


def load_model_record(path: Path, name: str) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    models = document.get("models") if document.get("schema_version") == 1 else None
    if not isinstance(models, list):
        raise RuntimeError("manifest must contain schema-version-1 models")
    matches = [model for model in models if model.get("name") == name]
    if len(matches) != 1:
        raise RuntimeError("selected model is missing or duplicate")
    return matches[0]


def checksum(rows: list[list[int]]) -> int:
    value = 0
    for row in rows:
        for token in row:
            value = (value * 131 + int(token)) & ((1 << 64) - 1)
    return value


def main() -> int:
    args = options()
    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm device is unavailable")
    model_record = load_model_record(args.manifest, args.model)
    device = torch.device("cuda:0")
    loaded = AutoModelForCausalLM.from_pretrained(
        Path(model_record["config"]).parent, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="sdpa").to(device).eval()
    parameter_count = sum(parameter.numel() for parameter in loaded.parameters())
    if parameter_count != int(model_record["parameter_count"]):
        raise RuntimeError("PyTorch parameter count changed")
    seed = [int(token) for token in model_record["inference"]["token_ids"]]
    prompts = []
    for request, length in enumerate(args.prompt_lengths):
        prompts.append([seed[(index + request) % len(seed)] for index in range(length)])

    def generate_all() -> list[list[int]]:
        generated = []
        for prompt, output_length in zip(prompts, args.new_token_lengths):
            input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
            prepared = loaded(input_ids=input_ids, use_cache=True, logits_to_keep=1)
            selected = torch.argmax(prepared.logits[:, -1, :], dim=-1)
            suffix = [int(selected.item())]
            past = prepared.past_key_values
            for _ in range(1, output_length):
                current = loaded(input_ids=selected[:, None],
                                 past_key_values=past, use_cache=True,
                                 logits_to_keep=1)
                selected = torch.argmax(current.logits[:, -1, :], dim=-1)
                suffix.append(int(selected.item()))
                past = current.past_key_values
            generated.append(suffix)
        return generated

    with torch.inference_mode():
        for _ in range(args.warmup):
            generate_all()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        elapsed_ms = 0.0
        expected = None
        for _ in range(args.steps):
            start = time.perf_counter()
            current = generate_all()
            torch.cuda.synchronize(device)
            elapsed_ms += (time.perf_counter() - start) * 1000.0
            if expected is None:
                expected = current
            elif current != expected:
                raise RuntimeError("PyTorch sequential generation changed across steps")
    measured_tokens = sum(args.new_token_lengths) * args.steps
    peak = int(torch.cuda.max_memory_allocated(device))
    resident = sum(parameter.numel() * parameter.element_size()
                   for parameter in loaded.parameters())
    device_total = int(torch.cuda.get_device_properties(device).total_memory)
    result = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "pytorch_sequential_request_reference",
        "framework": "pytorch",
        "serving_mode": "sequential_requests",
        "model": model_record["name"],
        "revision": model_record["revision"],
        "device": str(device),
        "parameter_count": parameter_count,
        "precision": "full_bf16_model",
        "request_count": len(prompts),
        "prompt_lengths": args.prompt_lengths,
        "new_token_lengths": args.new_token_lengths,
        "warmup": args.warmup,
        "steps": args.steps,
        "measured_tokens": measured_tokens,
        "measured_ms": elapsed_ms,
        "tokens_per_second": measured_tokens * 1000.0 / elapsed_ms,
        "resident_weight_bytes": resident,
        "peak_bytes": peak,
        "peak_memory_share_of_device": peak / device_total,
        "device_total_bytes": device_total,
        "generated_tokens": expected,
        "token_checksum": checksum(expected),
        "deterministic_across_steps": True,
        "comparison_boundary": "sequential requests; not a variable-position slot scheduler",
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
