#!/usr/bin/env python3
"""Compare every official-model hidden state with PyTorch on the same FP32 weights."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import subprocess
from pathlib import Path


BLOCK = re.compile(r"^inference\.blocks\.(\d+)$")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--models")
    parser.add_argument("--context", type=int, default=4)
    parser.add_argument("--trace-max-elements", type=int, default=200000)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    result = parser.parse_args()
    if not result.manifest.is_file() or not result.binary.is_file():
        parser.error("manifest and binary must exist")
    if result.context <= 0 or result.trace_max_elements <= 0 or \
            result.timeout_seconds <= 0:
        parser.error("numeric options must be positive")
    result.models = result.models.split(",") if result.models else None
    return result


def load_models(path: Path, selected: list[str] | None) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    models = document.get("models", [])
    if selected is not None:
        models = [model for model in models if model["name"] in selected]
        missing = sorted(set(selected) - {model["name"] for model in models})
        if missing:
            raise RuntimeError(f"models absent from manifest: {missing}")
    for model in models:
        for field in ("config", "weights"):
            if not Path(model[field]).is_file():
                raise RuntimeError(f"{model['name']} {field} is unavailable")
    if not models:
        raise RuntimeError("no models selected")
    return models


def engine_command(binary: Path, model: dict, tokens: list[int],
                   trace: Path, trace_max_elements: int) -> list[str]:
    return [
        str(binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", ",".join(map(str, tokens)), "--device", "hip",
        "--top-k", "1", "--new-tokens", "0", "--workload", "prefill",
        "--batch", "1", "--bf16-ffn", "false", "--bf16-attention", "false",
        "--prefill-logits", "last", "--prefill-warmup", "0",
        "--prefill-steps", "1", "--trace-output", str(trace),
        "--trace-max-elements", str(trace_max_elements),
        "--trace-value-filter",
        "inference.embedding,inference.blocks.,inference.final_norm,inference.logits",
    ]


def selected_engine_trace(path: Path) -> dict[str, dict]:
    records = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    selected = {}
    for row in records:
        if (row["name"] in ("inference.embedding", "inference.final_norm",
                            "inference.logits") or BLOCK.fullmatch(row["name"])):
            if row["name"] in selected:
                raise RuntimeError(f"duplicate engine trace stage: {row['name']}")
            if row.get("values_truncated") or not row.get("values"):
                raise RuntimeError(f"engine trace is incomplete: {row['name']}")
            selected[row["name"]] = row
    return selected


def pytorch_trace(model_record: dict, tokens: list[int]) -> dict[str, dict]:
    import torch
    from transformers import AutoModelForCausalLM

    model_path = str(Path(model_record["config"]).parent)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager").to("cuda").eval()
    captures: dict[str, torch.Tensor] = {}
    handles = []

    def capture(name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            captures[name] = tensor.detach().float().cpu().contiguous()
        return hook

    handles.append(model.model.embed_tokens.register_forward_hook(
        capture("inference.embedding")))
    for index, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(
            capture(f"inference.blocks.{index}")))
    handles.append(model.model.norm.register_forward_hook(
        capture("inference.final_norm")))
    handles.append(model.lm_head.register_forward_hook(
        capture("inference.logits.full")))
    input_ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False, return_dict=True)
    torch.cuda.synchronize()
    for handle in handles:
        handle.remove()
    full_logits = captures.pop("inference.logits.full")
    captures["inference.logits"] = full_logits[:, -1:, :].contiguous()
    result = {
        name: {
            "shape": list(tensor.shape),
            "values": tensor.reshape(-1).tolist(),
        }
        for name, tensor in captures.items()
    }
    del input_ids, full_logits, captures, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def difference(reference: list[float], actual: list[float]) -> dict:
    if len(reference) != len(actual) or not reference:
        raise RuntimeError("comparison requires equal non-empty vectors")
    maximum = 0.0
    maximum_index = 0
    square = 0.0
    reference_square = 0.0
    absolute_sum = 0.0
    for index, (left, right) in enumerate(zip(reference, actual)):
        delta = abs(float(left) - float(right))
        if delta > maximum:
            maximum = delta
            maximum_index = index
        absolute_sum += delta
        square += delta * delta
        reference_square += float(reference[index]) * float(reference[index])
    return {
        "elements": len(reference),
        "max_abs": maximum,
        "max_abs_index": maximum_index,
        "mean_abs": absolute_sum / len(reference),
        "rms_abs": math.sqrt(square / len(reference)),
        "relative_l2": math.sqrt(square / reference_square)
        if reference_square > 0 else 0.0,
        "exact": maximum == 0.0,
    }


def stage_order(name: str) -> int:
    if name == "inference.embedding":
        return -1
    match = BLOCK.fullmatch(name)
    if match:
        return int(match.group(1))
    if name == "inference.final_norm":
        return 1_000_000
    if name == "inference.logits":
        return 1_000_001
    raise RuntimeError(f"unknown stage: {name}")


def compare(engine: dict[str, dict], pytorch: dict[str, dict]) -> list[dict]:
    if set(engine) != set(pytorch):
        raise RuntimeError(
            f"stage names differ: engine-only={sorted(set(engine)-set(pytorch))} "
            f"pytorch-only={sorted(set(pytorch)-set(engine))}")
    rows = []
    for name in sorted(engine, key=stage_order):
        left, right = pytorch[name], engine[name]
        if [int(value) for value in right["shape"]] != left["shape"]:
            raise RuntimeError(
                f"stage shape differs at {name}: {left['shape']} vs {right['shape']}")
        rows.append({
            "name": name,
            "shape": left["shape"],
            **difference(left["values"], right["values"]),
        })
    return rows


def main() -> int:
    args = options()
    models = load_models(args.manifest, args.models)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    summaries = []
    for model in models:
        seed = [int(token) for token in model["inference"]["token_ids"]]
        tokens = [seed[index % len(seed)] for index in range(args.context)]
        trace_path = args.output_directory / f"{model['name']}-engine-trace.jsonl"
        completed = subprocess.run(
            engine_command(args.binary, model, tokens, trace_path,
                           args.trace_max_elements),
            capture_output=True, text=True, timeout=args.timeout_seconds)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        application = [json.loads(line) for line in completed.stdout.splitlines()
                       if line.strip()]
        if len(application) != 1 or application[0].get("status") != "pass":
            raise RuntimeError("engine application output contract failed")
        engine = selected_engine_trace(trace_path)
        trace_path.unlink()
        pytorch = pytorch_trace(model, tokens)
        stages = compare(engine, pytorch)
        first_nonzero = next((row["name"] for row in stages if not row["exact"]), None)
        maximum = max(stages, key=lambda row: row["relative_l2"])
        logits = next(row for row in stages if row["name"] == "inference.logits")
        summary = {
            "schema_version": 1,
            "status": "pass",
            "record_type": "official_pytorch_hidden_alignment",
            "model": model["name"],
            "revision": model["revision"],
            "context": args.context,
            "tokens": tokens,
            "stage_count": len(stages),
            "first_nonzero_stage": first_nonzero,
            "maximum_relative_l2_stage": maximum["name"],
            "maximum_relative_l2": maximum["relative_l2"],
            "logits_max_abs": logits["max_abs"],
            "logits_rms_abs": logits["rms_abs"],
            "engine_trace_record_count": application[0]["trace_record_count"],
            "stages": stages,
            "boundary": "synchronous complete FP32 hidden snapshots; no performance claim",
        }
        summaries.append(summary)
        with raw_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, sort_keys=True) + "\n")
        print(json.dumps({"model": model["name"], "status": "pass",
                          "first_nonzero_stage": first_nonzero}, sort_keys=True),
              flush=True)
    document = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "official_pytorch_hidden_alignment_matrix",
        "context": args.context,
        "models": [row["model"] for row in summaries],
        "summaries": summaries,
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
