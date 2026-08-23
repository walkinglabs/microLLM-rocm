#!/usr/bin/env python3
"""Audit scalar versus output-channel FP8 reconstruction of HF Linear weights."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


LINEAR_GROUPS = {
    "self_attn.q_proj.weight": "attention",
    "self_attn.k_proj.weight": "attention",
    "self_attn.v_proj.weight": "attention",
    "self_attn.o_proj.weight": "attention",
    "mlp.gate_proj.weight": "ffn",
    "mlp.up_proj.weight": "ffn",
    "mlp.down_proj.weight": "ffn",
    "lm_head.weight": "output_head",
}


def classify_weight(name: str) -> str | None:
    for suffix, group in LINEAR_GROUPS.items():
        if name.endswith(suffix):
            return group
    return None


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["group"])].append(row)
        grouped[(row["model"], "all_linear")].append(row)
    result = []
    for (model, group), selected in sorted(grouped.items()):
        scalar_sse = sum(row["scalar_squared_error"] for row in selected)
        column_sse = sum(row["column_squared_error"] for row in selected)
        reference_sse = sum(row["reference_squared_sum"] for row in selected)
        elements = sum(row["elements"] for row in selected)
        scalar_relative_l2 = math.sqrt(scalar_sse / reference_sse)
        column_relative_l2 = math.sqrt(column_sse / reference_sse)
        result.append({
            "model": model,
            "group": group,
            "tensors": len(selected),
            "elements": elements,
            "scalar_relative_l2": scalar_relative_l2,
            "column_relative_l2": column_relative_l2,
            "column_over_scalar_relative_l2":
                column_relative_l2 / scalar_relative_l2,
            "scalar_rms": math.sqrt(scalar_sse / elements),
            "column_rms": math.sqrt(column_sse / elements),
            "scalar_max_abs": max(row["scalar_max_abs"] for row in selected),
            "column_max_abs": max(row["column_max_abs"] for row in selected),
        })
    return result


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--models")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-scale", type=float, default=0.0001)
    args = parser.parse_args()
    if not args.manifest.is_file() or not math.isfinite(args.minimum_scale) or \
            args.minimum_scale <= 0:
        parser.error("manifest and minimum scale are invalid")
    args.models = args.models.split(",") if args.models else None
    return args


def model_entries(path: Path, selected: list[str] | None) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    models = document.get("models", [])
    by_name = {row.get("name"): row for row in models}
    names = selected or list(by_name)
    if not models or None in by_name or len(by_name) != len(models) or \
            not set(names) <= set(by_name):
        raise RuntimeError("model manifest is invalid")
    return [by_name[name] for name in names]


def main() -> int:
    args = options()
    import torch
    from safetensors import safe_open

    if not torch.cuda.is_available() or not hasattr(torch, "float8_e4m3fnuz"):
        raise RuntimeError("ROCm PyTorch with float8_e4m3fnuz is required")
    device = torch.device(args.device)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    rows = []
    for model in model_entries(args.manifest, args.models):
        weights_path = Path(model["weights"])
        if not weights_path.is_file():
            raise RuntimeError(f"weight file is missing: {weights_path}")
        with safe_open(weights_path, framework="pt", device="cpu") as source:
            selected_keys = [key for key in source.keys()
                             if classify_weight(key) is not None]
            if not selected_keys:
                raise RuntimeError(f"no Linear weights found for {model['name']}")
            for name in selected_keys:
                weight = source.get_tensor(name)
                if weight.ndim != 2:
                    raise RuntimeError(f"Linear weight is not rank two: {name}")
                source_dtype = str(weight.dtype)
                weight = weight.to(device=device, dtype=torch.float32)
                maximum = weight.abs().amax()
                scalar_scale = torch.maximum(
                    maximum / 240.0,
                    torch.tensor(args.minimum_scale, device=device))
                output_scales = torch.clamp(
                    weight.abs().amax(dim=1, keepdim=True) / 240.0,
                    min=args.minimum_scale)
                scalar_restored = (
                    weight.div(scalar_scale).to(torch.float8_e4m3fnuz)
                    .to(torch.float32).mul(scalar_scale))
                scalar_difference = scalar_restored.sub(weight)
                scalar_squared_error = scalar_difference.square().sum(
                    dtype=torch.float64).item()
                scalar_max_abs = scalar_difference.abs().amax().item()
                del scalar_restored, scalar_difference
                column_restored = (
                    weight.div(output_scales).to(torch.float8_e4m3fnuz)
                    .to(torch.float32).mul(output_scales))
                column_difference = column_restored.sub(weight)
                column_squared_error = column_difference.square().sum(
                    dtype=torch.float64).item()
                column_max_abs = column_difference.abs().amax().item()
                reference_squared_sum = weight.square().sum(
                    dtype=torch.float64).item()
                sorted_scales = output_scales.flatten().sort().values
                middle_scale = sorted_scales[sorted_scales.numel() // 2].item()
                row = {
                    "schema_version": 1,
                    "status": "pass",
                    "track": "external_fp8_weight_reconstruction_audit",
                    "model": model["name"],
                    "revision": model["revision"],
                    "name": name,
                    "group": classify_weight(name),
                    "shape": list(weight.shape),
                    "elements": weight.numel(),
                    "scalar_scale": scalar_scale.item(),
                    "column_scale_min": output_scales.amin().item(),
                    "column_scale_median": middle_scale,
                    "column_scale_max": output_scales.amax().item(),
                    "column_scale_max_over_median":
                        output_scales.amax().item() / middle_scale,
                    "scalar_squared_error": scalar_squared_error,
                    "column_squared_error": column_squared_error,
                    "reference_squared_sum": reference_squared_sum,
                    "scalar_relative_l2": math.sqrt(
                        scalar_squared_error / reference_squared_sum),
                    "column_relative_l2": math.sqrt(
                        column_squared_error / reference_squared_sum),
                    "column_over_scalar_relative_l2": math.sqrt(
                        column_squared_error / scalar_squared_error),
                    "scalar_max_abs": scalar_max_abs,
                    "column_max_abs": column_max_abs,
                    "dtype_source": source_dtype,
                    "dtype_compute": "torch.float32",
                    "dtype_quantized": "torch.float8_e4m3fnuz",
                }
                rows.append(row)
                with raw_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)
                del weight, output_scales, column_restored, column_difference
                del scalar_scale, sorted_scales
    aggregates = summarize(rows)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "track": "external_fp8_weight_reconstruction_audit",
        "models": [row["name"] for row in model_entries(args.manifest, args.models)],
        "minimum_scale": args.minimum_scale,
        "format": "torch.float8_e4m3fnuz",
        "weight_layout": (
            "HF [output,input]; output-channel rows correspond to microLLM "
            "transposed [input,output] columns"),
        "boundary": (
            "External PyTorch ROCm reconstruction diagnostic only; it selects a model scope but "
            "does not replace microLLM native full-logit evidence"),
        "tensor_rows": len(rows),
        "aggregates": aggregates,
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
