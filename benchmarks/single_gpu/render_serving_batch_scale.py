#!/usr/bin/env python3
"""Render and derive the fixed T2048 serving batch-scale experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


MODELS = ("qwen2.5-0.5b", "deepseek-r1-distill-qwen-1.5b")
BATCHES = (1, 2, 4, 8)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if not args.summary.is_file():
        parser.error("summary does not exist")
    return args


def finite(row: dict, name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
            not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} is not a finite positive measurement")
    return float(value)


def analyze(summary: dict) -> dict:
    if (summary.get("schema_version") != 1 or
            summary.get("track") != "official_inference_shape_matrix" or
            summary.get("status") != "pass" or
            summary.get("axes") != {
                "contexts": [2048], "batches": [1, 2, 4, 8],
                "decode_lengths": [64], "cases": ["cached"]} or
            summary.get("runs_per_framework") != 3):
        raise ValueError("serving batch summary identity changed")
    rows = summary.get("rows", [])
    by_key = {(row.get("model"), row.get("batch")): row for row in rows}
    if len(rows) != 8 or set(by_key) != {
            (model, batch) for model in MODELS for batch in BATCHES}:
        raise ValueError("serving batch summary is incomplete")
    result_rows = []
    for model in MODELS:
        for batch in BATCHES:
            row = by_key[(model, batch)]
            if row.get("status") != "pass":
                raise ValueError("serving batch case is not complete")
            micro_tps = finite(row, "microllm_throughput_tokens_per_second")
            torch_tps = finite(row, "pytorch_throughput_tokens_per_second")
            result_rows.append({
                "model": model, "batch": batch,
                "microllm_tokens_per_second": micro_tps,
                "pytorch_tokens_per_second": torch_tps,
                "microllm_over_pytorch": micro_tps / torch_tps,
                "microllm_batch_scaling": finite(
                    row, "microllm_batch_throughput_scaling"),
                "microllm_batch_efficiency": finite(
                    row, "microllm_batch_efficiency"),
                "pytorch_batch_scaling": finite(
                    row, "pytorch_batch_throughput_scaling"),
                "pytorch_batch_efficiency": finite(
                    row, "pytorch_batch_efficiency"),
                "microllm_latency_ms": finite(row, "microllm_latency_ms"),
                "pytorch_latency_ms": finite(row, "pytorch_latency_ms"),
                "microllm_peak_bytes": int(finite(row, "microllm_peak_bytes")),
                "pytorch_peak_bytes": int(finite(row, "pytorch_peak_bytes")),
                "microllm_peak_bytes_per_request": int(finite(
                    row, "microllm_peak_bytes_per_request")),
                "pytorch_peak_bytes_per_request": int(finite(
                    row, "pytorch_peak_bytes_per_request")),
                "kv_cache_bytes": int(finite(row, "microllm_kv_cache_actual_bytes")),
                "cross_framework_tokens_equal": bool(
                    row.get("cross_framework_tokens_equal")),
                "first_token_difference": int(
                    row.get("cross_framework_first_token_difference", -2)),
            })
    qwen = [row for row in result_rows if row["model"] == MODELS[0]]
    deep = [row for row in result_rows if row["model"] == MODELS[1]]
    return {
        "schema_version": 1,
        "record_type": "serving_batch_scale_analysis",
        "status": "pass",
        "context": 2048,
        "decode_tokens": 64,
        "batches": list(BATCHES),
        "process_rows": 48,
        "case_rows": 8,
        "microllm_auto_enabled_rows": 24,
        "pytorch_device_fallback_rows": 24,
        "qwen_b8_scaling": qwen[-1]["microllm_batch_scaling"],
        "qwen_b8_efficiency": qwen[-1]["microllm_batch_efficiency"],
        "qwen_b8_over_pytorch": qwen[-1]["microllm_over_pytorch"],
        "qwen_all_tokens_equal": all(
            row["cross_framework_tokens_equal"] for row in qwen),
        "deepseek_b4_scaling": deep[2]["microllm_batch_scaling"],
        "deepseek_b4_efficiency": deep[2]["microllm_batch_efficiency"],
        "deepseek_b4_over_pytorch": deep[2]["microllm_over_pytorch"],
        "deepseek_b8_scaling": deep[-1]["microllm_batch_scaling"],
        "deepseek_b8_efficiency": deep[-1]["microllm_batch_efficiency"],
        "deepseek_b8_over_pytorch": deep[-1]["microllm_over_pytorch"],
        "deepseek_equal_batches": [
            row["batch"] for row in deep if row["cross_framework_tokens_equal"]],
        "deepseek_divergent_batches": [
            row["batch"] for row in deep if not row["cross_framework_tokens_equal"]],
        "deepseek_first_difference": min(
            row["first_token_difference"] for row in deep
            if not row["cross_framework_tokens_equal"]),
        "scheduler_default_admitted": False,
        "next_experiment": "export complete microLLM cross-batch logits before any scheduler policy",
        "rows": result_rows,
    }


def render(analysis: dict) -> str:
    width, height = 1420, 760
    maximum = max(row["pytorch_tokens_per_second"]
                  for row in analysis["rows"])
    maximum = max(maximum, max(row["microllm_tokens_per_second"]
                               for row in analysis["rows"]))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.title{font-size:24px;font-weight:700}.label{font-size:14px}'
        '.small{font-size:12px}.muted{fill:#94a3b8}</style>',
        '<text x="34" y="42" class="title">T2048/N64 serving batch scale · MI300X</text>',
        '<text x="34" y="68" class="small muted">microLLM exact Auto vs PyTorch ROCm · '
        'bars=tokens/s · labels=efficiency / peak per request</text>',
    ]
    for panel, model in enumerate(MODELS):
        x = 34 + panel * 690
        title = "Qwen2.5-0.5B" if panel == 0 else "DeepSeek-Distill-1.5B"
        parts.append(f'<text x="{x}" y="108" class="label">{title}</text>')
        rows = [row for row in analysis["rows"] if row["model"] == model]
        for index, row in enumerate(rows):
            y = 132 + index * 135
            micro = 500 * row["microllm_tokens_per_second"] / maximum
            torch = 500 * row["pytorch_tokens_per_second"] / maximum
            token_color = "#22c55e" if row["cross_framework_tokens_equal"] \
                else "#ef4444"
            parts.extend((
                f'<text x="{x}" y="{y + 18}" class="label">B{row["batch"]}</text>',
                f'<rect x="{x + 48}" y="{y}" width="{micro:.2f}" height="25" '
                'rx="4" fill="#38bdf8"/>',
                f'<rect x="{x + 48}" y="{y + 34}" width="{torch:.2f}" height="25" '
                'rx="4" fill="#a78bfa"/>',
                f'<text x="{x + 58 + micro:.2f}" y="{y + 18}" class="small">'
                f'micro {row["microllm_tokens_per_second"]:.1f}</text>',
                f'<text x="{x + 58 + torch:.2f}" y="{y + 52}" class="small">'
                f'torch {row["pytorch_tokens_per_second"]:.1f}</text>',
                f'<text x="{x + 48}" y="{y + 83}" class="small muted">'
                f'eff {row["microllm_batch_efficiency"] * 100:.1f}% · '
                f'peak/req {row["microllm_peak_bytes_per_request"] / 2**30:.2f} GiB</text>',
                f'<circle cx="{x + 600}" cy="{y + 45}" r="7" fill="{token_color}"/>',
                f'<text x="{x + 615}" y="{y + 50}" class="small">'
                f'{row["microllm_over_pytorch"]:.3f}x · '
                f'tokens {"exact" if row["cross_framework_tokens_equal"] else "diverge"}</text>',
            ))
    parts.extend((
        '<rect x="34" y="690" width="1352" height="45" rx="10" '
        'fill="#111827" stroke="#334155"/>',
        '<text x="55" y="719" class="small">Green dot = cross-framework 64-token equality; '
        'red dot = precision investigation required before scheduler defaults.</text>',
        '</svg>',
    ))
    return "\n".join(parts) + "\n"


def main() -> int:
    args = options()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    analysis = analyze(summary)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "batch-scale.svg").write_text(
        render(analysis), encoding="utf-8")
    print(json.dumps(analysis, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"render_serving_batch_scale: {error}", file=sys.stderr)
        raise SystemExit(2) from error
