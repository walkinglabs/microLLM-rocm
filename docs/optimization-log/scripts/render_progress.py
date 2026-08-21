#!/usr/bin/env python3
"""Render deterministic, dependency-free SVGs for the optimization log."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results.tsv"
PROGRESS = ROOT / "assets" / "progress.svg"
BOTTLENECK = ROOT / "assets" / "bottleneck-map.svg"
BF16_RESULTS = ROOT / "bf16-results.tsv"
BF16_CHART = ROOT / "assets" / "bf16-gemm.svg"
BF16_POLICY_RESULTS = ROOT / "bf16-model-policy.tsv"
BF16_POLICY_CHART = ROOT / "assets" / "bf16-model-policy.svg"
BF16_FFN_SUMMARY = ROOT / "experiments" / "030-data" / "summary.json"
BF16_FFN_CHART = ROOT / "assets" / "bf16-ffn-island.svg"
BF16_MODEL_SUMMARY = ROOT / "experiments" / "031-data" / "summary.json"
BF16_MODEL_CHART = ROOT / "assets" / "bf16-model-inference.svg"
BF16_PREFILL_SUMMARY = ROOT / "experiments" / "032-data" / "summary.json"
BF16_PREFILL_CHART = ROOT / "assets" / "bf16-prefill-allocator.svg"
BF16_ATTENTION_SUMMARY = ROOT / "experiments" / "034-data" / "summary.json"
BF16_ATTENTION_PILOT = ROOT / "experiments" / "034-data" / "naive-pilot.jsonl"
BF16_ATTENTION_CHART = ROOT / "assets" / "bf16-attention.svg"
BF16_PLAN_SUMMARY = ROOT / "experiments" / "036-data" / "summary.json"
BF16_PLAN_CHART = ROOT / "assets" / "bf16-plan-cache.svg"
BF16_TRAINING_SUMMARY = ROOT / "experiments" / "037-data" / "summary.json"
BF16_TRAINING_CHART = ROOT / "assets" / "bf16-training.svg"
BF16_TRAINING_QKV_SUMMARY = ROOT / "experiments" / "039-data" / "summary.json"
BF16_TRAINING_QKV_CHART = ROOT / "assets" / "bf16-training-qkv-discard.svg"
BF16_TRAINING_MIRROR_SUMMARY = ROOT / "experiments" / "040-data" / "summary.json"
BF16_TRAINING_MIRROR_CHART = ROOT / "assets" / "bf16-training-mirrors.svg"
BF16_TRAINING_ISLAND_SUMMARY = ROOT / "experiments" / "041-data" / "summary.json"
BF16_TRAINING_ISLAND_CHART = ROOT / "assets" / "bf16-training-ffn-island-discard.svg"
BF16_TRAINING_SHAPE_SUMMARY = ROOT / "experiments" / "042-data" / "summary.json"
BF16_TRAINING_SHAPE_CHART = ROOT / "assets" / "bf16-training-shape-matrix.svg"
BF16_WEIGHT_GRADIENT_COMPARISON = ROOT / "experiments" / "043-data" / "comparison.json"
BF16_WEIGHT_GRADIENT_CHART = ROOT / "assets" / "bf16-weight-gradient-routing.svg"
FUSED_CAUSAL_GQA_COMPARISON = ROOT / "experiments" / "044-data" / "comparison.json"
FUSED_CAUSAL_GQA_CHART = ROOT / "assets" / "fused-causal-gqa-training.svg"
DEEPSEEK_SHAPE_SUMMARY = ROOT / "experiments" / "045-data" / "candidate" / "summary.json"
DEEPSEEK_LOAD_SUMMARY = ROOT / "experiments" / "045-data" / "load-summary.json"
DEEPSEEK_SHAPE_CHART = ROOT / "assets" / "deepseek-training-shapes.svg"
DEEPSEEK_PROFILE_SUMMARY = ROOT / "experiments" / "046-data" / "profile-summary.json"
DEEPSEEK_PROFILE_CHART = ROOT / "assets" / "deepseek-context128-profile.svg"
STABLE_GRADIENT_COMPARISON = ROOT / "experiments" / "047-data" / "comparison.json"
STABLE_GRADIENT_CHART = ROOT / "assets" / "stable-gradient-buffer-discard.svg"
CHUNKED_ADAMW_COMPARISON = ROOT / "experiments" / "048-data" / "comparison.json"
CHUNKED_ADAMW_CHART = ROOT / "assets" / "chunked-adamw-discard.svg"
VECTORIZED_ADAMW_COMPARISON = ROOT / "experiments" / "049-data" / "comparison.json"
VECTORIZED_ADAMW_CHART = ROOT / "assets" / "vectorized-adamw-explicit.svg"
STREAMING_LOAD_COMPARISON = ROOT / "experiments" / "050-data" / "comparison.json"
STREAMING_LOAD_CHART = ROOT / "assets" / "streaming-safetensors-load.svg"
CONTEXT512_COMPARISON = ROOT / "experiments" / "051-data" / "comparison.json"
CONTEXT512_PROFILE = ROOT / "experiments" / "051-data" / "profile-summary.json"
CONTEXT512_CHART = ROOT / "assets" / "context512-training-profile.svg"
SPLIT_KV_COMPARISON = ROOT / "experiments" / "052-data" / "comparison.json"
SPLIT_KV_CHART = ROOT / "assets" / "split-kv-backward-discard.svg"
BATCHED_GEMM_COMPARISON = ROOT / "experiments" / "053-data" / "comparison.json"
BATCHED_GEMM_CHART = ROOT / "assets" / "strided-batched-hipblaslt.svg"


def rows() -> list[dict]:
    with RESULTS.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(x: float, y: float, value: object, size: int = 18, color: str = "#172033",
         anchor: str = "start", weight: int = 400, rotate: int | None = None) -> str:
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter,Arial,sans-serif" '
            f'font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{transform}>{esc(value)}</text>')


def progress_svg(data: list[dict]) -> str:
    width, height = 1600, 900
    chart_x, chart_y, chart_w, chart_h = 90, 130, 930, 500
    bar_x, bar_y, bar_w, bar_h = 1100, 165, 420, 420
    max_experiment = max(12, max(int(row["experiment"]) for row in data) + 1)
    y_max = max(1.0, math.ceil(max(float(row["score"]) for row in data) * 2.0) / 2.0)

    def px(experiment: int) -> float:
        return chart_x + chart_w * experiment / max_experiment

    def py(score: float) -> float:
        return chart_y + chart_h * (y_max - score) / y_max

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "microLLM-rocm Optimization Progress", 30, anchor="middle", weight=700),
        text(width / 2, 79,
             f'{len(data)} measured experiment(s) · target: selected-matrix PyTorch parity',
             17, "#5b6474", anchor="middle"),
    ]

    # Main running-best plot.
    parts.append(f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
                 'fill="#ffffff" stroke="#cbd3df" rx="8"/>')
    tick_step = 0.5 if y_max > 1.5 else 0.25
    for index in range(int(round(y_max / tick_step)) + 1):
        tick = index * tick_step
        y = py(tick)
        parts.append(f'<line x1="{chart_x}" y1="{y:.1f}" x2="{chart_x + chart_w}" '
                     f'y2="{y:.1f}" stroke="#e5e9f0"/>')
        parts.append(text(chart_x - 14, y + 6, f"{tick:.2f}", 15, "#5b6474", anchor="end"))
    for tick in range(0, max_experiment + 1, 2):
        x = px(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{chart_y}" x2="{x:.1f}" '
                     f'y2="{chart_y + chart_h}" stroke="#f0f2f6"/>')
        parts.append(text(x, chart_y + chart_h + 28, tick, 14, "#5b6474", anchor="middle"))
    parity_y = py(1.0)
    parts.append(f'<line x1="{chart_x}" y1="{parity_y:.1f}" x2="{chart_x + chart_w}" '
                 f'y2="{parity_y:.1f}" stroke="#2563eb" stroke-width="2" '
                 'stroke-dasharray="9 7"/>')
    parts.append(text(chart_x + chart_w - 12, parity_y - 10, "PyTorch parity 1.0×", 15,
                      "#2563eb", anchor="end", weight=600))

    best_points: list[tuple[float, float]] = []
    running_best = -1.0
    colors = {"baseline": "#18a558", "keep": "#18a558", "discard": "#c8ced8",
              "crash": "#dc2626", "invalid": "#f97316"}
    for row in data:
        experiment = int(row["experiment"])
        score = float(row["score"])
        status = row["status"]
        x, y = px(experiment), py(score)
        if status in {"baseline", "keep"} and score > running_best:
            if best_points:
                old_x, old_y = best_points[-1]
                parts.append(f'<path d="M {old_x:.1f} {old_y:.1f} H {x:.1f} V {y:.1f}" '
                             'fill="none" stroke="#4ec27e" stroke-width="4"/>')
            best_points.append((x, y))
            running_best = score
        if status in {"crash", "invalid"}:
            parts.append(f'<path d="M {x-7:.1f} {y-7:.1f} L {x+7:.1f} {y+7:.1f} '
                         f'M {x+7:.1f} {y-7:.1f} L {x-7:.1f} {y+7:.1f}" '
                         f'stroke="{colors[status]}" stroke-width="4"/>')
        else:
            stroke = "#0e6938" if status in {"baseline", "keep"} else "#aeb6c2"
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" '
                         f'fill="{colors[status]}" stroke="{stroke}" stroke-width="2"/>')
        if status in {"baseline", "keep"}:
            parts.append(text(x + 12, y - 12, row["description"], 14, "#16834a",
                              rotate=-24))
    parts.append(text(chart_x + chart_w / 2, chart_y + chart_h + 65, "Experiment #", 17,
                      anchor="middle", weight=600))
    parts.append(text(28, chart_y + chart_h / 2, "Geometric throughput parity score", 17,
                      anchor="middle", weight=600, rotate=-90))

    # Legend.
    legend_y = 105
    for offset, (label, color) in enumerate((("Kept", "#18a558"),
                                              ("Discarded", "#c8ced8"),
                                              ("Crash / invalid", "#dc2626"))):
        x = 720 + offset * 150
        parts.append(f'<circle cx="{x}" cy="{legend_y}" r="6" fill="{color}"/>')
        parts.append(text(x + 12, legend_y + 5, label, 14, "#5b6474"))

    # Current workload bars use the latest kept/baseline row.
    current = next(row for row in reversed(data) if row["status"] in {"baseline", "keep"})
    workloads = (
        ("Qwen train", float(current["qwen_train"])),
        ("Qwen generate", float(current["qwen_generate"])),
        ("DeepSeek train", float(current["deepseek_train"])),
        ("DeepSeek generate", float(current["deepseek_generate"])),
    )
    parts.append(text(bar_x, 126, "Current workload parity", 21, weight=700))
    for index, (label, value) in enumerate(workloads):
        y = bar_y + index * 92
        parts.append(text(bar_x, y, label, 16, weight=600))
        parts.append(f'<rect x="{bar_x}" y="{y+14}" width="{bar_w}" height="24" '
                     'rx="5" fill="#e7ebf1"/>')
        parts.append(f'<rect x="{bar_x}" y="{y+14}" width="{bar_w*min(value,1.0):.1f}" '
                     'height="24" rx="5" fill="#4ec27e"/>')
        parts.append(text(bar_x + bar_w, y + 33, f"{value:.3f}×", 15, "#172033",
                          anchor="end", weight=700))
    parts.append(f'<line x1="{bar_x+bar_w}" y1="{bar_y+5}" x2="{bar_x+bar_w}" '
                 f'y2="{bar_y+bar_h}" stroke="#2563eb" stroke-width="2" '
                 'stroke-dasharray="6 5"/>')

    # Roadmap ribbon: labels are plans, not measured points.
    roadmap = (("M0", "Baseline", "complete"), ("M1", "Serial kernels", "complete"),
               ("M2", "Data movement", "complete"), ("M3", "Fused ops", "active"),
               ("M4", "BF16 / FP8", "planned"), ("M5", "HIP Graph", "planned"))
    box_w, gap, start_x, y = 220, 24, 90, 735
    parts.append(text(start_x, y - 25, "Roadmap (planned boxes are not results)", 18,
                      weight=700))
    for index, (milestone, label, status) in enumerate(roadmap):
        x = start_x + index * (box_w + gap)
        fill = "#e0f6e9" if status == "complete" else "#fff1dc" if status == "active" else "#f0f2f6"
        stroke = "#18a558" if status == "complete" else "#f97316" if status == "active" else "#c4cbd6"
        parts.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="88" rx="10" '
                     f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        parts.append(text(x + 18, y + 31, milestone, 18,
                          "#16834a" if status == "complete" else "#d45d00" if status == "active" else "#6b7280", weight=700))
        parts.append(text(x + 18, y + 61, label, 16, "#172033", weight=600))
    parts.append(text(90, 875, "Generated from docs/optimization-log/results.tsv · higher is better",
                      14, "#6b7280"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bottleneck_svg() -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="760" viewBox="0 0 1600 760">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(800, 45, "Baseline Bottleneck Map → Target Architecture", 30,
             anchor="middle", weight=700),
        text(400, 95, "Training: measured Qwen kernel share", 21, anchor="middle", weight=700),
        text(1200, 95, "Generation: measured Qwen kernel share", 21, anchor="middle", weight=700),
    ]

    def box(x, y, w, h, title, subtitle, fill, stroke):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
                     f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        parts.append(text(x + 18, y + 32, title, 18, "#172033", weight=700))
        parts.append(text(x + 18, y + 61, subtitle, 15, "#5b6474"))

    # Training column.
    box(90, 130, 620, 82, "CrossEntropy forward + backward", "75.7% · single GPU thread over vocabulary",
        "#fee2e2", "#dc2626")
    box(90, 232, 620, 82, "Weight/view strided copies", "8.4% · transpose().contiguous()",
        "#ffedd5", "#f97316")
    box(90, 334, 620, 82, "RMSNorm forward + backward", "10.2% · one thread per row",
        "#ffedd5", "#f97316")
    box(90, 436, 620, 82, "AdamW", "1.5% · already device native",
        "#e0f6e9", "#18a558")

    # Generation column.
    box(890, 130, 620, 82, "Tied output transpose copy", "43.4% · about 544 MB per cached forward",
        "#fee2e2", "#dc2626")
    box(890, 232, 620, 82, "RMSNorm", "37.7% · 539 serial row kernels in trace",
        "#fee2e2", "#dc2626")
    box(890, 334, 620, 82, "KV Cache + physical GQA expansion", "CPU concatenate / expand / copy back",
        "#ffedd5", "#f97316")
    box(890, 436, 620, 82, "Allocator and launch churn", "7407 alloc · 7403 free · 4099 launches",
        "#ffedd5", "#f97316")

    parts.append('<path d="M 400 545 V 590 H 600 V 618" fill="none" stroke="#64748b" '
                 'stroke-width="3" marker-end="url(#arrow)"/>')
    parts.append('<path d="M 1200 545 V 590 H 1000 V 618" fill="none" stroke="#64748b" '
                 'stroke-width="3" marker-end="url(#arrow)"/>')
    parts.append('<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" '
                 'refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#64748b"/>'
                 '</marker></defs>')
    box(300, 625, 1000, 92, "Target", "parallel reductions · transpose-aware GEMM · device KV/GQA · pooled memory · fused ops",
        "#dbeafe", "#2563eb")
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_svg() -> str:
    with BF16_RESULTS.open(encoding="utf-8", newline="") as stream:
        data = list(csv.DictReader(stream, delimiter="\t"))
    width = 1500
    height = max(700, 220 + len(data) * 96)
    left, top, chart_w = 360, 125, 980
    chart_bottom = top + len(data) * 96 - 40
    axis_y = chart_bottom + 30
    minimum, maximum = 0.75, 1.20
    px = lambda value: left + chart_w * (value - minimum) / (maximum - minimum)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "MI300X BF16 Mixed GEMM · M=1", 30,
             anchor="middle", weight=700),
        text(width / 2, 80, "Includes FP32→BF16 activation cast · FP32 output", 16,
             "#5b6474", anchor="middle"),
    ]
    for tick in (0.8, 0.9, 1.0, 1.1, 1.2):
        x = px(tick)
        parts.append(f'<line x1="{x:.1f}" y1="105" x2="{x:.1f}" y2="{chart_bottom}" '
                     f'stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}" '
                     f'stroke-width="{2 if tick == 1.0 else 1}"/>' )
        parts.append(text(x, axis_y, f"{tick:.1f}×", 14, "#5b6474", anchor="middle"))
    for index, row in enumerate(data):
        y = top + index * 96
        speedup = float(row["speedup"])
        x0, x1 = px(1.0), px(speedup)
        color = "#18a558" if speedup >= 1.0 else "#dc6b5a"
        label = f'{row["case"]}  1×{row["k"]}×{row["n"]}'
        parts.append(text(left - 24, y + 24, label, 16, "#172033", anchor="end", weight=600))
        parts.append(f'<rect x="{min(x0,x1):.1f}" y="{y}" '
                     f'width="{max(abs(x1-x0),2):.1f}" height="34" rx="5" fill="{color}"/>')
        parts.append(text(x1 + (10 if speedup >= 1.0 else -10), y + 24,
                          f"{speedup:.3f}×", 15, color,
                          anchor="start" if speedup >= 1.0 else "end", weight=700))
    parts.append(text(width / 2, height - 15,
                      "Generated from docs/optimization-log/bf16-results.tsv · higher is better",
                      14, "#6b7280", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_policy_svg() -> str:
    with BF16_POLICY_RESULTS.open(encoding="utf-8", newline="") as stream:
        data = list(csv.DictReader(stream, delimiter="\t"))
    width, height = 1500, 560
    left, top, chart_w = 360, 150, 980
    minimum, maximum = 0.75, 1.05
    px = lambda value: left + chart_w * (value - minimum) / (maximum - minimum)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 015 · BF16 Shape Policy", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "3-process median decode throughput relative to retained FP32 path",
             16, "#5b6474", anchor="middle"),
    ]
    for tick in (0.8, 0.9, 1.0):
        x = px(tick)
        parts.append(f'<line x1="{x:.1f}" y1="120" x2="{x:.1f}" y2="390" '
                     f'stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}" '
                     f'stroke-width="{2 if tick == 1.0 else 1}"/>')
        parts.append(text(x, 420, f"{tick:.1f}×", 14, "#5b6474", anchor="middle"))
    for index, row in enumerate(data):
        y = top + index * 120
        ratio = float(row["throughput_ratio"])
        x0, x1 = px(1.0), px(ratio)
        parts.append(text(left - 24, y + 24, row["model"], 16, "#172033",
                          anchor="end", weight=600))
        parts.append(f'<rect x="{min(x0,x1):.1f}" y="{y}" '
                     f'width="{max(abs(x1-x0),2):.1f}" height="34" rx="5" '
                     'fill="#dc6b5a"/>')
        parts.append(text(x1 - 10, y + 24, f"{ratio:.3f}×", 15, "#b83f32",
                          anchor="end", weight=700))
        gib = int(row["extra_engine_bytes"]) / (1024 ** 3)
        parts.append(text(left - 24, y + 52, f"extra engine memory +{gib:.2f} GiB",
                          14, "#6b7280", anchor="end"))
    parts.append(text(width / 2, 485,
                      "Both models generated the same token IDs; speed and memory gates failed",
                      16, "#b83f32", anchor="middle", weight=600))
    parts.append(text(width / 2, 535,
                      "Generated from docs/optimization-log/bf16-model-policy.tsv · higher is better",
                      14, "#6b7280", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_ffn_svg() -> str:
    summary = json.loads(BF16_FFN_SUMMARY.read_text(encoding="utf-8"))
    data = summary["rows"]
    width, height = 1600, 760
    left, top, chart_w = 370, 145, 980
    minimum, maximum = 1.0, 1.65

    def px(value: float) -> float:
        return left + chart_w * (value - minimum) / (maximum - minimum)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 030 · Continuous BF16 FFN Island", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "MI300X · median of 3 process medians · device Event time · higher is better",
             16, "#5b6474", anchor="middle"),
    ]
    for tick in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6):
        x = px(tick)
        parts.append(f'<line x1="{x:.1f}" y1="115" x2="{x:.1f}" y2="620" '
                     f'stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}" '
                     f'stroke-width="{2 if tick == 1.0 else 1}"/>')
        parts.append(text(x, 650, f"{tick:.1f}×", 14, "#5b6474", anchor="middle"))
    for index, row in enumerate(data):
        y = top + index * 118
        label = f'{row["model"].capitalize()}  M={row["tokens"]}'
        parts.append(text(left - 24, y + 26, label, 17, "#172033",
                          anchor="end", weight=700))
        for offset, (key, title, color) in enumerate((
            ("island_speedup_vs_fp32", "vs FP32", "#18a558"),
            ("island_speedup_vs_per_linear", "vs per-Linear BF16", "#2563eb"),
        )):
            ratio = float(row[key])
            bar_y = y + offset * 40
            x0, x1 = px(1.0), px(ratio)
            parts.append(f'<rect x="{x0:.1f}" y="{bar_y}" '
                         f'width="{max(x1-x0,2):.1f}" height="28" rx="5" fill="{color}"/>')
            parts.append(text(x1 + 10, bar_y + 21, f"{ratio:.3f}×  {title}", 14,
                              color, weight=700))
        error = row["paths"]["island"]["relative_l2_error_vs_fp32"] * 100.0
        parts.append(text(left - 24, y + 66, f"relative L2 {error:.2f}%", 13,
                          "#6b7280", anchor="end"))
    parts.append(text(width / 2, 705,
                      "The FP32 running-best curve is unchanged; this is a separate BF16 operator track",
                      16, "#9a4f00", anchor="middle", weight=600))
    parts.append(text(width / 2, 738,
                      "Generated from experiments/030-data/summary.json",
                      14, "#6b7280", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_model_inference_svg() -> str:
    summary = json.loads(BF16_MODEL_SUMMARY.read_text(encoding="utf-8"))
    data = summary["rows"]
    width, height = 1600, 760
    left, top, chart_w = 420, 150, 930
    minimum, maximum = 0.45, 1.25

    def px(value: float) -> float:
        return left + chart_w * (value - minimum) / (maximum - minimum)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 031 · Official-model BF16 FFN Inference", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "median of 3 processes · exact greedy tokens · mixed policy vs two references",
             16, "#5b6474", anchor="middle"),
    ]
    for tick in (0.5, 0.75, 1.0, 1.25):
        x = px(tick)
        parts.append(f'<line x1="{x:.1f}" y1="118" x2="{x:.1f}" y2="600" '
                     f'stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}" '
                     f'stroke-width="{2 if tick == 1.0 else 1}"/>')
        parts.append(text(x, 630, f"{tick:.2f}×", 14, "#5b6474", anchor="middle"))
    metrics = (
        ("decode_speedup", "decode vs microLLM FP32", "#18a558"),
        ("prefill_speedup", "prefill vs microLLM FP32", "#4ec27e"),
        ("microllm_bf16_ffn_decode_ratio_vs_pytorch_bf16",
         "decode vs PyTorch BF16", "#2563eb"),
        ("microllm_bf16_ffn_prefill_ratio_vs_pytorch_bf16",
         "prefill vs PyTorch BF16", "#7c3aed"),
    )
    for row_index, row in enumerate(data):
        y = top + row_index * 220
        label = "Qwen2.5-0.5B" if row["model"].startswith("qwen") else "DeepSeek Distill 1.5B"
        parts.append(text(left - 26, y + 26, label, 18, "#172033",
                          anchor="end", weight=700))
        for offset, (key, title, base_color) in enumerate(metrics):
            ratio = float(row[key])
            bar_y = y + offset * 40
            x0, x1 = px(minimum), px(ratio)
            color = base_color if ratio >= 1.0 or "PyTorch" not in title else "#dc6b5a"
            parts.append(f'<rect x="{x0:.1f}" y="{bar_y}" width="{max(x1-x0,2):.1f}" '
                         f'height="27" rx="5" fill="{color}" opacity="0.9"/>')
            parts.append(text(x1 + 9, bar_y + 20, f"{ratio:.3f}×  {title}", 14,
                              color, weight=700))
        saved = (1.0 - float(row["current_memory_ratio"])) * 100.0
        parts.append(text(left - 26, y + 66, f"engine current −{saved:.1f}%", 14,
                          "#16834a", anchor="end", weight=600))
    parts.append(text(width / 2, 685,
                      "Green: improves retained microLLM FP32 · red: selected PyTorch BF16 gate still fails",
                      16, "#5b6474", anchor="middle", weight=600))
    parts.append(text(width / 2, 728,
                      "microLLM uses BF16 only for FFN weights/activations; PyTorch reference is full BF16",
                      14, "#9a4f00", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_prefill_allocator_svg() -> str:
    before = json.loads(BF16_MODEL_SUMMARY.read_text(encoding="utf-8"))["rows"]
    after = json.loads(BF16_PREFILL_SUMMARY.read_text(encoding="utf-8"))["rows"]
    before_by_model = {row["model"]: row for row in before}
    pytorch = {row["model"]: row for row in before}
    width, height = 1600, 720
    left, top, chart_w = 420, 150, 930
    minimum, maximum = 0.45, 1.30

    def px(value: float) -> float:
        return left + chart_w * (value - minimum) / (maximum - minimum)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 032 · Enable the Allocator for Prefill", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "microLLM BF16-FFN throughput relative to the fixed PyTorch full-BF16 reference",
             16, "#5b6474", anchor="middle"),
    ]
    for tick in (0.5, 0.75, 1.0, 1.25):
        x = px(tick)
        parts.append(f'<line x1="{x:.1f}" y1="118" x2="{x:.1f}" y2="575" '
                     f'stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}" '
                     f'stroke-width="{2 if tick == 1.0 else 1}"/>')
        parts.append(text(x, 605, f"{tick:.2f}×", 14, "#5b6474", anchor="middle"))
    for row_index, candidate in enumerate(after):
        model = candidate["model"]
        baseline = before_by_model[model]
        reference = pytorch[model]
        values = (
            (baseline["microllm_bf16_ffn_decode_ratio_vs_pytorch_bf16"],
             "Exp031 decode", "#9ca3af"),
            (candidate["bf16_ffn_decode_tokens_per_second"] /
             reference["pytorch_bf16_decode_tokens_per_second"],
             "Exp032 decode", "#18a558"),
            (baseline["microllm_bf16_ffn_prefill_ratio_vs_pytorch_bf16"],
             "Exp031 prefill", "#c8ced8"),
            (candidate["bf16_ffn_prefill_tokens_per_second"] /
             reference["pytorch_bf16_prefill_tokens_per_second"],
             "Exp032 prefill", "#2563eb"),
        )
        y = top + row_index * 205
        label = "Qwen2.5-0.5B" if model.startswith("qwen") else "DeepSeek Distill 1.5B"
        parts.append(text(left - 26, y + 26, label, 18, "#172033",
                          anchor="end", weight=700))
        for offset, (ratio, title, color) in enumerate(values):
            bar_y = y + offset * 38
            if ratio < 1.0 and "Exp032" in title:
                color = "#dc6b5a"
            x0, x1 = px(minimum), px(ratio)
            parts.append(f'<rect x="{x0:.1f}" y="{bar_y}" width="{max(x1-x0,2):.1f}" '
                         f'height="26" rx="5" fill="{color}"/>')
            parts.append(text(x1 + 9, bar_y + 19, f"{ratio:.3f}×  {title}", 14,
                              color, weight=700))
    parts.append(text(width / 2, 660,
                      "Three of four selected PyTorch BF16 rows now pass; DeepSeek decode remains red",
                      16, "#9a4f00", anchor="middle", weight=600))
    parts.append(text(width / 2, 697,
                      "Generated from Experiment 031 PyTorch reference + Experiment 032 microLLM raw medians",
                      14, "#6b7280", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_attention_svg() -> str:
    baseline = {row["model"]: row for row in
                json.loads(BF16_PREFILL_SUMMARY.read_text(encoding="utf-8"))["rows"]}
    candidate = json.loads(BF16_ATTENTION_SUMMARY.read_text(encoding="utf-8"))["rows"]
    pilot = {row["model"]: row for row in
             (json.loads(line) for line in BF16_ATTENTION_PILOT.read_text(
                 encoding="utf-8").splitlines())}
    width, height = 1600, 700
    left, top, chart_w = 420, 150, 930
    minimum, maximum = 0.94, 1.08

    def px(value: float) -> float:
        return left + chart_w * (value - minimum) / (maximum - minimum)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 034 · BF16 Attention Shared Cast", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "throughput relative to retained BF16-FFN model · median candidate vs one-run pilot",
             16, "#5b6474", anchor="middle"),
    ]
    for tick in (0.95, 1.0, 1.05):
        x = px(tick)
        parts.append(f'<line x1="{x:.1f}" y1="118" x2="{x:.1f}" y2="555" '
                     f'stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}" '
                     f'stroke-width="{2 if tick == 1.0 else 1}"/>')
        parts.append(text(x, 585, f"{tick:.2f}×", 14, "#5b6474", anchor="middle"))
    for row_index, row in enumerate(candidate):
        model = row["model"]
        old = baseline[model]
        first = pilot[model]
        values = (
            (first["decode_tokens_per_second"] / old["bf16_ffn_decode_tokens_per_second"],
             "per-Linear cast decode", "#c8ced8"),
            (first["prefill_tokens_per_second"] / old["bf16_ffn_prefill_tokens_per_second"],
             "per-Linear cast prefill", "#c8ced8"),
            (row["decode_speedup_vs_bf16_ffn"], "shared cast decode", "#18a558"),
            (row["prefill_speedup_vs_bf16_ffn"], "shared cast prefill", "#2563eb"),
        )
        y = top + row_index * 195
        label = "Qwen2.5-0.5B" if model.startswith("qwen") else "DeepSeek Distill 1.5B"
        parts.append(text(left - 26, y + 26, label, 18, "#172033",
                          anchor="end", weight=700))
        for offset, (ratio, title, color) in enumerate(values):
            bar_y = y + offset * 38
            if ratio < 1.0 and "shared" in title:
                color = "#f59e0b"
            x0, x1 = px(1.0), px(ratio)
            parts.append(f'<rect x="{min(x0,x1):.1f}" y="{bar_y}" '
                         f'width="{max(abs(x1-x0),2):.1f}" height="26" rx="5" fill="{color}"/>')
            parts.append(text(x1 + (9 if ratio >= 1.0 else -9), bar_y + 19,
                              f"{ratio:.3f}×  {title}", 14, color,
                              anchor="start" if ratio >= 1.0 else "end", weight=700))
    parts.append(text(width / 2, 635,
                      "Shared input cast turns both decode rows positive; DeepSeek prefill stays within −5% gate",
                      16, "#5b6474", anchor="middle", weight=600))
    parts.append(text(width / 2, 678,
                      "DeepSeek decode remains 0.533× PyTorch BF16 — output head / broader islands remain",
                      14, "#9a4f00", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_plan_cache_svg() -> str:
    data = json.loads(BF16_PLAN_SUMMARY.read_text(encoding="utf-8"))["rows"]
    width, height = 1600, 700
    left, top, chart_w = 420, 150, 930
    minimum, maximum = 1.0, 3.8

    def px(value: float) -> float:
        return left + chart_w * (value - minimum) / (maximum - minimum)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 036 · Immutable BF16 hipBLASLt Plans", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "median of 3 processes · exact tokens · no algorithm or memory-policy change",
             16, "#5b6474", anchor="middle"),
    ]
    for tick in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
        x = px(tick)
        parts.append(f'<line x1="{x:.1f}" y1="118" x2="{x:.1f}" y2="555" '
                     f'stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}" '
                     f'stroke-width="{2 if tick == 1.0 else 1}"/>')
        parts.append(text(x, 585, f"{tick:.1f}×", 14, "#5b6474", anchor="middle"))
    metrics = (
        ("decode_speedup_vs_bf16_attention", "decode vs Exp034", "#18a558"),
        ("prefill_speedup_vs_bf16_attention", "prefill vs Exp034", "#4ec27e"),
        ("decode_ratio_vs_pytorch_bf16", "decode vs PyTorch BF16", "#2563eb"),
        ("prefill_ratio_vs_pytorch_bf16", "prefill vs PyTorch BF16", "#7c3aed"),
    )
    for row_index, row in enumerate(data):
        y = top + row_index * 195
        label = "Qwen2.5-0.5B" if row["model"].startswith("qwen") else "DeepSeek Distill 1.5B"
        parts.append(text(left - 26, y + 26, label, 18, "#172033",
                          anchor="end", weight=700))
        for offset, (key, title, color) in enumerate(metrics):
            ratio = float(row[key])
            bar_y = y + offset * 38
            x0, x1 = px(1.0), px(ratio)
            parts.append(f'<rect x="{x0:.1f}" y="{bar_y}" width="{max(x1-x0,2):.1f}" '
                         f'height="26" rx="5" fill="{color}"/>')
            parts.append(text(x1 + 9, bar_y + 19, f"{ratio:.3f}×  {title}", 14,
                              color, weight=700))
    parts.append(text(width / 2, 635,
                      "All four selected PyTorch BF16 throughput gates pass",
                      17, "#16834a", anchor="middle", weight=700))
    parts.append(text(width / 2, 678,
                      "Scope: pinned short-prompt MI300X inference — not training or universal-model parity",
                      14, "#9a4f00", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_training_svg() -> str:
    data = json.loads(BF16_TRAINING_SUMMARY.read_text(encoding="utf-8"))["rows"]
    width, height = 1600, 680
    left, top, chart_w = 430, 160, 900
    minimum, maximum = 0.75, 3.30

    def px(value: float) -> float:
        return left + chart_w * (value - minimum) / (maximum - minimum)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 037 · BF16 Linear Training with FP32 Masters", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "2 warm-up + 5 measured steps · median of 3 processes · lower loss in every run",
             16, "#5b6474", anchor="middle"),
    ]
    for tick in (1.0, 1.5, 2.0, 2.5, 3.0):
        x = px(tick)
        parts.append(f'<line x1="{x:.1f}" y1="125" x2="{x:.1f}" y2="520" '
                     f'stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}" '
                     f'stroke-width="{2 if tick == 1.0 else 1}"/>')
        parts.append(text(x, 550, f"{tick:.1f}×", 14, "#5b6474", anchor="middle"))
    for index, row in enumerate(data):
        y = top + index * 180
        label = "Qwen2.5-0.5B" if row["model"].startswith("qwen") else "DeepSeek Distill 1.5B"
        parts.append(text(left - 26, y + 26, label, 18, "#172033",
                          anchor="end", weight=700))
        values = (
            (row["bf16_speedup_vs_microllm_fp32"], "vs microLLM FP32", "#f59e0b"),
            (row["microllm_bf16_ratio_vs_pytorch_bf16_amp"],
             "vs PyTorch BF16 autocast", "#18a558"),
            (row["bf16_peak_ratio_vs_microllm_fp32"], "peak memory vs microLLM FP32", "#64748b"),
        )
        for offset, (ratio, title, color) in enumerate(values):
            bar_y = y + offset * 42
            x0, x1 = px(1.0), px(ratio)
            parts.append(f'<rect x="{min(x0,x1):.1f}" y="{bar_y}" '
                         f'width="{max(abs(x1-x0),2):.1f}" height="28" rx="5" fill="{color}"/>')
            parts.append(text(x1 + (9 if ratio >= 1.0 else -9), bar_y + 21,
                              f"{ratio:.3f}×  {title}", 14, color,
                              anchor="start" if ratio >= 1.0 else "end", weight=700))
        parts.append(text(left - 26, y + 68,
                          f'loss {row["microllm_bf16_first_loss"]:.3f}→'
                          f'{row["microllm_bf16_final_loss"]:.3f}',
                          14, "#5b6474", anchor="end"))
    parts.append(text(width / 2, 610,
                      "Correct and faster than PyTorch reference, but slower than microLLM FP32 and no memory saving",
                      16, "#9a4f00", anchor="middle", weight=600))
    parts.append(text(width / 2, 654,
                      "Next: continuous BF16 training islands / forward-weight lifecycle — not a larger accuracy claim",
                      14, "#6b7280", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_training_qkv_svg() -> str:
    data = json.loads(BF16_TRAINING_QKV_SUMMARY.read_text(encoding="utf-8"))["rows"]
    width, height = 1500, 620
    left, top, chart_w = 430, 150, 850
    minimum, maximum = 0.85, 1.05

    def px(value: float) -> float:
        return left + chart_w * (value - minimum) / (maximum - minimum)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 039 · Training Shared-QKV Cast", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "candidate throughput and allocation calls relative to retained BF16 training",
             16, "#5b6474", anchor="middle"),
    ]
    for tick in (0.85, 0.90, 0.95, 1.0, 1.05):
        x = px(tick)
        parts.append(f'<line x1="{x:.1f}" y1="118" x2="{x:.1f}" y2="470" '
                     f'stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}" '
                     f'stroke-width="{2 if tick == 1.0 else 1}"/>')
        parts.append(text(x, 500, f"{tick:.2f}×", 14, "#5b6474", anchor="middle"))
    allocation_ratios = {"qwen2.5-0.5b": 10760 / 11000,
                         "deepseek-r1-distill-qwen-1.5b": 12545 / 12825}
    for index, row in enumerate(data):
        y = top + index * 155
        label = "Qwen2.5-0.5B" if row["model"].startswith("qwen") else "DeepSeek Distill 1.5B"
        parts.append(text(left - 26, y + 26, label, 18, "#172033",
                          anchor="end", weight=700))
        values = (
            (row["speedup_vs_bf16_independent_linears"], "throughput vs BF16 baseline", "#f59e0b"),
            (allocation_ratios[row["model"]], "allocation calls vs baseline", "#64748b"),
            (row["ratio_vs_microllm_fp32"], "throughput vs microLLM FP32", "#dc6b5a"),
        )
        for offset, (ratio, title, color) in enumerate(values):
            bar_y = y + offset * 39
            x0, x1 = px(1.0), px(ratio)
            parts.append(f'<rect x="{min(x0,x1):.1f}" y="{bar_y}" '
                         f'width="{max(abs(x1-x0),2):.1f}" height="27" rx="5" fill="{color}"/>')
            parts.append(text(x1 - 9, bar_y + 20, f"{ratio:.3f}×  {title}", 14,
                              color, anchor="end", weight=700))
    parts.append(text(width / 2, 555,
                      "Allocation hypothesis passed; two-model throughput objective regressed to ~0.991×",
                      16, "#b83f32", anchor="middle", weight=700))
    parts.append(text(width / 2, 598,
                      "Candidate graph API removed · raw results retained",
                      14, "#6b7280", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_training_mirror_svg() -> str:
    data = json.loads(BF16_TRAINING_MIRROR_SUMMARY.read_text(encoding="utf-8"))["rows"]
    width, height = 1500, 650
    left, top, chart_w = 430, 155, 850
    minimum, maximum = 0.90, 1.13

    def px(value: float) -> float:
        return left + chart_w * (value - minimum) / (maximum - minimum)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 040 · Persistent BF16 Training Weight Mirrors", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "median of 3 processes · throughput and peak engine memory vs Experiment 037",
             16, "#5b6474", anchor="middle"),
    ]
    for tick in (0.90, 0.95, 1.00, 1.05, 1.10):
        x = px(tick)
        parts.append(f'<line x1="{x:.1f}" y1="120" x2="{x:.1f}" y2="490" '
                     f'stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}" '
                     f'stroke-width="{2 if tick == 1.0 else 1}"/>')
        parts.append(text(x, 520, f"{tick:.2f}×", 14, "#5b6474", anchor="middle"))
    for index, row in enumerate(data):
        y = top + index * 160
        label = "Qwen2.5-0.5B" if row["model"].startswith("qwen") else \
            "DeepSeek Distill 1.5B"
        parts.append(text(left - 26, y + 28, label, 18, "#172033",
                          anchor="end", weight=700))
        values = (
            (row["speedup_vs_bf16_independent_linears"],
             "throughput vs old BF16", "#18a558"),
            (row["peak_ratio_vs_baseline"], "peak memory vs old BF16", "#dc6b5a"),
            (row["ratio_vs_microllm_fp32"], "throughput vs microLLM FP32", "#f59e0b"),
        )
        for offset, (ratio, title, color) in enumerate(values):
            bar_y = y + offset * 41
            x0, x1 = px(1.0), px(ratio)
            parts.append(f'<rect x="{min(x0,x1):.1f}" y="{bar_y}" '
                         f'width="{max(abs(x1-x0),2):.1f}" height="28" rx="5" '
                         f'fill="{color}"/>')
            anchor = "start" if ratio >= 1.0 else "end"
            dx = 9 if ratio >= 1.0 else -9
            parts.append(text(x1 + dx, bar_y + 21, f"{ratio:.3f}×  {title}", 14,
                              color, anchor=anchor, weight=700))
        parts.append(text(left - 26, y + 75,
                          f'{row["ratio_vs_pytorch_bf16_amp"]:.3f}× PyTorch BF16 autocast',
                          14, "#5b6474", anchor="end"))
    parts.append(text(width / 2, 575,
                      "Both models pass the >5% speed gate; peak memory rises 7.9% / 10.8%",
                      16, "#9a4f00", anchor="middle", weight=700))
    parts.append(text(width / 2, 620,
                      "Kept as an explicit speed/memory trade-off · mirrors are derived checkpoint state",
                      14, "#6b7280", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_training_island_svg() -> str:
    row = json.loads(BF16_TRAINING_ISLAND_SUMMARY.read_text(encoding="utf-8"))["qwen"]
    width, height = 1500, 620
    left, top, chart_w = 430, 150, 850
    minimum, maximum = 0.98, 1.02

    def px(value: float) -> float:
        return left + chart_w * (value - minimum) / (maximum - minimum)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 041 · BF16 FFN Training Island", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "same-window Qwen control · old absolute baseline invalidated by shared-GPU drift",
             16, "#5b6474", anchor="middle"),
    ]
    for tick in (0.98, 0.99, 1.00, 1.01, 1.02):
        x = px(tick)
        parts.append(f'<line x1="{x:.1f}" y1="118" x2="{x:.1f}" y2="430" '
                     f'stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}" '
                     f'stroke-width="{2 if tick == 1.0 else 1}"/>')
        parts.append(text(x, 460, f"{tick:.2f}×", 14, "#5b6474", anchor="middle"))
    values = (
        (row["speedup_vs_same_window_control"], "throughput", "#f59e0b"),
        (row["allocation_ratio"], "allocation calls", "#64748b"),
        (row["peak_ratio"], "peak engine memory", "#64748b"),
    )
    for index, (ratio, label, color) in enumerate(values):
        y = top + index * 85
        parts.append(text(left - 24, y + 22, label, 18, "#172033",
                          anchor="end", weight=700))
        x0, x1 = px(1.0), px(ratio)
        parts.append(f'<rect x="{min(x0,x1):.1f}" y="{y}" '
                     f'width="{max(abs(x1-x0),2):.1f}" height="30" rx="5" '
                     f'fill="{color}"/>')
        parts.append(text(x1 + (10 if ratio >= 1.0 else -10), y + 23,
                          f"{ratio:.3f}×", 15, color,
                          anchor="start" if ratio >= 1.0 else "end", weight=700))
    parts.append(text(width / 2, 520,
                      "1.1% is below the 5% keep gate · candidate removed",
                      17, "#b83f32", anchor="middle", weight=700))
    parts.append(text(width / 2, 565,
                      "DeepSeek early-stopped after >3 min because Qwen had already failed the gate",
                      14, "#6b7280", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_training_shape_svg() -> str:
    rows = json.loads(BF16_TRAINING_SHAPE_SUMMARY.read_text(encoding="utf-8"))["rows"]
    width, height = 1600, 720
    chart_left, chart_top, chart_width, chart_height = 145, 130, 1320, 430
    y_max = 1.10

    def py(value: float) -> float:
        return chart_top + chart_height * (y_max - value) / y_max

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 042 · Qwen BF16 Training Shape Matrix", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "1 warm-up + 2 measured updates · median of 3 fresh processes per framework",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_left}" y="{chart_top}" width="{chart_width}" '
        f'height="{chart_height}" fill="#ffffff" stroke="#cbd3df" rx="8"/>',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = py(tick)
        parts.append(f'<line x1="{chart_left}" y1="{y:.1f}" '
                     f'x2="{chart_left + chart_width}" y2="{y:.1f}" '
                     f'stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}" '
                     f'stroke-width="{2 if tick == 1.0 else 1}"/>')
        parts.append(text(chart_left - 14, y + 6, f"{tick:.2f}×", 14,
                          "#5b6474", anchor="end"))
    group_width = chart_width / len(rows)
    for index, row in enumerate(rows):
        center = chart_left + group_width * (index + 0.5)
        values = ((row["throughput_ratio_microllm_over_pytorch"], "#dc6b5a"),
                  (row["peak_memory_ratio"], "#64748b"))
        for offset, (value, color) in enumerate(values):
            x = center - 70 + offset * 78
            y = py(value)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="62" '
                         f'height="{py(0)-y:.1f}" rx="5" fill="{color}"/>')
            parts.append(text(x + 31, y - 9, f"{value:.3f}×", 14, color,
                              anchor="middle", weight=700))
        parts.append(text(center, chart_top + chart_height + 35,
                          f'{row["batch"]}×{row["context"]}', 17,
                          "#172033", anchor="middle", weight=700))
    parts.append(text(width / 2, 620,
                      "red: throughput vs PyTorch BF16 autocast · gray: peak engine/allocated memory",
                      16, "#5b6474", anchor="middle"))
    parts.append(text(width / 2, 670,
                      "Context 32 is the worst throughput shape; context 128 crosses PyTorch peak memory",
                      16, "#b83f32", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_weight_gradient_svg() -> str:
    rows = json.loads(BF16_WEIGHT_GRADIENT_COMPARISON.read_text(encoding="utf-8"))["rows"]
    width, height = 1700, 760
    panel_top, panel_height = 145, 440
    left_x, right_x, panel_width = 105, 925, 690
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 043 · Wide Weight-Gradient GEMM Routing", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "Qwen BF16 training · median of 3 fresh processes · peak memory unchanged",
             16, "#5b6474", anchor="middle"),
        text(left_x + panel_width / 2, 118, "Speedup vs Experiment 042", 20,
             anchor="middle", weight=700),
        text(right_x + panel_width / 2, 118, "microLLM / PyTorch throughput", 20,
             anchor="middle", weight=700),
    ]
    for x in (left_x, right_x):
        parts.append(f'<rect x="{x}" y="{panel_top}" width="{panel_width}" '
                     f'height="{panel_height}" fill="#ffffff" stroke="#cbd3df" rx="8"/>')

    def left_y(value: float) -> float:
        return panel_top + panel_height * (5.0 - value) / 5.0

    def right_y(value: float) -> float:
        return panel_top + panel_height * (1.0 - value)

    for tick in range(6):
        y = left_y(float(tick))
        parts.append(f'<line x1="{left_x}" y1="{y:.1f}" x2="{left_x+panel_width}" '
                     f'y2="{y:.1f}" stroke="#e5e9f0"/>')
        parts.append(text(left_x - 10, y + 5, f"{tick}×", 13, "#5b6474", anchor="end"))
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = right_y(tick)
        parts.append(f'<line x1="{right_x}" y1="{y:.1f}" '
                     f'x2="{right_x+panel_width}" y2="{y:.1f}" '
                     f'stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}"/>')
        parts.append(text(right_x - 10, y + 5, f"{tick:.2f}×", 13,
                          "#5b6474", anchor="end"))
    group = panel_width / len(rows)
    for index, row in enumerate(rows):
        label = f'{row["batch"]}×{row["context"]}'
        center_left = left_x + group * (index + 0.5)
        speedup = row["self_speedup"]
        y = left_y(speedup)
        parts.append(f'<rect x="{center_left-31:.1f}" y="{y:.1f}" width="62" '
                     f'height="{left_y(0)-y:.1f}" rx="5" fill="#18a558"/>')
        parts.append(text(center_left, y - 9, f"{speedup:.3f}×", 14, "#16834a",
                          anchor="middle", weight=700))
        parts.append(text(center_left, panel_top + panel_height + 32, label, 16,
                          "#172033", anchor="middle", weight=700))

        center_right = right_x + group * (index + 0.5)
        for offset, (value, color) in enumerate((
                (row["before_ratio_vs_pytorch"], "#c8ced8"),
                (row["after_ratio_vs_pytorch"], "#18a558"))):
            x = center_right - 49 + offset * 54
            ry = right_y(value)
            parts.append(f'<rect x="{x:.1f}" y="{ry:.1f}" width="44" '
                         f'height="{right_y(0)-ry:.1f}" rx="4" fill="{color}"/>')
        parts.append(text(center_right, panel_top + panel_height + 32, label, 16,
                          "#172033", anchor="middle", weight=700))
        parts.append(text(center_right, right_y(row["after_ratio_vs_pytorch"]) - 9,
                          f'{row["after_ratio_vs_pytorch"]:.3f}×', 13, "#16834a",
                          anchor="middle", weight=700))
    parts.append(text(width / 2, 665,
                      "context 32: readable transpose hotspot removed · 4.476× self speedup",
                      17, "#16834a", anchor="middle", weight=700))
    parts.append(text(width / 2, 712,
                      "gray: before · green: retained route · no shape yet reaches PyTorch parity",
                      14, "#6b7280", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def fused_causal_gqa_svg() -> str:
    rows = json.loads(FUSED_CAUSAL_GQA_COMPARISON.read_text(encoding="utf-8"))["rows"]
    width, height = 1600, 710
    left, top, chart_w, chart_h = 150, 145, 1300, 400

    def py(value: float) -> float:
        return top + chart_h * (1.25 - value) / 1.25

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 044 · Fused Full-Sequence Causal GQA", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "median of 3 processes · direct GQA · recomputed backward probabilities",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{left}" y="{top}" width="{chart_w}" height="{chart_h}" '
        f'fill="#ffffff" stroke="#cbd3df" rx="8"/>',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
        y = py(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+chart_w}" '
                     f'y2="{y:.1f}" stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}" '
                     f'stroke-width="{2 if tick == 1.0 else 1}"/>')
        parts.append(text(left - 12, y + 5, f"{tick:.2f}×", 13,
                          "#5b6474", anchor="end"))
    group = chart_w / len(rows)
    for index, row in enumerate(rows):
        center = left + group * (index + 0.5)
        values = ((row["self_speedup"], "#18a558"),
                  (row["ratio_vs_pytorch"], "#dc6b5a"),
                  (row["peak_ratio_after_vs_before"], "#64748b"))
        for offset, (value, color) in enumerate(values):
            x = center - 75 + offset * 52
            y = py(value)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="43" '
                         f'height="{py(0)-y:.1f}" rx="4" fill="{color}"/>')
            if offset == 0:
                parts.append(text(x + 21.5, y - 8, f"{value:.3f}×", 13,
                                  color, anchor="middle", weight=700))
        parts.append(text(center, top + chart_h + 34,
                          f'{row["batch"]}×{row["context"]}', 16,
                          "#172033", anchor="middle", weight=700))
        parts.append(text(center, top + chart_h + 57,
                          f'-{row["peak_bytes_saved"]/1048576:.1f} MiB', 13,
                          "#64748b", anchor="middle"))
    parts.append(text(width / 2, 635,
                      "green: self speedup · red: microLLM/PyTorch · gray: peak ratio",
                      15, "#5b6474", anchor="middle"))
    parts.append(text(width / 2, 680,
                      "All selected shapes improve and allocate less; parity remains open",
                      16, "#16834a", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def deepseek_shape_svg() -> str:
    rows = json.loads(DEEPSEEK_SHAPE_SUMMARY.read_text(encoding="utf-8"))["rows"]
    loads = json.loads(DEEPSEEK_LOAD_SUMMARY.read_text(encoding="utf-8"))
    width, height = 1700, 760
    left_x, right_x, top, panel_w, panel_h = 110, 930, 150, 680, 430
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 045 · DeepSeek Training Shapes and Loading", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "DeepSeek-R1-Distill-Qwen-1.5B · median of 3 fresh processes",
             16, "#5b6474", anchor="middle"),
        text(left_x + panel_w / 2, 120, "Training ratios", 20,
             anchor="middle", weight=700),
        text(right_x + panel_w / 2, 120, "Checkpoint load seconds", 20,
             anchor="middle", weight=700),
    ]
    for x in (left_x, right_x):
        parts.append(f'<rect x="{x}" y="{top}" width="{panel_w}" height="{panel_h}" '
                     f'fill="#ffffff" stroke="#cbd3df" rx="8"/>')

    def ratio_y(value: float) -> float:
        return top + panel_h * (1.0 - value)

    def load_y(value: float) -> float:
        return top + panel_h * (70.0 - value) / 70.0

    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = ratio_y(tick)
        parts.append(f'<line x1="{left_x}" y1="{y:.1f}" x2="{left_x+panel_w}" '
                     f'y2="{y:.1f}" stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}"/>')
        parts.append(text(left_x - 10, y + 5, f"{tick:.2f}×", 13,
                          "#5b6474", anchor="end"))
    for tick in (0, 20, 40, 60):
        y = load_y(float(tick))
        parts.append(f'<line x1="{right_x}" y1="{y:.1f}" x2="{right_x+panel_w}" '
                     f'y2="{y:.1f}" stroke="#e5e9f0"/>')
        parts.append(text(right_x - 10, y + 5, f"{tick}s", 13,
                          "#5b6474", anchor="end"))
    group = panel_w / len(rows)
    micro_loads = loads["after_microllm_load_ms_median_by_shape"]
    torch_loads = loads["pytorch_load_ms_median_by_shape"]
    for index, row in enumerate(rows):
        label = f'{row["batch"]}×{row["context"]}'
        key = f'{row["batch"]}x{row["context"]}'
        center = left_x + group * (index + 0.5)
        for offset, (value, color) in enumerate((
                (row["throughput_ratio_microllm_over_pytorch"], "#dc6b5a"),
                (row["peak_memory_ratio"], "#64748b"))):
            x = center - 47 + offset * 51
            y = ratio_y(value)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="43" '
                         f'height="{ratio_y(0)-y:.1f}" rx="4" fill="{color}"/>')
        parts.append(text(center, top + panel_h + 31, label, 15,
                          "#172033", anchor="middle", weight=700))
        parts.append(text(center, ratio_y(row["throughput_ratio_microllm_over_pytorch"])-8,
                          f'{row["throughput_ratio_microllm_over_pytorch"]:.3f}×', 12,
                          "#b83f32", anchor="middle", weight=700))

        load_center = right_x + group * (index + 0.5)
        for offset, (milliseconds, color) in enumerate((
                (micro_loads[key], "#f59e0b"), (torch_loads[key], "#18a558"))):
            seconds = milliseconds / 1000.0
            x = load_center - 47 + offset * 51
            y = load_y(seconds)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="43" '
                         f'height="{load_y(0)-y:.1f}" rx="4" fill="{color}"/>')
        parts.append(text(load_center, top + panel_h + 31, label, 15,
                          "#172033", anchor="middle", weight=700))
        parts.append(text(load_center - 25, load_y(micro_loads[key]/1000.0)-8,
                          f'{micro_loads[key]/1000.0:.1f}s', 12, "#9a4f00",
                          anchor="middle", weight=700))
    parts.append(text(width / 2, 655,
                      "left red: throughput · left gray: peak memory · right orange/green: microLLM/PyTorch load",
                      14, "#5b6474", anchor="middle"))
    parts.append(text(width / 2, 705,
                      "Training uses less peak memory; loading remains ~30× slower",
                      16, "#9a4f00", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def deepseek_context128_profile_svg() -> str:
    data = json.loads(DEEPSEEK_PROFILE_SUMMARY.read_text(encoding="utf-8"))
    categories = data["categories"]
    width, height = 1800, 900
    chart_x, chart_y, chart_w = 115, 150, 1030
    panel_x, panel_y, panel_w, panel_h = 1225, 150, 470, 620
    row_h, bar_h = 66, 28
    maximum = max(row["kernel_time_percent"] for row in categories)
    colors = ("#d04a3a", "#e49b38", "#7c5ce0", "#3b82c4", "#5ca46d",
              "#8c69c8", "#2f9b8f", "#b36a85", "#aab2bf")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 046 · DeepSeek Context-128 Profile", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "MI300X · checkpoint load + 1 warm-up + 2 measured steps · 1.369 s Kernel time",
             16, "#5b6474", anchor="middle"),
        text(chart_x, 119, "Kernel-time categories", 20, weight=700),
        f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        text(panel_x + panel_w / 2, panel_y + 44, "What the counts prove", 21,
             anchor="middle", weight=700),
    ]
    for index, (row, color) in enumerate(zip(categories, colors)):
        y = chart_y + index * row_h
        label = row["name"]
        if len(label) > 37:
            label = label[:36] + "…"
        parts.append(text(chart_x, y + 18, label, 15, weight=600))
        bar_y = y + 27
        parts.append(f'<rect x="{chart_x}" y="{bar_y}" width="{chart_w}" '
                     f'height="{bar_h}" rx="5" fill="#eef1f5"/>')
        value_w = chart_w * row["kernel_time_percent"] / maximum
        parts.append(f'<rect x="{chart_x}" y="{bar_y}" width="{value_w:.1f}" '
                     f'height="{bar_h}" rx="5" fill="{color}"/>')
        suffix = "training" if row["training_only"] else "mixed scope"
        parts.append(text(chart_x + chart_w + 16, bar_y + 20,
                          f'{row["kernel_time_percent"]:.2f}% · {row["calls"]:,} · {suffix}',
                          14, "#445064"))

    notes = (
        ("339 × 3 = 1,017", "parameter tensors × steps", "#d04a3a"),
        ("28 × 3 = 84", "Attention calls per direction", "#7c5ce0"),
        ("23.00% is mixed", "copy includes load transpose", "#e49b38"),
    )
    for index, (headline, detail, color) in enumerate(notes):
        y = panel_y + 105 + index * 115
        parts.append(f'<circle cx="{panel_x + 46}" cy="{y - 8}" r="12" fill="{color}"/>')
        parts.append(text(panel_x + 78, y, headline, 20, color, weight=700))
        parts.append(text(panel_x + 78, y + 28, detail, 15, "#5b6474"))
    divider_y = panel_y + 465
    parts.append(f'<line x1="{panel_x + 35}" y1="{divider_y}" '
                 f'x2="{panel_x + panel_w - 35}" y2="{divider_y}" stroke="#d7dde6"/>')
    parts.append(text(panel_x + panel_w / 2, divider_y + 48, "Next falsifiable path", 18,
                      anchor="middle", weight=700))
    parts.append(text(panel_x + panel_w / 2, divider_y + 87, "stable gradient addresses", 17,
                      "#2563eb", anchor="middle", weight=700))
    parts.append(text(panel_x + panel_w / 2, divider_y + 120, "↓", 24,
                      "#64748b", anchor="middle"))
    parts.append(text(panel_x + panel_w / 2, divider_y + 154, "multi-tensor AdamW", 17,
                      "#2563eb", anchor="middle", weight=700))
    parts.append(text(width / 2, 845,
                      "A large bar is not enough: phase boundaries and count identities decide attribution",
                      16, "#9a4f00", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def stable_gradient_discard_svg() -> str:
    data = json.loads(STABLE_GRADIENT_COMPARISON.read_text(encoding="utf-8"))
    width, height = 1600, 720
    panels = ((155, "Throughput", data["speed_ratio"], "#d04a3a",
               f'{data["throughput_change_percent"]:.2f}%'),
              (850, "Peak memory", data["peak_ratio"], "#2f9b68",
               f'{data["peak_change_percent"]:.2f}%'))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 047 · Stable Gradient Buffer Discarded", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "Qwen2.5-0.5B · BF16 Linear / FP32 master · batch 1 · context 128 · median of 3",
             16, "#5b6474", anchor="middle"),
    ]
    for x, title, ratio, color, change in panels:
        panel_w, panel_h, top = 590, 390, 135
        parts.append(f'<rect x="{x}" y="{top}" width="{panel_w}" height="{panel_h}" '
                     'fill="#ffffff" stroke="#cbd3df" rx="10"/>')
        parts.append(text(x + panel_w / 2, top + 45, title, 22,
                          anchor="middle", weight=700))
        base_y, bar_h = top + 285, 210
        for index, (label, value, fill) in enumerate((
                ("Experiment 044", 1.0, "#64748b"),
                ("stable buffer", ratio, color))):
            bar_x = x + 130 + index * 190
            height_value = bar_h * value / 1.05
            parts.append(f'<rect x="{bar_x}" y="{base_y-height_value:.1f}" width="110" '
                         f'height="{height_value:.1f}" rx="6" fill="{fill}"/>')
            parts.append(text(bar_x + 55, base_y + 30, label, 14,
                              "#445064", anchor="middle"))
            parts.append(text(bar_x + 55, base_y - height_value - 10,
                              f"{value:.3f}×", 17, fill, anchor="middle", weight=700))
        parts.append(text(x + panel_w / 2, top + 360, change, 22, color,
                          anchor="middle", weight=700))
    parts.append(text(width / 2, 585,
                      "Address tests passed, but one first-contribution copy per leaf crossed the −5% speed gate",
                      17, "#b83f32", anchor="middle", weight=700))
    parts.append(text(width / 2, 632,
                      "Next: current pointers in 16-tensor Kernel-argument chunks · no persistent table · no gradient copy",
                      16, "#2563eb", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def chunked_adamw_discard_svg() -> str:
    data = json.loads(CHUNKED_ADAMW_COMPARISON.read_text(encoding="utf-8"))
    rows = data["small_tensor_candidate"]["rows"]
    pilot = data["all_tensor_pilot"]
    width, height = 1700, 760
    chart_x, chart_y, chart_w, chart_h = 100, 150, 1040, 410
    panel_x, panel_y, panel_w, panel_h = 1210, 150, 390, 410
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 048 · Chunked AdamW Discarded", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "Qwen2.5-0.5B · matched 1 warm-up + 2 measured · median of 3 fresh processes",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        text(chart_x + chart_w / 2, 125, "Small-tensor grouping speedup", 20,
             anchor="middle", weight=700),
        text(panel_x + panel_w / 2, 125, "All-tensor pilot", 20,
             anchor="middle", weight=700),
    ]
    def y(value: float) -> float:
        return chart_y + chart_h * (1.10 - value) / 0.20
    for tick in (0.90, 0.95, 1.00, 1.05, 1.10):
        position = y(tick)
        color = "#2563eb" if tick == 1.0 else "#e5e9f0"
        parts.append(f'<line x1="{chart_x}" y1="{position:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{position:.1f}" stroke="{color}"/>')
        parts.append(text(chart_x - 12, position + 5, f"{tick:.2f}×", 13,
                          "#5b6474", anchor="end"))
    group = chart_w / len(rows)
    for index, row in enumerate(rows):
        center = chart_x + group * (index + 0.5)
        ratio = row["speedup"]
        base = y(1.0)
        top = y(ratio)
        color = "#2f9b68" if ratio >= 1.0 else "#d04a3a"
        parts.append(f'<rect x="{center-48:.1f}" y="{min(base, top):.1f}" width="96" '
                     f'height="{max(4.0, abs(base-top)):.1f}" rx="5" fill="{color}"/>')
        parts.append(text(center, chart_y + chart_h + 31,
                          f'{row["batch"]}×{row["context"]}', 16,
                          anchor="middle", weight=700))
        parts.append(text(center, top - 10 if ratio >= 1 else top + 22,
                          f"{ratio:.3f}×", 15, color, anchor="middle", weight=700))
    parts.append(text(panel_x + panel_w / 2, panel_y + 105,
                      f'{pilot["speedup"]:.3f}×', 54, "#d04a3a",
                      anchor="middle", weight=700))
    parts.append(text(panel_x + panel_w / 2, panel_y + 145,
                      f'{pilot["throughput_change_percent"]:.1f}% throughput', 18,
                      "#b83f32", anchor="middle", weight=700))
    parts.append(text(panel_x + panel_w / 2, panel_y + 225, "290 → 19 launches", 24,
                      "#172033", anchor="middle", weight=700))
    parts.append(text(panel_x + panel_w / 2, panel_y + 270,
                      "fewer launches", 16, "#64748b", anchor="middle"))
    parts.append(text(panel_x + panel_w / 2, panel_y + 300,
                      "≠ faster Kernel", 18, "#d04a3a", anchor="middle", weight=700))
    parts.append(text(width / 2, 655,
                      "Small grouping cuts dispatches 39%, but no official shape reaches the +5% keep gate",
                      17, "#9a4f00", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def vectorized_adamw_explicit_svg() -> str:
    data = json.loads(VECTORIZED_ADAMW_COMPARISON.read_text(encoding="utf-8"))
    operator = data["operator_rows"]
    model = data["qwen_vectorized_pilot"]
    width, height = 1900, 820
    left_x, right_x, top, panel_h = 95, 1300, 150, 500
    left_w, right_w = 1100, 500
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 049 · Vectorized AdamW Is Explicit-Only", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "MI300X · exact checkpoint element counts · HIP Event mean of 10",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{left_x}" y="{top}" width="{left_w}" height="{panel_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        f'<rect x="{right_x}" y="{top}" width="{right_w}" height="{panel_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        text(left_x + left_w / 2, 125, "float4 / scalar operator speedup (with mirror)", 20,
             anchor="middle", weight=700),
        text(right_x + right_w / 2, 125, "Forced Qwen speedup", 20,
             anchor="middle", weight=700),
    ]
    def operator_y(value: float) -> float:
        return top + panel_h * (1.25 - value) / 0.85
    for tick in (0.5, 0.75, 1.0, 1.25):
        y = operator_y(tick)
        parts.append(f'<line x1="{left_x}" y1="{y:.1f}" x2="{left_x+left_w}" '
                     f'y2="{y:.1f}" stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}"/>')
        parts.append(text(left_x - 10, y + 5, f"{tick:.2f}×", 13,
                          "#5b6474", anchor="end"))
    group = left_w / len(operator)
    for index, row in enumerate(operator):
        center = left_x + group * (index + 0.5)
        base = operator_y(1.0)
        value_y = operator_y(row["speedup"])
        color = "#2f9b68" if row["speedup"] >= 1.05 else (
            "#64748b" if row["speedup"] >= 1.0 else "#d04a3a")
        parts.append(f'<rect x="{center-23:.1f}" y="{min(base,value_y):.1f}" width="46" '
                     f'height="{max(3.0,abs(base-value_y)):.1f}" fill="{color}" rx="4"/>')
        label = f'{row["elements"]/1e6:.1f}M' if row["elements"] >= 1000000 \
            else f'{row["elements"]/1000:.0f}K'
        parts.append(text(center, top + panel_h + 29, label, 12,
                          "#445064", anchor="middle", rotate=-35))
    def model_y(value: float) -> float:
        return top + panel_h * (1.05 - value) / 0.15
    for tick in (0.90, 0.95, 1.0, 1.05):
        y = model_y(tick)
        parts.append(f'<line x1="{right_x}" y1="{y:.1f}" x2="{right_x+right_w}" '
                     f'y2="{y:.1f}" stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}"/>')
    group = right_w / len(model)
    for index, row in enumerate(model):
        center = right_x + group * (index + 0.5)
        base = model_y(1.0)
        value_y = model_y(row["speedup"])
        parts.append(f'<rect x="{center-35:.1f}" y="{base:.1f}" width="70" '
                     f'height="{max(3.0,value_y-base):.1f}" fill="#d04a3a" rx="4"/>')
        parts.append(text(center, top + panel_h + 29,
                          f'{row["batch"]}×{row["context"]}', 14,
                          anchor="middle", weight=700))
        parts.append(text(center, value_y + 22, f'{row["speedup"]:.3f}×', 13,
                          "#b83f32", anchor="middle", weight=700))
    parts.append(text(width / 2, 735,
                      "Keep implementation + benchmark; Auto remains Scalar because every official row regresses",
                      17, "#9a4f00", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def streaming_safetensors_load_svg() -> str:
    data = json.loads(STREAMING_LOAD_COMPARISON.read_text(encoding="utf-8"))
    loads = data["load_rows"]
    training = data["deepseek_training_rows"]
    width, height = 1800, 800
    left_x, right_x, top, panel_h = 105, 1080, 150, 470
    left_w, right_w = 850, 610
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 050 · Streaming Safetensors Load", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "MI300X · strict header preflight · original BF16 payload · bounded staging",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{left_x}" y="{top}" width="{left_w}" height="{panel_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        f'<rect x="{right_x}" y="{top}" width="{right_w}" height="{panel_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        text(left_x + left_w / 2, 125, "Checkpoint load seconds", 20,
             anchor="middle", weight=700),
        text(right_x + right_w / 2, 125, "DeepSeek training non-regression", 20,
             anchor="middle", weight=700),
    ]
    def load_y(seconds: float) -> float:
        return top + panel_h * (70.0 - seconds) / 70.0
    group = left_w / len(loads)
    for index, row in enumerate(loads):
        center = left_x + group * (index + 0.5)
        before = row["before_load_ms"] / 1000.0
        after = row["after_load_ms"] / 1000.0
        for offset, (value, color, label) in enumerate((
                (before, "#aab2bf", "before"), (after, "#18a558", "stream"))):
            x = center - 90 + offset * 100
            y = load_y(value)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="82" '
                         f'height="{load_y(0)-y:.1f}" fill="{color}" rx="5"/>')
            parts.append(text(x + 41, y - 9, f"{value:.3f}s", 14, color,
                              anchor="middle", weight=700))
            parts.append(text(x + 41, top + panel_h + 23, label, 12,
                              "#5b6474", anchor="middle"))
        model = "Qwen 0.5B" if row["model"].startswith("qwen") else "DeepSeek 1.5B"
        parts.append(text(center, top + panel_h + 55, model, 16,
                          anchor="middle", weight=700))
        parts.append(text(center, top + 70, f'{row["speedup"]:.1f}× faster', 19,
                          "#16834a", anchor="middle", weight=700))
    def training_y(value: float) -> float:
        return top + panel_h * (1.02 - value) / 0.04
    for tick in (0.98, 0.99, 1.0, 1.01, 1.02):
        y = training_y(tick)
        parts.append(f'<line x1="{right_x}" y1="{y:.1f}" x2="{right_x+right_w}" '
                     f'y2="{y:.1f}" stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}"/>')
    group = right_w / len(training)
    for index, row in enumerate(training):
        center = right_x + group * (index + 0.5)
        base = training_y(1.0)
        value_y = training_y(row["self_speedup"])
        color = "#2f9b68" if row["self_speedup"] >= 1.0 else "#64748b"
        parts.append(f'<rect x="{center-38:.1f}" y="{min(base,value_y):.1f}" width="76" '
                     f'height="{max(3.0,abs(base-value_y)):.1f}" fill="{color}" rx="4"/>')
        parts.append(text(center, top + panel_h + 31,
                          f'{row["batch"]}×{row["context"]}', 14,
                          anchor="middle", weight=700))
        parts.append(text(center, value_y - 9 if row["self_speedup"] >= 1.0 else value_y + 20,
                          f'{row["self_speedup"]:.3f}×', 13, color,
                          anchor="middle", weight=700))
    parts.append(text(width / 2, 715,
                      "H2D equals the BF16 file bytes; no D2H; training throughput and peak remain unchanged",
                      17, "#16834a", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def context512_training_profile_svg() -> str:
    comparison = json.loads(CONTEXT512_COMPARISON.read_text(encoding="utf-8"))
    profile = json.loads(CONTEXT512_PROFILE.read_text(encoding="utf-8"))
    rows = comparison["rows"]
    categories = profile["categories"]
    width, height = 1850, 820
    left_x, right_x, top, panel_h = 100, 1040, 150, 500
    left_w, right_w = 820, 710
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 051 · Context-512 Stable Failure", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "BF16 Linear / FP32 master · median of 3 fresh processes · Qwen retained profile",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{left_x}" y="{top}" width="{left_w}" height="{panel_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        f'<rect x="{right_x}" y="{top}" width="{right_w}" height="{panel_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        text(left_x + left_w / 2, 125, "microLLM / PyTorch", 20,
             anchor="middle", weight=700),
        text(right_x + right_w / 2, 125, "Qwen Kernel time categories", 20,
             anchor="middle", weight=700),
    ]
    def ratio_y(value: float) -> float:
        return top + panel_h * (1.30 - value) / 1.30
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
        y = ratio_y(tick)
        parts.append(f'<line x1="{left_x}" y1="{y:.1f}" x2="{left_x+left_w}" '
                     f'y2="{y:.1f}" stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}"/>')
    group = left_w / len(rows)
    for index, row in enumerate(rows):
        center = left_x + group * (index + 0.5)
        for offset, (value, color, label) in enumerate((
                (row["throughput_ratio"], "#d04a3a", "speed"),
                (row["peak_ratio"], "#64748b", "peak"))):
            x = center - 82 + offset * 88
            y = ratio_y(value)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="76" '
                         f'height="{ratio_y(0)-y:.1f}" fill="{color}" rx="5"/>')
            parts.append(text(x + 38, y - 9, f"{value:.3f}×", 14, color,
                              anchor="middle", weight=700))
            parts.append(text(x + 38, top + panel_h + 23, label, 12,
                              "#5b6474", anchor="middle"))
        label = "Qwen 0.5B" if row["model"].startswith("qwen") else "DeepSeek 1.5B"
        parts.append(text(center, top + panel_h + 55, label, 16,
                          anchor="middle", weight=700))
    maximum = max(row["percent"] for row in categories)
    for index, row in enumerate(categories):
        y = top + 45 + index * 68
        bar_w = 390 * row["percent"] / maximum
        color = "#d04a3a" if "GQA" in row["name"] else "#64748b"
        parts.append(text(right_x + 30, y, row["name"], 15, weight=600))
        parts.append(f'<rect x="{right_x+270}" y="{y-18}" width="390" height="26" '
                     'fill="#eef1f5" rx="4"/>')
        parts.append(f'<rect x="{right_x+270}" y="{y-18}" width="{bar_w:.1f}" '
                     f'height="26" fill="{color}" rx="4"/>')
        parts.append(text(right_x + 675, y + 1, f'{row["percent"]:.2f}%', 13,
                          color, anchor="end", weight=700))
    parts.append(text(width / 2, 735,
                      "Attention is 64.50% of Kernel time; backward atomics are the next falsifiable target",
                      17, "#9a4f00", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def split_kv_backward_discard_svg() -> str:
    data = json.loads(SPLIT_KV_COMPARISON.read_text(encoding="utf-8"))
    metrics = (
        ("End-to-end throughput", data["throughput_ratio"], "#d04a3a"),
        ("Attention backward", data["backward_speedup"], "#d04a3a"),
        ("Kernel dispatches", data["candidate_kernel_dispatches"] /
         data["baseline_kernel_dispatches"], "#64748b"),
        ("Measured peak", data["peak_ratio"], "#64748b"),
    )
    width, height = 1600, 720
    chart_x, chart_y, chart_w, chart_h = 150, 150, 1300, 390
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 052 · Split K/V Backward Discarded", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "Qwen2.5-0.5B · context 512 · ratio is candidate / retained baseline",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]
    def y(value: float) -> float:
        return chart_y + chart_h * (1.10 - value) / 0.50
    for tick in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1):
        position = y(tick)
        parts.append(f'<line x1="{chart_x}" y1="{position:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{position:.1f}" '
                     f'stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}"/>')
        parts.append(text(chart_x - 12, position + 5, f"{tick:.1f}×", 13,
                          "#5b6474", anchor="end"))
    group = chart_w / len(metrics)
    for index, (label, ratio, color) in enumerate(metrics):
        center = chart_x + group * (index + 0.5)
        base = y(0.6)
        top = y(ratio)
        parts.append(f'<rect x="{center-65:.1f}" y="{top:.1f}" width="130" '
                     f'height="{base-top:.1f}" fill="{color}" rx="6"/>')
        parts.append(text(center, top - 12, f"{ratio:.3f}×", 18, color,
                          anchor="middle", weight=700))
        parts.append(text(center, chart_y + chart_h + 34, label, 15,
                          "#172033", anchor="middle", weight=700))
    parts.append(text(width / 2, 620,
                      "Rows 478 ms + K/V rescan 843 ms > atomic backward 986 ms · code removed",
                      17, "#b83f32", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def strided_batched_hipblaslt_svg() -> str:
    data = json.loads(BATCHED_GEMM_COMPARISON.read_text(encoding="utf-8"))
    rows = data["rows"]
    width, height = 1650, 720
    chart_x, chart_y, chart_w, chart_h = 130, 145, 1390, 400
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 053 · Strided-Batched hipBLASLt", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "Qwen context-512 exact matrices · batch_heads=14 · HIP Event mean of 10",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]
    maximum = 120.0
    def y(value: float) -> float:
        return chart_y + chart_h * (maximum - value) / maximum
    for tick in (0, 25, 50, 75, 100):
        position = y(float(tick))
        parts.append(f'<line x1="{chart_x}" y1="{position:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{position:.1f}" stroke="#e5e9f0"/>')
        parts.append(text(chart_x - 12, position + 5, f"{tick}×", 13,
                          "#5b6474", anchor="end"))
    group = chart_w / len(rows)
    for index, row in enumerate(rows):
        center = chart_x + group * (index + 0.5)
        if row["valid_speedup"]:
            top = y(row["speedup"])
            parts.append(f'<rect x="{center-70:.1f}" y="{top:.1f}" width="140" '
                         f'height="{y(0)-top:.1f}" fill="#18a558" rx="6"/>')
            parts.append(text(center, top - 12, f'{row["speedup"]:.1f}×', 20,
                              "#16834a", anchor="middle", weight=700))
        else:
            top = y(12.0)
            parts.append(f'<rect x="{center-70:.1f}" y="{top:.1f}" width="140" '
                         f'height="{y(0)-top:.1f}" fill="#d6a33c" rx="6"/>')
            parts.append(text(center, top - 12, "baseline invalid", 17,
                              "#9a4f00", anchor="middle", weight=700))
            parts.append(text(center, top + 42, f'{row["hipblaslt_ms"]:.3f} ms', 14,
                              "#ffffff", anchor="middle", weight=700))
        parts.append(text(center, chart_y + chart_h + 35, row["name"], 17,
                          anchor="middle", weight=700))
    parts.append(text(width / 2, 625,
                      "All transpose layouts pass; Auto stays unchanged until model integration passes",
                      17, "#2563eb", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = {PROGRESS: progress_svg(rows()), BOTTLENECK: bottleneck_svg(),
                BF16_CHART: bf16_svg(), BF16_POLICY_CHART: bf16_policy_svg(),
                BF16_FFN_CHART: bf16_ffn_svg(),
                BF16_MODEL_CHART: bf16_model_inference_svg(),
                BF16_PREFILL_CHART: bf16_prefill_allocator_svg(),
                BF16_ATTENTION_CHART: bf16_attention_svg(),
                BF16_PLAN_CHART: bf16_plan_cache_svg(),
                BF16_TRAINING_CHART: bf16_training_svg(),
                BF16_TRAINING_QKV_CHART: bf16_training_qkv_svg(),
                BF16_TRAINING_MIRROR_CHART: bf16_training_mirror_svg(),
                BF16_TRAINING_ISLAND_CHART: bf16_training_island_svg(),
                BF16_TRAINING_SHAPE_CHART: bf16_training_shape_svg(),
                BF16_WEIGHT_GRADIENT_CHART: bf16_weight_gradient_svg(),
                FUSED_CAUSAL_GQA_CHART: fused_causal_gqa_svg(),
                DEEPSEEK_SHAPE_CHART: deepseek_shape_svg(),
                DEEPSEEK_PROFILE_CHART: deepseek_context128_profile_svg(),
                STABLE_GRADIENT_CHART: stable_gradient_discard_svg(),
                CHUNKED_ADAMW_CHART: chunked_adamw_discard_svg(),
                VECTORIZED_ADAMW_CHART: vectorized_adamw_explicit_svg(),
                STREAMING_LOAD_CHART: streaming_safetensors_load_svg(),
                CONTEXT512_CHART: context512_training_profile_svg(),
                SPLIT_KV_CHART: split_kv_backward_discard_svg(),
                BATCHED_GEMM_CHART: strided_batched_hipblaslt_svg()}
    if args.check:
        stale = [str(path.relative_to(ROOT)) for path, value in expected.items()
                 if not path.is_file() or path.read_text(encoding="utf-8") != value]
        if stale:
            raise SystemExit("stale generated optimization assets: " + ", ".join(stale))
        print("optimization charts are current")
        return 0
    for path, value in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
