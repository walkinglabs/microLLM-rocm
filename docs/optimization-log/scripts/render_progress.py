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
                BF16_PLAN_CHART: bf16_plan_cache_svg()}
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
