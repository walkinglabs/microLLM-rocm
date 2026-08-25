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
BATCHED_BACKWARD_COMPARISON = ROOT / "experiments" / "054-data" / "comparison.json"
BATCHED_BACKWARD_PROFILE = ROOT / "experiments" / "054-data" / "profile-summary.json"
BATCHED_BACKWARD_CHART = ROOT / "assets" / "batched-attention-backward.svg"
SAVED_ATTENTION_COMPARISON = ROOT / "experiments" / "055-data" / "comparison.json"
SAVED_ATTENTION_PROFILE = ROOT / "experiments" / "055-data" / "profile-summary.json"
SAVED_ATTENTION_CHART = ROOT / "assets" / "saved-attention-probabilities.svg"
BF16_ADAMW_SUMMARY = (ROOT.parents[1] / "benchmarks" / "results" /
                      "2026-08-24-bf16-adamw-moments" / "formal" /
                      "summary.json")
BF16_ADAMW_CHART = ROOT / "assets" / "bf16-adamw-moments.svg"
HYBRID_ADAMW_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                     "2026-08-24-hybrid-bf16-adamw")
HYBRID_ADAMW_CHART = ROOT / "assets" / "hybrid-bf16-adamw.svg"
POST_HYBRID_PROFILE_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                            "2026-08-24-post-hybrid-training-profile")
POST_HYBRID_PROFILE_CHART = ROOT / "assets" / "post-hybrid-training-profile.svg"
GROUPED_WGRAD_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                      "2026-08-24-grouped-weight-gradient-discard")
GROUPED_WGRAD_CHART = ROOT / "assets" / "grouped-weight-gradient-discard.svg"
PACKED_WGRAD_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                     "2026-08-24-packed-weight-gradient-discard")
PACKED_WGRAD_CHART = ROOT / "assets" / "packed-weight-gradient-discard.svg"
FP32_WGRAD_SOLUTION_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                            "2026-08-24-fp32-weight-gradient-solutions")
FP32_WGRAD_SOLUTION_CHART = (
    ROOT / "assets" / "fp32-weight-gradient-solutions-discard.svg")
TRAINING_GRAPH_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                       "2026-08-24-training-graph-capture")
TRAINING_GRAPH_CHART = ROOT / "assets" / "training-graph-capture-boundary.svg"
ADAMW_GRAPH_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                    "2026-08-24-adamw-graph-replay")
ADAMW_GRAPH_CHART = ROOT / "assets" / "adamw-graph-replay.svg"
ADAMW_GRAPH_MULTI_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                          "2026-08-24-adamw-graph-multi")
ADAMW_GRAPH_MULTI_CHART = ROOT / "assets" / "adamw-graph-multi.svg"
GRADIENT_ADDRESS_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                         "2026-08-24-gradient-address-stability")
GRADIENT_ADDRESS_CHART = ROOT / "assets" / "gradient-address-stability.svg"
OPTIMIZER_GRAPH_PREFLIGHT_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                                  "2026-08-24-optimizer-graph-model-preflight")
OPTIMIZER_GRAPH_PREFLIGHT_CHART = (
    ROOT / "assets" / "optimizer-graph-model-preflight.svg")
QUIESCENT_HANDOFF_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                          "2026-08-24-quiescent-allocator-handoff")
QUIESCENT_HANDOFF_CHART = ROOT / "assets" / "quiescent-allocator-handoff.svg"
OPTIMIZER_GRAPH_MODEL_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                              "2026-08-24-optimizer-graph-model-gate")
OPTIMIZER_GRAPH_MODEL_CHART = ROOT / "assets" / "optimizer-graph-model-gate.svg"
ROCWMMA_QK_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                   "2026-08-25-rocwmma-qk-tile")
ROCWMMA_QK_CHART = ROOT / "assets" / "rocwmma-qk-tile.svg"
ROCWMMA_ONLINE_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                       "2026-08-25-rocwmma-online-attention")
ROCWMMA_ONLINE_CHART = ROOT / "assets" / "rocwmma-online-attention.svg"
ROCWMMA_OPERATOR_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                         "2026-08-25-rocwmma-online-operator")
ROCWMMA_OPERATOR_CHART = ROOT / "assets" / "rocwmma-online-operator.svg"
ROCWMMA_MODEL_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                      "2026-08-25-rocwmma-online-model-gate")
ROCWMMA_MODEL_CHART = ROOT / "assets" / "rocwmma-online-model-discard.svg"
ROCWMMA_DIRECT_MODEL_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                             "2026-08-25-rocwmma-direct-bf16-model-gate")
ROCWMMA_DIRECT_MODEL_CHART = (
    ROOT / "assets" / "rocwmma-direct-bf16-model-discard.svg")
CURRENT_INFERENCE_PROFILE_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                                  "2026-08-25-current-inference-profile")
CURRENT_INFERENCE_PROFILE_CHART = (
    ROOT / "assets" / "current-inference-profile.svg")
FP32_ATTENTION_T1024_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                             "2026-08-25-fp32-attention-t1024-solutions")
FP32_ATTENTION_T1024_MODEL_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-fp32-attention-t1024-qk-model-gate")
FP32_ATTENTION_T1024_CHART = (
    ROOT / "assets" / "fp32-attention-t1024-discard.svg")
BF16_SWIGLU_VECTOR_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                           "2026-08-25-bf16-swiglu-vector-operator")
BF16_SWIGLU_VECTOR_MODEL_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-bf16-swiglu-vector-model-gate")
BF16_SWIGLU_VECTOR_CHART = ROOT / "assets" / "bf16-swiglu-vector-discard.svg"
BF16_GROUPED_SWISH_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                           "2026-08-25-bf16-grouped-swish-operator")
BF16_GROUPED_SWISH_MODEL_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-bf16-grouped-swish-model-gate")
BF16_GROUPED_SWISH_CHART = ROOT / "assets" / "bf16-grouped-swish-discard.svg"
BF16_RMS_NORM_OUTPUT_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                             "2026-08-25-bf16-rms-norm-output-operator")
BF16_RMS_NORM_OUTPUT_CHART = ROOT / "assets" / "bf16-rms-norm-output.svg"
BF16_FFN_NORM_MODEL_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                            "2026-08-25-bf16-ffn-norm-model-gate")
BF16_FFN_NORM_MODEL_CHART = ROOT / "assets" / "bf16-ffn-norm-model.svg"
POST_BF16_FFN_NORM_PROFILE_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-post-bf16-ffn-norm-profile")
POST_BF16_FFN_NORM_PROFILE_CHART = (
    ROOT / "assets" / "post-bf16-ffn-norm-profile.svg")
BF16_ATTENTION_NORM_MODEL_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-bf16-attention-norm-model-gate")
BF16_ATTENTION_NORM_MODEL_CHART = (
    ROOT / "assets" / "bf16-attention-norm-model.svg")
POST_BF16_ATTENTION_NORM_PROFILE_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-post-bf16-attention-norm-profile")
POST_BF16_ATTENTION_NORM_PROFILE_CHART = (
    ROOT / "assets" / "post-bf16-attention-norm-profile.svg")
BF16_PV_OUTPUT_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                       "2026-08-25-bf16-pv-output-capability")
BF16_PV_OUTPUT_CHART = ROOT / "assets" / "bf16-pv-output-discard.svg"
BF16_VALUE_PV_ROOT = (ROOT.parents[1] / "benchmarks" / "results" /
                      "2026-08-25-bf16-value-pv-capability")
BF16_VALUE_PV_CHART = ROOT / "assets" / "bf16-value-pv-discard.svg"
INFERENCE_LOCAL_SATURATION_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-inference-local-saturation")
INFERENCE_LOCAL_SATURATION_CHART = (
    ROOT / "assets" / "inference-local-saturation.svg")
CURRENT_TRAINING_PROFILE_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-current-training-profile")
CURRENT_TRAINING_PROFILE_CHART = (
    ROOT / "assets" / "current-training-profile.svg")
BF16_WGRAD_SHAPE_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-bf16-weight-gradient-operator")
BF16_WGRAD_SHAPE_CHART = (
    ROOT / "assets" / "bf16-weight-gradient-shapes.svg")
BF16_WGRAD_MODEL_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-bf16-weight-gradient-model-gate")
BF16_WGRAD_MODEL_CHART = (
    ROOT / "assets" / "bf16-weight-gradient-model.svg")
BF16_WGRAD_TRAJECTORY_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-bf16-weight-gradient-trajectory")
BF16_WGRAD_TRAJECTORY_CHART = (
    ROOT / "assets" / "bf16-weight-gradient-trajectory-discard.svg")
BF16_WGRAD_ALLOCATION_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-bf16-weight-gradient-allocation-attribution")
BF16_WGRAD_ALLOCATION_CHART = (
    ROOT / "assets" / "bf16-weight-gradient-allocation-attribution.svg")
BF16_WGRAD_WORKSPACE_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-bf16-weight-gradient-workspace-gate")
BF16_WGRAD_WORKSPACE_CHART = (
    ROOT / "assets" / "bf16-weight-gradient-workspace-discard.svg")
TRAINING_LOCAL_SATURATION_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-training-local-saturation")
TRAINING_LOCAL_SATURATION_CHART = (
    ROOT / "assets" / "training-local-saturation.svg")
CURRENT_DATA_PARALLEL_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-current-data-parallel")
CURRENT_DATA_PARALLEL_CHART = (
    ROOT / "assets" / "current-data-parallel-audit.svg")
DATA_PARALLEL_VERIFICATION_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-data-parallel-verification-matrix")
DATA_PARALLEL_VERIFICATION_CHART = (
    ROOT / "assets" / "data-parallel-verification-interval.svg")
DATA_PARALLEL_BUCKET_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-data-parallel-bucket-matrix")
DATA_PARALLEL_BUCKET_CHART = (
    ROOT / "assets" / "data-parallel-bucket-matrix.svg")
DATA_PARALLEL_MODEL_S_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-data-parallel-model-s-bucket-matrix")
DATA_PARALLEL_MODEL_S_CHART = (
    ROOT / "assets" / "data-parallel-model-s-buckets.svg")
DATA_PARALLEL_COPY_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-data-parallel-bucket-copy-attribution")
DATA_PARALLEL_COPY_CHART = (
    ROOT / "assets" / "data-parallel-bucket-copy-attribution.svg")
DATA_PARALLEL_INPLACE_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-data-parallel-inplace-average")
DATA_PARALLEL_INPLACE_CHART = (
    ROOT / "assets" / "data-parallel-inplace-average.svg")
DATA_PARALLEL_PERSISTENT_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-data-parallel-persistent-buckets")
DATA_PARALLEL_PERSISTENT_CHART = (
    ROOT / "assets" / "data-parallel-persistent-buckets.svg")
DATA_PARALLEL_GRADIENT_VIEW_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-data-parallel-gradient-views")
DATA_PARALLEL_GRADIENT_VIEW_CHART = (
    ROOT / "assets" / "data-parallel-gradient-bucket-views.svg")
DATA_PARALLEL_DIRECT_GRADIENT_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-data-parallel-direct-bucket-gradients")
DATA_PARALLEL_DIRECT_GRADIENT_CHART = (
    ROOT / "assets" / "data-parallel-direct-bucket-gradient-discard.svg")
GRADIENT_PRODUCER_OUT_ROOT = (
    ROOT.parents[1] / "benchmarks" / "results" /
    "2026-08-25-gradient-producer-out-matrix")
GRADIENT_PRODUCER_OUT_CHART = (
    ROOT / "assets" / "gradient-producer-out-matrix.svg")


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


def batched_attention_backward_svg() -> str:
    comparison = json.loads(BATCHED_BACKWARD_COMPARISON.read_text(encoding="utf-8"))
    profile = json.loads(BATCHED_BACKWARD_PROFILE.read_text(encoding="utf-8"))
    rows = comparison["rows"]
    width, height = 1800, 790
    left_x, right_x, top, panel_h = 100, 1050, 150, 470
    left_w, right_w = 820, 650
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 054 · Batched Attention Backward", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "context 512 · median of 3 fresh processes · measured peak unchanged",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{left_x}" y="{top}" width="{left_w}" height="{panel_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        f'<rect x="{right_x}" y="{top}" width="{right_w}" height="{panel_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        text(left_x + left_w / 2, 125, "Official training ratios", 20,
             anchor="middle", weight=700),
        text(right_x + right_w / 2, 125, "Qwen process profile", 20,
             anchor="middle", weight=700),
    ]
    def ratio_y(value: float) -> float:
        return top + panel_h * (1.5 - value) / 1.5
    for tick in (0.0, 0.5, 1.0, 1.5):
        y = ratio_y(tick)
        parts.append(f'<line x1="{left_x}" y1="{y:.1f}" x2="{left_x+left_w}" '
                     f'y2="{y:.1f}" stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}"/>')
    group = left_w / len(rows)
    for index, row in enumerate(rows):
        center = left_x + group * (index + 0.5)
        for offset, (value, color, label) in enumerate((
                (row["self_speedup"], "#18a558", "self"),
                (row["pytorch_ratio_after"], "#d04a3a", "vs PT"))):
            x = center - 82 + offset * 88
            y = ratio_y(value)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="76" '
                         f'height="{ratio_y(0)-y:.1f}" fill="{color}" rx="5"/>')
            parts.append(text(x + 38, y - 9, f"{value:.3f}×", 14, color,
                              anchor="middle", weight=700))
            parts.append(text(x + 38, top + panel_h + 24, label, 12,
                              "#5b6474", anchor="middle"))
        label = "Qwen 0.5B" if row["model"].startswith("qwen") else "DeepSeek 1.5B"
        parts.append(text(center, top + panel_h + 56, label, 16,
                          anchor="middle", weight=700))
    profile_rows = (
        ("Total Kernel", profile["before"]["kernel_time_ns"],
         profile["after"]["kernel_time_ns"]),
        ("Attention backward", profile["before"]["atomic_backward_time_ns"],
         profile["identified_backward_time_ns"]),
    )
    maximum = max(before for _, before, _ in profile_rows)
    for index, (label, before, after) in enumerate(profile_rows):
        y = top + 95 + index * 175
        parts.append(text(right_x + 35, y - 30, label, 17, weight=700))
        for offset, (value, color, name) in enumerate((
                (before, "#aab2bf", "before"), (after, "#18a558", "after"))):
            bar_y = y + offset * 52
            bar_w = 470 * value / maximum
            parts.append(f'<rect x="{right_x+35}" y="{bar_y}" width="470" height="30" '
                         'fill="#eef1f5" rx="4"/>')
            parts.append(f'<rect x="{right_x+35}" y="{bar_y}" width="{bar_w:.1f}" '
                         f'height="30" fill="{color}" rx="4"/>')
            parts.append(text(right_x + 525, bar_y + 21,
                              f'{name} {value/1e6:.1f} ms', 13, color,
                              anchor="start", weight=700))
    parts.append(text(width / 2, 710,
                      "Kernel time 1.350× faster despite +4.3% dispatches · T128 fallback 1.008×",
                      17, "#16834a", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def saved_attention_probabilities_svg() -> str:
    comparison = json.loads(SAVED_ATTENTION_COMPARISON.read_text(encoding="utf-8"))
    profile = json.loads(SAVED_ATTENTION_PROFILE.read_text(encoding="utf-8"))
    rows = comparison["rows"]
    width, height = 1800, 790
    left_x, right_x, top, panel_h = 100, 1080, 150, 470
    left_w, right_w = 850, 620
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 055 · Saved Attention Probabilities", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "context 512 · explicit speed / memory trade-off · median of 3",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{left_x}" y="{top}" width="{left_w}" height="{panel_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        f'<rect x="{right_x}" y="{top}" width="{right_w}" height="{panel_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        text(left_x + left_w / 2, 125, "Model speedup and peak cost", 20,
             anchor="middle", weight=700),
        text(right_x + right_w / 2, 125, "Qwen row/forward Kernel time", 20,
             anchor="middle", weight=700),
    ]
    def ratio_y(value: float) -> float:
        return top + panel_h * (1.20 - value) / 0.25
    for tick in (0.95, 1.0, 1.05, 1.10, 1.15, 1.20):
        y = ratio_y(tick)
        parts.append(f'<line x1="{left_x}" y1="{y:.1f}" x2="{left_x+left_w}" '
                     f'y2="{y:.1f}" stroke="{("#2563eb" if tick == 1.0 else "#e5e9f0")}"/>')
    group = left_w / len(rows)
    for index, row in enumerate(rows):
        center = left_x + group * (index + 0.5)
        for offset, (value, color, label) in enumerate((
                (row["self_speedup"], "#18a558", "speed"),
                (row["peak_ratio"], "#d6a33c", "peak"))):
            x = center - 82 + offset * 88
            base = ratio_y(1.0)
            value_y = ratio_y(value)
            parts.append(f'<rect x="{x:.1f}" y="{min(base,value_y):.1f}" width="76" '
                         f'height="{max(3.0,abs(base-value_y)):.1f}" fill="{color}" rx="5"/>')
            parts.append(text(x + 38, value_y - 9 if value >= 1 else value_y + 20,
                              f"{value:.3f}×", 14, color, anchor="middle", weight=700))
            parts.append(text(x + 38, top + panel_h + 24, label, 12,
                              "#5b6474", anchor="middle"))
        label = "Qwen 0.5B" if row["model"].startswith("qwen") else "DeepSeek 1.5B"
        parts.append(text(center, top + panel_h + 56, label, 16,
                          anchor="middle", weight=700))
    kernel_rows = (
        ("Backward row", profile["before"]["row_backward_time_ns"],
         profile["after"]["saved_row_backward_time_ns"]),
        ("Forward row", profile["before"]["forward_time_ns"],
         profile["after"]["forward_time_ns"]),
    )
    maximum = max(before for _, before, _ in kernel_rows)
    for index, (label, before, after) in enumerate(kernel_rows):
        y = top + 95 + index * 175
        parts.append(text(right_x + 35, y - 30, label, 17, weight=700))
        for offset, (value, color, name) in enumerate((
                (before, "#aab2bf", "before"), (after, "#18a558", "after"))):
            bar_y = y + offset * 52
            bar_w = 430 * value / maximum
            parts.append(f'<rect x="{right_x+35}" y="{bar_y}" width="430" height="30" '
                         'fill="#eef1f5" rx="4"/>')
            parts.append(f'<rect x="{right_x+35}" y="{bar_y}" width="{bar_w:.1f}" '
                         f'height="30" fill="{color}" rx="4"/>')
            parts.append(text(right_x + 485, bar_y + 21,
                              f'{name} {value/1e6:.1f} ms', 13, color, weight=700))
    parts.append(text(width / 2, 710,
                      "Both models +336 MiB peak · T128 fallback 0.991× · saved row 1.553× faster",
                      17, "#9a4f00", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_adamw_moments_svg() -> str:
    data = json.loads(BF16_ADAMW_SUMMARY.read_text(encoding="utf-8"))
    rows = data["models"]
    width, height = 1700, 760
    left_x, right_x, top, panel_h = 95, 965, 145, 470
    left_w, right_w = 760, 640
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 214 · BF16 AdamW Moments", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "B1/T512 · 1 warm-up + 2 measured · median of 5 fresh processes",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{left_x}" y="{top}" width="{left_w}" height="{panel_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        f'<rect x="{right_x}" y="{top}" width="{right_w}" height="{panel_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        text(left_x + left_w / 2, 125, "Speed ratios · BF16 / FP32 moments", 20,
             anchor="middle", weight=700),
        text(right_x + right_w / 2, 125, "Measured engine peak memory", 20,
             anchor="middle", weight=700),
    ]

    def speed_y(value: float) -> float:
        return top + panel_h * (1.25 - value) / 0.30

    for tick in (0.95, 1.0, 1.05, 1.10, 1.15, 1.20, 1.25):
        y = speed_y(tick)
        color = "#2563eb" if tick == 1.0 else "#d97706" if tick == 1.10 else "#e5e9f0"
        dash = ' stroke-dasharray="8 6"' if tick in (1.0, 1.10) else ""
        parts.append(f'<line x1="{left_x}" y1="{y:.1f}" '
                     f'x2="{left_x+left_w}" y2="{y:.1f}" stroke="{color}"{dash}/>')
        parts.append(text(left_x - 12, y + 5, f"{tick:.2f}×", 13,
                          "#5b6474", anchor="end"))
    group = left_w / len(rows)
    for index, row in enumerate(rows):
        center = left_x + group * (index + 0.5)
        for offset, (value, color, label) in enumerate((
                (row["throughput_speedup"], "#18a558", "end-to-end"),
                (row["optimizer_speedup"], "#7c3aed", "optimizer"))):
            x = center - 92 + offset * 104
            base = speed_y(0.95)
            y = speed_y(value)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="84" '
                         f'height="{base-y:.1f}" fill="{color}" rx="5"/>')
            parts.append(text(x + 42, y - 10, f"{value:.3f}×", 14,
                              color, anchor="middle", weight=700))
            parts.append(text(x + 42, top + panel_h + 24, label, 12,
                              "#5b6474", anchor="middle"))
        model_label = "Qwen 0.5B" if row["model"].startswith("qwen") else "DeepSeek 1.5B"
        parts.append(text(center, top + panel_h + 55, model_label, 16,
                          anchor="middle", weight=700))

    maximum_gib = max(row["fp32_peak_bytes_median"] for row in rows) / (1024 ** 3)
    for index, row in enumerate(rows):
        y = top + 85 + index * 175
        model_label = "Qwen 0.5B" if row["model"].startswith("qwen") else "DeepSeek 1.5B"
        parts.append(text(right_x + 30, y - 28, model_label, 16, weight=700))
        for offset, (field, color, label) in enumerate((
                ("fp32_peak_bytes_median", "#9aa3b2", "FP32 moments"),
                ("bf16_peak_bytes_median", "#18a558", "BF16 moments"))):
            value = row[field] / (1024 ** 3)
            bar_y = y + offset * 52
            bar_w = 405 * value / maximum_gib
            parts.append(f'<rect x="{right_x+30}" y="{bar_y}" width="405" height="30" '
                         'fill="#eef1f5" rx="4"/>')
            parts.append(f'<rect x="{right_x+30}" y="{bar_y}" width="{bar_w:.1f}" '
                         f'height="30" fill="{color}" rx="4"/>')
            parts.append(text(right_x + 455, bar_y + 21,
                              f"{label} {value:.2f} GiB", 13, color, weight=700))
        parts.append(text(right_x + 455, y + 127,
                          f'moment bytes 2× smaller · peak {row["peak_ratio"]:.3f}×',
                          13, "#166534", weight=700))
    parts.append(text(width / 2, 710,
                      "Required gates pass · Qwen optimizer 1.069× misses the 1.10× stretch gate",
                      17, "#9a4f00", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def hybrid_bf16_adamw_svg() -> str:
    thresholds = (4096, 65536, 262144, 1048576, 4194304, 16777216)
    rows = []
    for threshold in thresholds:
        summary = json.loads((HYBRID_ADAMW_ROOT /
                              f"threshold-{threshold}" / "summary.json").read_text(
                                  encoding="utf-8"))
        models = {row["model"]: row for row in summary["models"]}
        qwen = models["qwen2.5-0.5b"]
        deep = models["deepseek-r1-distill-qwen-1.5b"]
        rows.append({
            "threshold": threshold,
            "qwen_optimizer": qwen["optimizer_speedup"],
            "deep_optimizer": deep["optimizer_speedup"],
            "e2e_geomean": math.sqrt(
                qwen["throughput_speedup"] * deep["throughput_speedup"]),
        })
    formal = json.loads((HYBRID_ADAMW_ROOT / "formal-threshold-1048576" /
                         "summary.json").read_text(encoding="utf-8"))
    formal_rows = formal["models"]
    width, height = 1750, 760
    chart_x, chart_y, chart_w, chart_h = 105, 145, 1010, 455
    panel_x, panel_y, panel_w, panel_h = 1215, 145, 440, 455
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 215 · Hybrid BF16 AdamW", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "six thresholds · B1/T512 · 3-process sweep + 5-process formal gate",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        text(chart_x + chart_w / 2, 125, "Threshold sweep", 20,
             anchor="middle", weight=700),
        text(panel_x + panel_w / 2, 125, "1M formal ratios", 20,
             anchor="middle", weight=700),
    ]

    def ratio_y(value: float) -> float:
        return chart_y + chart_h * (1.30 - value) / 0.45

    for tick in (0.85, 0.95, 1.0, 1.10, 1.20, 1.30):
        y = ratio_y(tick)
        color = "#2563eb" if tick == 1.0 else "#d97706" if tick == 1.10 else "#e5e9f0"
        parts.append(f'<line x1="{chart_x}" y1="{y:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{y:.1f}" stroke="{color}"/>')
        parts.append(text(chart_x - 12, y + 5, f"{tick:.2f}×", 13,
                          "#5b6474", anchor="end"))
    labels = ("4K", "64K", "256K", "1M", "4M", "16M")
    series = (
        ("qwen_optimizer", "#7c3aed", "Qwen optimizer"),
        ("deep_optimizer", "#d97706", "DeepSeek optimizer"),
        ("e2e_geomean", "#18a558", "E2E geometric mean"),
    )
    points: dict[str, list[tuple[float, float]]] = {key: [] for key, _, _ in series}
    for index, row in enumerate(rows):
        x = chart_x + chart_w * (index + 0.5) / len(rows)
        parts.append(text(x, chart_y + chart_h + 30, labels[index], 14,
                          "#5b6474", anchor="middle", weight=600))
        for key, _, _ in series:
            points[key].append((x, ratio_y(row[key])))
    for key, color, label in series:
        path = " ".join(("M" if index == 0 else "L") +
                        f" {x:.1f} {y:.1f}"
                        for index, (x, y) in enumerate(points[key]))
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="4"/>')
        for index, (x, y) in enumerate(points[key]):
            marker = "#18a558" if index == 3 else "#dc2626" if index == 5 else color
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" '
                         f'fill="{marker}" stroke="#ffffff" stroke-width="2"/>')
        legend_x = chart_x + 30 + list(s[0] for s in series).index(key) * 300
        parts.append(f'<line x1="{legend_x}" y1="{chart_y+28}" '
                     f'x2="{legend_x+34}" y2="{chart_y+28}" stroke="{color}" stroke-width="4"/>')
        parts.append(text(legend_x + 44, chart_y + 33, label, 13,
                          "#5b6474", weight=600))
    for index, row in enumerate(formal_rows):
        model = "Qwen" if row["model"].startswith("qwen") else "DeepSeek"
        y = panel_y + 80 + index * 180
        parts.append(text(panel_x + 25, y - 28, model, 16, weight=700))
        for offset, (field, color, label) in enumerate((
                ("throughput_speedup", "#18a558", "end-to-end"),
                ("optimizer_speedup", "#7c3aed", "optimizer"))):
            value = row[field]
            bar_y = y + offset * 53
            bar_w = 310 * (value - 0.9) / 0.4
            parts.append(f'<rect x="{panel_x+25}" y="{bar_y}" width="310" height="30" '
                         'fill="#eef1f5" rx="4"/>')
            parts.append(f'<rect x="{panel_x+25}" y="{bar_y}" width="{bar_w:.1f}" '
                         f'height="30" fill="{color}" rx="4"/>')
            parts.append(text(panel_x + 350, bar_y + 21,
                              f"{label} {value:.3f}×", 13, color,
                              anchor="end", weight=700))
    parts.append(text(width / 2, 700,
                      "Keep 1M · 16M selects 1.31B DeepSeek elements and falls to 0.896× optimizer / 0.980× E2E",
                      17, "#9a4f00", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def post_hybrid_training_profile_svg() -> str:
    summary = json.loads((POST_HYBRID_PROFILE_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    comparisons = summary["comparisons"]
    profiles = {
        model: json.loads((POST_HYBRID_PROFILE_ROOT / model /
                           "profile-delta.json").read_text(encoding="utf-8"))
        for model in ("qwen", "deepseek")
    }
    width, height = 1720, 760
    left_x, right_x, top, panel_h = 100, 930, 145, 460
    left_w, right_w = 720, 690
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 216 · Post-Hybrid Training Profile", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "load + 3 steps minus load + 1 step · per-step Kernel time",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{left_x}" y="{top}" width="{left_w}" height="{panel_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        f'<rect x="{right_x}" y="{top}" width="{right_w}" height="{panel_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        text(left_x + left_w / 2, 125, "Kernel time before / after", 20,
             anchor="middle", weight=700),
        text(right_x + right_w / 2, 125, "Current category share", 20,
             anchor="middle", weight=700),
    ]
    maximum_ms = max(row["previous_kernel_ns_per_step"] for row in comparisons) / 1e6
    for index, row in enumerate(comparisons):
        y = top + 85 + index * 185
        model = "Qwen 0.5B" if row["model"].startswith("qwen") else "DeepSeek 1.5B"
        parts.append(text(left_x + 25, y - 28, model, 16, weight=700))
        for offset, (value, color, label) in enumerate((
                (row["previous_kernel_ns_per_step"] / 1e6, "#9aa3b2", "Exp213"),
                (row["kernel_ns_per_step"] / 1e6, "#18a558", "hybrid"))):
            bar_y = y + offset * 52
            bar_w = 500 * value / maximum_ms
            parts.append(f'<rect x="{left_x+25}" y="{bar_y}" width="500" height="30" '
                         'fill="#eef1f5" rx="4"/>')
            parts.append(f'<rect x="{left_x+25}" y="{bar_y}" width="{bar_w:.1f}" '
                         f'height="30" fill="{color}" rx="4"/>')
            parts.append(text(left_x + 545, bar_y + 21,
                              f"{label} {value:.2f} ms", 13, color, weight=700))
        parts.append(text(left_x + 545, y + 128,
                          f'AdamW {row["adamw_speedup_vs_experiment_213"]:.3f}× faster',
                          13, "#166534", weight=700))
    shown = ("hipBLASLt GEMM", "AdamW", "other kernels",
             "RMSNorm forward/backward", "bias gradient", "cross entropy",
             "FP32/BF16 cast")
    colors = ("#2563eb", "#7c3aed", "#94a3b8", "#0f766e",
              "#d97706", "#dc2626", "#16a34a")
    for model_index, model in enumerate(("qwen", "deepseek")):
        categories = {row["category"]: row for row in profiles[model]["categories"]}
        y = top + 48 + model_index * 205
        label = "Qwen 0.5B" if model == "qwen" else "DeepSeek 1.5B"
        parts.append(text(right_x + 25, y, label, 16, weight=700))
        for index, (name, color) in enumerate(zip(shown, colors, strict=True)):
            row = categories[name]
            bar_y = y + 23 + index * 23
            bar_w = 420 * row["kernel_share"] / 0.65
            short = ("GEMM" if name == "hipBLASLt GEMM" else
                     "RMSNorm" if name.startswith("RMSNorm") else
                     "bias grad" if name == "bias gradient" else
                     "cross entropy" if name == "cross entropy" else
                     "cast" if name == "FP32/BF16 cast" else name)
            parts.append(text(right_x + 25, bar_y + 13, short, 11,
                              "#5b6474"))
            parts.append(f'<rect x="{right_x+150}" y="{bar_y}" width="420" height="16" '
                         'fill="#eef1f5" rx="3"/>')
            parts.append(f'<rect x="{right_x+150}" y="{bar_y}" width="{bar_w:.1f}" '
                         f'height="16" fill="{color}" rx="3"/>')
            parts.append(text(right_x + 590, bar_y + 13,
                              f'{row["kernel_share"]*100:.2f}%', 11, color,
                              anchor="end", weight=700))
    parts.append(text(width / 2, 700,
                      "AdamW threshold track closed · GEMM now owns 59.33% / 63.81%",
                      17, "#9a4f00", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def grouped_weight_gradient_discard_svg() -> str:
    records = [json.loads(line) for line in (GROUPED_WGRAD_ROOT / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    by_key = {(row["input_layout"], row["model"], row["projection"]): row
              for row in records}
    width, height = 1640, 700
    chart_x, chart_y, chart_w, chart_h = 140, 155, 1360, 350
    columns = (("qwen", "qkv", "Qwen QKV"),
               ("qwen", "gate-up", "Qwen gate/up"),
               ("deepseek", "qkv", "DeepSeek QKV"),
               ("deepseek", "gate-up", "DeepSeek gate/up"))
    rows = (("direct", "direct N,T · 8,153 inventory"),
            ("materialized", "shared transpose + N,N · 9,172 inventory"))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 217 · Grouped Weight-Gradient Capability", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "FP32 · rows 512 · every inventory candidate screened by isAlgoSupported",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]
    cell_w = chart_w / len(columns)
    cell_h = chart_h / len(rows)
    for column, (_, _, label) in enumerate(columns):
        x = chart_x + cell_w * (column + 0.5)
        parts.append(text(x, chart_y - 18, label, 15, anchor="middle", weight=700))
    for row_index, (layout, label) in enumerate(rows):
        y = chart_y + cell_h * (row_index + 0.5)
        parts.append(text(chart_x - 18, y - 7, label, 14, "#5b6474",
                          anchor="end", weight=600))
        for column, (model, projection, _) in enumerate(columns):
            record = by_key[(layout, model, projection)]
            x = chart_x + cell_w * column + 20
            box_y = chart_y + cell_h * row_index + 20
            parts.append(f'<rect x="{x:.1f}" y="{box_y:.1f}" '
                         f'width="{cell_w-40:.1f}" height="{cell_h-40:.1f}" '
                         'fill="#fff1f2" stroke="#e11d48" stroke-width="2" rx="8"/>')
            parts.append(text(x + (cell_w - 40) / 2, box_y + 48,
                              "0 supported", 20, "#be123c", anchor="middle", weight=700))
            parts.append(text(x + (cell_w - 40) / 2, box_y + 78,
                              f'{record["algorithm_count"]:,} inventoried', 13,
                              "#64748b", anchor="middle"))
            parts.append(text(x + (cell_w - 40) / 2, box_y + 105,
                              f'baseline {record["baseline_event_ms_p50"]:.3f} ms', 12,
                              "#64748b", anchor="middle"))
    parts.append(text(width / 2, 595,
                      "8 / 8 capability failures · no Autograd route · ordinary GEMM fallback is not grouped support",
                      17, "#b42335", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def packed_weight_gradient_discard_svg() -> str:
    summary = json.loads((PACKED_WGRAD_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (PACKED_WGRAD_ROOT / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = summary["comparisons"]
    width, height = 1660, 720
    chart_x, chart_y, chart_w, chart_h = 130, 145, 980, 410
    panel_x, panel_y, panel_w, panel_h = 1200, 145, 340, 410
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 218 · Packed Weight-Gradient Discard", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "pack D2D + one ordinary GEMM · median of 3 fresh processes",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
        text(chart_x + chart_w / 2, 125, "Event speedup", 20,
             anchor="middle", weight=700),
        text(panel_x + panel_w / 2, 125, "Largest packed output", 20,
             anchor="middle", weight=700),
    ]

    def y(value: float) -> float:
        return chart_y + chart_h * (1.10 - value) / 0.35

    for tick in (0.75, 0.85, 0.95, 1.0, 1.05, 1.10):
        position = y(tick)
        color = "#2563eb" if tick == 1.0 else "#d97706" if tick == 1.05 else "#e5e9f0"
        parts.append(f'<line x1="{chart_x}" y1="{position:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{position:.1f}" stroke="{color}"/>')
        parts.append(text(chart_x - 12, position + 5, f"{tick:.2f}×", 13,
                          "#5b6474", anchor="end"))
    group_w = chart_w / len(comparisons)
    labels = []
    for index, row in enumerate(comparisons):
        center = chart_x + group_w * (index + 0.5)
        speedup = row["event_speedup_median"]
        top = y(speedup)
        base = y(0.75)
        parts.append(f'<rect x="{center-55:.1f}" y="{top:.1f}" width="110" '
                     f'height="{base-top:.1f}" fill="#d04a3a" rx="6"/>')
        parts.append(text(center, top - 12, f"{speedup:.3f}×", 16,
                          "#b42335", anchor="middle", weight=700))
        model = "Qwen" if row["model"] == "qwen" else "DeepSeek"
        projection = "QKV" if row["projection"] == "qkv" else "gate/up"
        label = f"{model} {projection}"
        labels.append(label)
        parts.append(text(center, chart_y + chart_h + 30, label, 14,
                          "#172033", anchor="middle", weight=700))
    largest = max(raw, key=lambda row: row["packed_output_bytes"])
    output_mib = largest["packed_output_bytes"] / (1024 ** 2)
    gradient_mib = largest["packed_gradient_bytes"] / (1024 ** 2)
    parts.append(f'<rect x="{panel_x+70}" y="{panel_y+80}" width="200" '
                 'height="250" fill="#fff1f2" stroke="#e11d48" rx="10"/>')
    parts.append(text(panel_x + panel_w / 2, panel_y + 145,
                      f"{output_mib:.0f} MiB", 34, "#be123c",
                      anchor="middle", weight=700))
    parts.append(text(panel_x + panel_w / 2, panel_y + 180,
                      "packed output", 15, "#5b6474", anchor="middle"))
    parts.append(text(panel_x + panel_w / 2, panel_y + 235,
                      f"+ {gradient_mib:.0f} MiB gradient pack", 15,
                      "#9a4f00", anchor="middle", weight=700))
    parts.append(text(panel_x + panel_w / 2, panel_y + 280,
                      "DeepSeek gate/up", 15, "#5b6474", anchor="middle"))
    parts.append(text(width / 2, 650,
                      "0 / 4 pass 1.05× · complete-output max error 1.15e-7 · no Autograd route",
                      17, "#b42335", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def fp32_weight_gradient_solutions_svg() -> str:
    operator = json.loads((FP32_WGRAD_SOLUTION_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    model = json.loads((FP32_WGRAD_SOLUTION_ROOT / "model-pilot" /
                        "summary.json").read_text(encoding="utf-8"))
    operator_rows = {row["model"]: row for row in operator["summaries"]}
    model_rows = {row["model"]: row for row in model["comparisons"]}
    width, height = 1600, 720
    chart_x, chart_y, chart_w, chart_h = 160, 150, 1280, 410
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 219 · FP32 Weight-Gradient Solutions", 30,
             anchor="middle", weight=700),
        text(width / 2, 80,
             "stable common operator index · exact registry hits · 3-process model gate",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]

    def y(value: float) -> float:
        return chart_y + chart_h * (1.16 - value) / 0.22

    for tick in (0.94, 0.98, 1.0, 1.05, 1.10, 1.15):
        position = y(tick)
        color = "#2563eb" if tick == 1.0 else "#d97706" if tick == 1.05 else "#e5e9f0"
        parts.append(f'<line x1="{chart_x}" y1="{position:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{position:.1f}" stroke="{color}"/>')
        parts.append(text(chart_x - 12, position + 5, f"{tick:.2f}×", 13,
                          "#5b6474", anchor="end"))
    models = (("qwen", "qwen2.5-0.5b", "Qwen 0.5B", 144),
              ("deepseek", "deepseek-r1-distill-qwen-1.5b", "DeepSeek 1.5B", 168))
    group_w = chart_w / len(models)
    for index, (short, full, label, hits) in enumerate(models):
        center = chart_x + group_w * (index + 0.5)
        values = ((operator_rows[short]["selected_median_speedup"],
                   "#18a558", "operator"),
                  (model_rows[full]["throughput_speedup"],
                   "#d04a3a", "end-to-end"))
        for offset, (value, color, name) in enumerate(values):
            x = center - 115 + offset * 125
            top = y(value)
            base = y(0.94)
            parts.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="100" '
                         f'height="{base-top:.1f}" fill="{color}" rx="6"/>')
            parts.append(text(x + 50, top - 10, f"{value:.3f}×", 16,
                              color, anchor="middle", weight=700))
            parts.append(text(x + 50, chart_y + chart_h + 25, name, 13,
                              "#5b6474", anchor="middle"))
        parts.append(text(center, chart_y + chart_h + 58, label, 16,
                          anchor="middle", weight=700))
        parts.append(text(center, chart_y + 45,
                          f'index {operator_rows[short]["selected_index"]} · {hits} exact hits',
                          13, "#5b6474", anchor="middle", weight=600))
    parts.append(text(width / 2, 650,
                      "Operator wins 1.077× / 1.133×, but model throughput is 0.993× / 0.996× · no default",
                      17, "#9a4f00", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def training_graph_capture_svg() -> str:
    summary = json.loads((TRAINING_GRAPH_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    rows = {(row["precision"], row["stage"]): row
            for row in summary["cases"]}
    width, height = 1600, 760
    stages = (("forward", "Forward"), ("backward", "Backward"),
              ("optimizer", "AdamW"), ("full-step", "Full step"))
    precisions = (("fp32", "FP32"), ("bf16", "BF16"))
    chart_x, chart_y, cell_w, cell_h = 300, 155, 260, 155
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 220 · Training HIP Graph Boundary", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "24 fresh processes · allocation-safe recovery · device nodes are not full training state",
             16, "#5b6474", anchor="middle"),
    ]
    for column, (_, label) in enumerate(stages):
        parts.append(text(chart_x + column * cell_w + cell_w / 2, 130,
                          label, 17, anchor="middle", weight=700))
    for row_index, (precision, label) in enumerate(precisions):
        y = chart_y + row_index * cell_h
        parts.append(text(chart_x - 28, y + 82, label, 18,
                          anchor="end", weight=700))
        for column, (stage, _) in enumerate(stages):
            record = rows[(precision, stage)]
            x = chart_x + column * cell_w
            supported = record["capture_supported"]
            fill = "#ecfdf3" if supported else "#fff1f2"
            stroke = "#16a34a" if supported else "#e11d48"
            headline = (f'{record["captured_nodes"]} nodes'
                        if supported else "blocked safely")
            detail = ("host step unchanged"
                      if stage == "optimizer" else "dynamic Storage")
            parts.append(f'<rect x="{x+8}" y="{y+8}" width="{cell_w-16}" '
                         f'height="{cell_h-16}" fill="{fill}" stroke="{stroke}" rx="10"/>')
            parts.append(text(x + cell_w / 2, y + 68, headline, 21, stroke,
                              anchor="middle", weight=700))
            parts.append(text(x + cell_w / 2, y + 101, detail, 14,
                              "#5b6474", anchor="middle"))
    panel_y = 505
    parts.append(f'<rect x="170" y="{panel_y}" width="1260" height="145" '
                 'fill="#fff7ed" stroke="#f59e0b" rx="12"/>')
    parts.append(text(210, panel_y + 40, "Two independent blockers", 18,
                      "#9a4f00", weight=700))
    parts.append(text(210, panel_y + 78,
                      "1. Forward/backward rebuild dynamic Tensor Storage; replay needs a graph-wide liveness plan and stable workspaces.",
                      15, "#5b6474"))
    parts.append(text(210, panel_y + 110,
                      "2. AdamW captures 21 device nodes, but replay cannot advance its CPU-owned step/bias-correction state.",
                      15, "#5b6474"))
    parts.append(text(width / 2, 705,
                      "Decision: keep the safety guard and probe; reject a complete-training Graph claim",
                      18, "#9a4f00", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def adamw_graph_replay_svg() -> str:
    summary = json.loads((ADAMW_GRAPH_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    rows = summary["comparisons"]
    width, height = 1600, 760
    chart_x, chart_y, chart_w, chart_h = 150, 145, 1300, 430
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 221 · Device-Owned AdamW Graph Step", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "60 fresh processes · 53 replayed steps · complete state samples aligned",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]

    def y(value: float) -> float:
        return chart_y + chart_h * (1.65 - value) / 1.55

    for tick in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        position = y(tick)
        color = "#2563eb" if tick == 1.0 else "#e5e9f0"
        parts.append(f'<line x1="{chart_x}" y1="{position:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{position:.1f}" stroke="{color}"/>')
        parts.append(text(chart_x - 12, position + 5, f"{tick:.2f}×", 13,
                          "#5b6474", anchor="end"))
    group_w = chart_w / 5
    case_order = ((1, 1024, "1×1K"), (16, 1024, "16×1K"),
                  (64, 1024, "64×1K"), (256, 1024, "256×1K"),
                  (16, 262144, "16×256K"))
    by_key = {(row["precision"], row["tensors"], row["elements"]): row
              for row in rows}
    for index, (tensors, elements, label) in enumerate(case_order):
        center = chart_x + group_w * (index + 0.5)
        for offset, (precision, color) in enumerate(
                (("fp32", "#16a34a"), ("bf16", "#f97316"))):
            value = by_key[(precision, tensors, elements)]["wall_speedup"]
            x = center - 80 + offset * 85
            top = y(value)
            base = y(0.1)
            parts.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="68" '
                         f'height="{base-top:.1f}" fill="{color}" rx="5"/>')
            parts.append(text(x + 34, top - 9, f"{value:.3f}×", 13,
                              color, anchor="middle", weight=700))
        parts.append(text(center, chart_y + chart_h + 28, label, 15,
                          anchor="middle", weight=700))
    parts.append(text(570, 635, "FP32", 15, "#16a34a", weight=700))
    parts.append(text(650, 635, "BF16 moment", 15, "#f97316", weight=700))
    parts.append(text(width / 2, 686,
                      "FP32 64/256 small tensors: 1.427×/1.436× · BF16 and large tensors regress · explicit only",
                      17, "#9a4f00", anchor="middle", weight=700))
    parts.append(text(width / 2, 720,
                      "maximum state error 7.45e-8 · no timed payload transfer · no universal route",
                      15, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def adamw_graph_multi_svg() -> str:
    summary = json.loads((ADAMW_GRAPH_MULTI_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    rows = summary["comparisons"]
    width, height = 1600, 780
    chart_x, chart_y, chart_w, chart_h = 150, 145, 1300, 450
    minimum, maximum = 0.125, 64.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 222 · Two-Node AdamW Multi-Tensor Graph", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "90 fresh processes · immutable descriptors prepared once · logarithmic speedup axis",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]

    def y(value: float) -> float:
        return chart_y + chart_h * (
            math.log2(maximum) - math.log2(value)) / (
                math.log2(maximum) - math.log2(minimum))

    for tick in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        position = y(tick)
        color = "#2563eb" if tick == 1.0 else "#e5e9f0"
        parts.append(f'<line x1="{chart_x}" y1="{position:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{position:.1f}" stroke="{color}"/>')
        parts.append(text(chart_x - 12, position + 5, f"{tick:g}×", 13,
                          "#5b6474", anchor="end"))
    case_order = ((1, 1024, "1×1K"), (16, 1024, "16×1K"),
                  (64, 1024, "64×1K"), (256, 1024, "256×1K"),
                  (16, 262144, "16×256K"))
    by_key = {(row["precision"], row["tensors"], row["elements"]): row
              for row in rows}
    group_w = chart_w / len(case_order)
    for index, (tensors, elements, label) in enumerate(case_order):
        center = chart_x + group_w * (index + 0.5)
        for offset, (precision, color) in enumerate(
                (("fp32", "#16a34a"), ("bf16", "#f97316"))):
            record = by_key[(precision, tensors, elements)]
            value = record["multi_wall_speedup"]
            x = center - 80 + offset * 85
            top = y(value)
            base = y(minimum)
            parts.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="68" '
                         f'height="{base-top:.1f}" fill="{color}" rx="5"/>')
            parts.append(text(x + 34, top - 8, f"{value:.2f}×", 13,
                              color, anchor="middle", weight=700))
            point = y(record["per_tensor_wall_speedup"])
            parts.append(f'<circle cx="{x+34:.1f}" cy="{point:.1f}" r="5" '
                         'fill="#172033"/>')
        parts.append(text(center, chart_y + chart_h + 28, label, 15,
                          anchor="middle", weight=700))
    parts.append(text(500, 655, "bars = two-node multi Graph", 14,
                      "#5b6474", weight=700))
    parts.append(text(760, 655, "black dot = per-tensor Graph", 14,
                      "#172033", weight=700))
    parts.append(text(width / 2, 704,
                      "BF16 64/256×1K rescued to 10.81×/36.93× · BF16 large 1.63×",
                      18, "#16a34a", anchor="middle", weight=700))
    parts.append(text(width / 2, 738,
                      "FP32 16×256K remains 0.908× · single Tensor remains slower · explicit candidate only",
                      16, "#9a4f00", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def gradient_address_stability_svg() -> str:
    summary = json.loads((GRADIENT_ADDRESS_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    rows = summary["comparisons"]
    width, height = 1600, 760
    chart_x, chart_y, chart_w, chart_h = 160, 155, 1280, 390
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 223 · Gradient Storage Address Stability", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "18 fresh processes · one allocator warmup · two measured backward passes",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]
    labels = []
    for row in rows:
        label = ("Tiny " + row["precision"].upper()
                 if row["model"] == "tiny" else
                 ("Qwen" if row["model"] == "qwen" else "DeepSeek") +
                 f' T{row["context"]}')
        labels.append(label)
    group_w = chart_w / len(rows)
    for index, row in enumerate(rows):
        total = row["stable_gradient_bytes"] + row["changed_gradient_bytes"]
        stable = row["stable_gradient_bytes"] / total if total else 0.0
        center = chart_x + group_w * (index + 0.5)
        bar_x, bar_y, bar_w, bar_h = center - 55, chart_y + 35, 110, 285
        changed_height = bar_h * (1.0 - stable)
        stable_height = bar_h - changed_height
        parts.append(f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_w}" '
                     f'height="{stable_height:.1f}" fill="#16a34a" rx="5"/>')
        if changed_height > 0:
            parts.append(f'<rect x="{bar_x:.1f}" y="{bar_y+stable_height:.1f}" '
                         f'width="{bar_w}" height="{changed_height:.1f}" '
                         'fill="#e11d48" rx="5"/>')
        parts.append(text(center, bar_y - 10,
                          f'{row["stable_gradient_tensors"]}/{row["parameter_tensors"]}',
                          15, "#172033", anchor="middle", weight=700))
        parts.append(text(center, chart_y + chart_h - 28, labels[index], 14,
                          anchor="middle", weight=700))
        changed_gib = row["changed_gradient_bytes"] / (1024 ** 3)
        parts.append(text(center, chart_y + chart_h - 6,
                          f'{changed_gib:.3f} GiB changed', 12,
                          "#b42335" if changed_gib else "#5b6474",
                          anchor="middle"))
    parts.append(text(580, 610, "green = stable bytes", 15, "#16a34a", weight=700))
    parts.append(text(800, 610, "red = changed bytes", 15, "#e11d48", weight=700))
    parts.append(text(width / 2, 665,
                      "Qwen T8/T512 and DeepSeek T8 stable · DeepSeek T512 changes 198 tensors / 7.108 GB",
                      18, "#9a4f00", anchor="middle", weight=700))
    parts.append(text(width / 2, 707,
                      "eligibility must be snapshot + context specific; no raw pointer values are exported",
                      15, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def optimizer_graph_model_preflight_svg() -> str:
    summary = json.loads((OPTIMIZER_GRAPH_PREFLIGHT_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    width, height = 1600, 720
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 224 · Model Optimizer Graph Preflight", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "12 fresh processes · graph-ready Stream lifecycle · safety gate before launch",
             16, "#5b6474", anchor="middle"),
    ]
    steps = (
        ("1", "Create non-default", "Graph Stream", "#fff7ed", "#f59e0b"),
        ("2", "Default-Stream", "pool disabled", "#fff1f2", "#e11d48"),
        ("3", "Next backward", "snapshot differs", "#fff1f2", "#e11d48"),
        ("4", "Safety gate", "rejects replay", "#ecfdf3", "#16a34a"),
        ("5", "Graph launches", "0 / 12", "#ecfdf3", "#16a34a"),
    )
    start_x, y, box_w, gap = 70, 165, 250, 55
    for index, (number, first, second, fill, stroke) in enumerate(steps):
        x = start_x + index * (box_w + gap)
        parts.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="150" '
                     f'fill="{fill}" stroke="{stroke}" rx="12"/>')
        parts.append(text(x + 24, y + 38, number, 22, stroke, weight=700))
        parts.append(text(x + box_w / 2, y + 78, first, 17,
                          anchor="middle", weight=700))
        parts.append(text(x + box_w / 2, y + 108, second, 17, stroke,
                          anchor="middle", weight=700))
        if index + 1 < len(steps):
            parts.append(f'<path d="M {x+box_w+8} {y+75} L {x+box_w+gap-8} {y+75}" '
                         'stroke="#64748b" stroke-width="3"/>')
            parts.append(f'<path d="M {x+box_w+gap-18} {y+68} L {x+box_w+gap-8} {y+75} '
                         f'L {x+box_w+gap-18} {y+82}" fill="none" '
                         'stroke="#64748b" stroke-width="3"/>')
    comparisons = summary["comparisons"]
    panel_y = 380
    parts.append(f'<rect x="170" y="{panel_y}" width="1260" height="175" '
                 'fill="#ffffff" stroke="#cbd3df" rx="12"/>')
    group_w = 1260 / len(comparisons)
    for index, row in enumerate(comparisons):
        center = 170 + group_w * (index + 0.5)
        label = ("Qwen" if row["model"] == "qwen" else "DeepSeek") + \
                f' T{row["context"]}'
        parts.append(text(center, panel_y + 42, label, 17,
                          anchor="middle", weight=700))
        parts.append(text(center, panel_y + 82, "3/3 rejected", 21,
                          "#e11d48", anchor="middle", weight=700))
        parts.append(text(center, panel_y + 116,
                          f'{row["preparation_ms_median"]:.2f} ms setup', 14,
                          "#5b6474", anchor="middle"))
        parts.append(text(center, panel_y + 145, "0 launches", 14,
                          "#16a34a", anchor="middle", weight=700))
    parts.append(text(width / 2, 625,
                      "Previous default-Stream stability does not survive Graph Stream creation",
                      18, "#9a4f00", anchor="middle", weight=700))
    parts.append(text(width / 2, 666,
                      "Decision: no optimizer-only model Graph until Stream-aware retirement or stable gradients",
                      16, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def quiescent_allocator_handoff_svg() -> str:
    summary = json.loads((QUIESCENT_HANDOFF_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    width, height = 1600, 740
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 225 · Quiescent Allocator Handoff", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "24 fresh processes · device-wide completion proof · three phase handoffs per run",
             16, "#5b6474", anchor="middle"),
    ]
    flow_y = 135
    flow = (("default backward", "pool enabled", "#ecfdf3", "#16a34a"),
            ("Graph submit", "pool disabled", "#fff1f2", "#e11d48"),
            ("device quiescent", "all Streams done", "#eff6ff", "#2563eb"),
            ("next backward", "pool enabled", "#ecfdf3", "#16a34a"))
    for index, (top, bottom, fill, stroke) in enumerate(flow):
        x = 150 + index * 350
        parts.append(f'<rect x="{x}" y="{flow_y}" width="270" height="105" '
                     f'fill="{fill}" stroke="{stroke}" rx="12"/>')
        parts.append(text(x + 135, flow_y + 43, top, 17,
                          anchor="middle", weight=700))
        parts.append(text(x + 135, flow_y + 75, bottom, 15, stroke,
                          anchor="middle", weight=700))
        if index < len(flow) - 1:
            parts.append(f'<path d="M {x+280} {flow_y+52} L {x+340} {flow_y+52}" '
                         'stroke="#64748b" stroke-width="3"/>')
    panel_x, panel_y, panel_w, panel_h = 150, 310, 1300, 255
    parts.append(f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" '
                 f'height="{panel_h}" fill="#ffffff" stroke="#cbd3df" rx="12"/>')
    comparisons = summary["comparisons"]
    group_w = panel_w / len(comparisons)
    for index, row in enumerate(comparisons):
        center = panel_x + group_w * (index + 0.5)
        label = ("Qwen" if row["model"] == "qwen" else "DeepSeek") + \
                f' T{row["context"]}'
        parts.append(text(center, panel_y + 40, label, 17,
                          anchor="middle", weight=700))
        parts.append(text(center - 65, panel_y + 93, "disabled", 13,
                          "#5b6474", anchor="middle"))
        parts.append(text(center - 65, panel_y + 137, "rejected", 18,
                          "#e11d48", anchor="middle", weight=700))
        rescued = row["rescued"]
        parts.append(text(center + 65, panel_y + 93, "handoff", 13,
                          "#5b6474", anchor="middle"))
        parts.append(text(center + 65, panel_y + 137,
                          "rescued" if rescued else "still rejected", 18,
                          "#16a34a" if rescued else "#e11d48",
                          anchor="middle", weight=700))
        parts.append(text(center, panel_y + 195,
                          f'{row["policies"]["handoff"]["handoff_count"]} handoffs/run',
                          14, "#2563eb", anchor="middle", weight=700))
        parts.append(text(center, panel_y + 224, "0 Graph launches", 13,
                          "#5b6474", anchor="middle"))
    parts.append(text(width / 2, 635,
                      "Qwen T8/T512 + DeepSeek T8 rescued · DeepSeek T512 remains a real allocator-order counterexample",
                      17, "#9a4f00", anchor="middle", weight=700))
    parts.append(text(width / 2, 680,
                      "keep explicit handoff; every later non-default submission disables reuse again",
                      15, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def optimizer_graph_model_gate_svg() -> str:
    summary = json.loads((OPTIMIZER_GRAPH_MODEL_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    rows = summary["comparisons"]
    width, height = 1600, 740
    chart_x, chart_y, chart_w, chart_h = 180, 150, 1240, 390
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 226 · Model Optimizer Graph Gate", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "21 fresh processes · exact loss/parameter · two nodes · handoff cost included in step",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]

    def y(value: float) -> float:
        return chart_y + chart_h * (1.10 - value) / 0.55

    for tick in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1):
        position = y(tick)
        color = "#2563eb" if tick == 1.0 else "#e5e9f0"
        parts.append(f'<line x1="{chart_x}" y1="{position:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{position:.1f}" stroke="{color}"/>')
        parts.append(text(chart_x - 12, position + 5, f"{tick:.1f}×", 13,
                          "#5b6474", anchor="end"))
    group_w = chart_w / len(rows)
    for index, row in enumerate(rows):
        center = chart_x + group_w * (index + 0.5)
        for offset, (field, color, label) in enumerate((
                ("optimizer_speedup", "#e11d48", "optimizer"),
                ("step_speedup", "#f59e0b", "full step"))):
            value = row[field]
            x = center - 100 + offset * 115
            top = y(value)
            base = y(0.55)
            parts.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="90" '
                         f'height="{base-top:.1f}" fill="{color}" rx="6"/>')
            parts.append(text(x + 45, top - 9, f"{value:.3f}×", 15,
                              color, anchor="middle", weight=700))
            parts.append(text(x + 45, chart_y + chart_h + 23, label, 13,
                              "#5b6474", anchor="middle"))
        model = "Qwen" if row["model"] == "qwen" else "DeepSeek"
        parts.append(text(center, chart_y + chart_h + 58,
                          f'{model} T{row["context"]}', 17,
                          anchor="middle", weight=700))
    parts.append(text(width / 2, 630,
                      "optimizer: 0.798× / 0.807× / 0.656× · metadata H2D removed but device work regresses",
                      18, "#b42335", anchor="middle", weight=700))
    parts.append(text(width / 2, 675,
                      "Qwen T8 full-step 1.050× is isolated noise/overlap; Qwen T512 and DeepSeek regress · no route",
                      15, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def rocwmma_qk_tile_svg() -> str:
    summary = json.loads((ROCWMMA_QK_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    rows = summary["comparisons"]
    sequences = summary["sequences"]
    width, height = 1600, 760
    chart_x, chart_y, chart_w, chart_h = 150, 150, 1300, 390
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 227 · rocWMMA QK Tile Boundary", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "48 fresh processes · complete BF16×BF16→FP32 outputs · tile32/wave1 after screening",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]

    def y(value: float) -> float:
        return chart_y + chart_h * (2.1 - value) / 1.6

    for tick in (0.5, 1.0, 1.5, 2.0):
        position = y(tick)
        color = "#2563eb" if tick == 1.0 else "#e5e9f0"
        parts.append(f'<line x1="{chart_x}" y1="{position:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{position:.1f}" stroke="{color}"/>')
        parts.append(text(chart_x - 12, position + 5, f"{tick:.1f}×", 13,
                          "#5b6474", anchor="end"))
    x_by_sequence = {
        sequence: chart_x + chart_w * (index + 0.5) / len(sequences)
        for index, sequence in enumerate(sequences)
    }
    for sequence, x_pos in x_by_sequence.items():
        parts.append(text(x_pos, chart_y + chart_h + 30, f"T{sequence}", 14,
                          "#5b6474", anchor="middle"))
    for inner, color in ((64, "#16a34a"), (128, "#e11d48")):
        selected = sorted((row for row in rows if row["inner"] == inner),
                          key=lambda row: row["sequence"])
        points = [(x_by_sequence[row["sequence"]],
                   y(row["rocwmma_over_hipblaslt"])) for row in selected]
        parts.append('<polyline fill="none" stroke="{}" stroke-width="4" points="{}"/>'.format(
            color, " ".join(f"{x:.1f},{point_y:.1f}" for x, point_y in points)))
        for row, (x_pos, y_pos) in zip(selected, points):
            parts.append(f'<circle cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="6" fill="{color}"/>')
            if row["sequence"] in (512, 2048):
                parts.append(text(x_pos, y_pos - 12,
                                  f'{row["rocwmma_over_hipblaslt"]:.3f}×',
                                  13, color, anchor="middle", weight=700))
        parts.append(text(chart_x + chart_w - 15,
                          chart_y + 32 + (0 if inner == 64 else 26),
                          f"D{inner}", 15, color, anchor="end", weight=700))
    parts.append(text(width / 2, 625,
                      "T512: 1.784× / 1.654× faster than default hipBLASLt · T2048 D128: only 0.688×",
                      18, "#9a4f00", anchor="middle", weight=700))
    parts.append(text(width / 2, 670,
                      "Admit an online-Attention prototype; do not route models until causal/GQA/tail/memory gates pass",
                      15, "#5b6474", anchor="middle"))
    parts.append(text(width / 2, 710,
                      "Vertical axis: rocWMMA speed ÷ hipBLASLt speed; 1.0× is parity",
                      13, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def rocwmma_online_attention_svg() -> str:
    summary = json.loads((ROCWMMA_ONLINE_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    rows = summary["comparisons"]
    sequences = summary["sequences"]
    width, height = 1600, 760
    chart_x, chart_y, chart_w, chart_h = 150, 150, 1300, 390
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 228 · Online rocWMMA Attention", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "42 fresh processes · causal GQA · MFMA QK/PV · no global score tensor",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]

    def y(value: float) -> float:
        return chart_y + chart_h * (4.2 - value) / 3.4

    for tick in (1.0, 2.0, 3.0, 4.0):
        position = y(tick)
        color = "#2563eb" if tick == 1.0 else "#e5e9f0"
        parts.append(f'<line x1="{chart_x}" y1="{position:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{position:.1f}" stroke="{color}"/>')
        parts.append(text(chart_x - 12, position + 5, f"{tick:.1f}×", 13,
                          "#5b6474", anchor="end"))
    x_by_sequence = {
        sequence: chart_x + chart_w * (index + 0.5) / len(sequences)
        for index, sequence in enumerate(sequences)
    }
    for sequence, x_pos in x_by_sequence.items():
        parts.append(text(x_pos, chart_y + chart_h + 30, f"T{sequence}", 14,
                          "#5b6474", anchor="middle"))
    for family, color, label in (("qwen", "#16a34a", "Qwen H14/KV2/D64"),
                                 ("deepseek", "#e11d48", "Deep H12/KV2/D128")):
        selected = sorted((row for row in rows if row["family"] == family),
                          key=lambda row: row["sequence"])
        points = [(x_by_sequence[row["sequence"]],
                   y(row["online_over_current"])) for row in selected]
        parts.append('<polyline fill="none" stroke="{}" stroke-width="4" points="{}"/>'.format(
            color, " ".join(f"{x:.1f},{point_y:.1f}" for x, point_y in points)))
        for row, (x_pos, y_pos) in zip(selected, points):
            parts.append(f'<circle cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="6" fill="{color}"/>')
            if row["sequence"] in (512, 2048):
                parts.append(text(x_pos, y_pos - 12,
                                  f'{row["online_over_current"]:.3f}×',
                                  13, color, anchor="middle", weight=700))
        parts.append(text(chart_x + chart_w - 15,
                          chart_y + 30 + (0 if family == "qwen" else 26),
                          label, 15, color, anchor="end", weight=700))
    parts.append(text(width / 2, 625,
                      "candidate/current: every shape ≥1.260× · T2048 score removed: 224 MiB / 192 MiB",
                      18, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 670,
                      "Short scalar kernels remain faster; admit operator integration with fallback, not a model route",
                      15, "#9a4f00", anchor="middle"))
    parts.append(text(width / 2, 710,
                      "Vertical axis: online rocWMMA Event speed ÷ current framework Attention Event speed",
                      13, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def rocwmma_online_operator_svg() -> str:
    summary = json.loads((ROCWMMA_OPERATOR_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    rows = summary["comparisons"]
    width, height = 1700, 790
    chart_x, chart_y, chart_w, chart_h = 120, 150, 1460, 410
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 229 · Public Online-Attention Operator", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "42 fresh processes · B1/B2 native routes · T31/T33/D32 explicit fallbacks",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]

    def y(value: float) -> float:
        return chart_y + chart_h * (2.7 - value) / 2.2

    for tick in (0.5, 1.0, 1.5, 2.0, 2.5):
        position = y(tick)
        color = "#2563eb" if tick == 1.0 else "#e5e9f0"
        parts.append(f'<line x1="{chart_x}" y1="{position:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{position:.1f}" stroke="{color}"/>')
        parts.append(text(chart_x - 10, position + 5, f"{tick:.1f}×", 13,
                          "#5b6474", anchor="end"))
    group_width = chart_w / len(rows)
    for index, row in enumerate(rows):
        center = chart_x + group_width * (index + 0.5)
        value = row["candidate_over_current"]
        color = "#16a34a" if row["native"] else "#f59e0b"
        base = y(0.5)
        top = y(value)
        parts.append(f'<rect x="{center-27:.1f}" y="{top:.1f}" width="54" '
                     f'height="{base-top:.1f}" fill="{color}" rx="5"/>')
        parts.append(text(center, top - 8, f"{value:.2f}×", 12, color,
                          anchor="middle", weight=700))
        label = row["case"].replace("qwen-", "Q-").replace("deep-", "D-")
        parts.append(text(center, chart_y + chart_h + 18, label, 11,
                          "#5b6474", anchor="end", rotate=-48))
    parts.append(text(width / 2, 665,
                      "10/10 native cases ≥1.534× · exact routing counters · timed payload transfer = 0",
                      18, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 710,
                      "Fallbacks stay exact but only 0.607×–0.696× because BF16→FP32 casts are explicit",
                      16, "#9a4f00", anchor="middle", weight=700))
    parts.append(text(width / 2, 752,
                      "Green = native rocWMMA · orange = fallback · vertical axis is candidate/current Event speed",
                      13, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def rocwmma_online_model_svg() -> str:
    summary = json.loads((ROCWMMA_MODEL_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    rows = summary["comparisons"]
    width, height = 1600, 760
    chart_x, chart_y, chart_w, chart_h = 150, 150, 1300, 390
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 230 · Full-Model Online Attention Gate", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "36 fresh processes · full 151936 logits · 2 warm-up + 5 measured prefills",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]

    def y(value: float) -> float:
        return chart_y + chart_h * (1.05 - value) / 0.35

    for tick in (0.7, 0.8, 0.9, 1.0):
        position = y(tick)
        color = "#2563eb" if tick == 1.0 else "#e5e9f0"
        parts.append(f'<line x1="{chart_x}" y1="{position:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{position:.1f}" stroke="{color}"/>')
        parts.append(text(chart_x - 12, position + 5, f"{tick:.1f}×", 13,
                          "#5b6474", anchor="end"))
    group_width = chart_w / len(rows)
    for index, row in enumerate(rows):
        center = chart_x + group_width * (index + 0.5)
        value = row["speedup"]
        color = "#16a34a" if row["model"].startswith("qwen") else "#e11d48"
        top = y(value)
        base = y(0.7)
        parts.append(f'<rect x="{center-55:.1f}" y="{top:.1f}" width="110" '
                     f'height="{base-top:.1f}" fill="{color}" rx="6"/>')
        parts.append(text(center, top - 10, f"{value:.3f}×", 15, color,
                          anchor="middle", weight=700))
        family = "Qwen" if row["model"].startswith("qwen") else "Deep"
        parts.append(text(center, chart_y + chart_h + 28,
                          f'{family} {row["case"]}', 14,
                          "#5b6474", anchor="middle"))
        parts.append(text(center, chart_y + chart_h + 52,
                          f'-{row["peak_bytes_saved"] / 1048576:.1f} MiB', 12,
                          "#166534", anchor="middle"))
    parts.append(text(width / 2, 630,
                      "Every model case regresses to 0.761×–0.884× despite saving 3.5–57.0 MiB",
                      18, "#b42335", anchor="middle", weight=700))
    parts.append(text(width / 2, 675,
                      "Qwen logits reach Max/RMS 0.511/0.112 · top tokens stay equal · model route rejected",
                      16, "#9a4f00", anchor="middle"))
    parts.append(text(width / 2, 716,
                      "Operator remains public; bars are online/current prefill throughput",
                      13, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def rocwmma_direct_bf16_model_svg() -> str:
    before = json.loads((ROCWMMA_MODEL_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    after = json.loads((ROCWMMA_DIRECT_MODEL_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    before_rows = {(row["model"], row["case"]): row for row in before["comparisons"]}
    rows = after["comparisons"]
    width, height = 1600, 760
    chart_x, chart_y, chart_w, chart_h = 150, 150, 1300, 390
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 231 · Direct-BF16 Model Rebuttal", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "RoPE writes BF16 · grouped QKV retains BF16 V · zero Attention-core casts",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]

    def y(value: float) -> float:
        return chart_y + chart_h * (1.05 - value) / 0.35

    for tick in (0.7, 0.8, 0.9, 1.0):
        position = y(tick)
        color = "#2563eb" if tick == 1.0 else "#e5e9f0"
        parts.append(f'<line x1="{chart_x}" y1="{position:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{position:.1f}" stroke="{color}"/>')
        parts.append(text(chart_x - 12, position + 5, f"{tick:.1f}×", 13,
                          "#5b6474", anchor="end"))
    group_width = chart_w / len(rows)
    for index, row in enumerate(rows):
        center = chart_x + group_width * (index + 0.5)
        old = before_rows[(row["model"], row["case"])]["speedup"]
        new = row["speedup"]
        for offset, (value, color, label) in enumerate((
                (old, "#94a3b8", "3 casts"),
                (new, "#f59e0b", "direct"))):
            x_pos = center - 62 + offset * 66
            top = y(value)
            base = y(0.7)
            parts.append(f'<rect x="{x_pos:.1f}" y="{top:.1f}" width="56" '
                         f'height="{base-top:.1f}" fill="{color}" rx="5"/>')
            parts.append(text(x_pos + 28, top - 8, f"{value:.3f}×", 12,
                              color, anchor="middle", weight=700))
            parts.append(text(x_pos + 28, chart_y + chart_h + 18, label, 11,
                              "#5b6474", anchor="middle"))
        family = "Qwen" if row["model"].startswith("qwen") else "Deep"
        parts.append(text(center, chart_y + chart_h + 48,
                          f'{family} {row["case"]}', 14,
                          "#5b6474", anchor="middle"))
    parts.append(text(width / 2, 635,
                      "Every case improves slightly, but direct BF16 still reaches only 0.777×–0.906×",
                      18, "#b42335", anchor="middle", weight=700))
    parts.append(text(width / 2, 680,
                      "Qwen Max/RMS remains 0.485/0.110 · cast-removal hypothesis rejected",
                      16, "#9a4f00", anchor="middle"))
    parts.append(text(width / 2, 720,
                      "Gray = prior three-cast route · orange = direct BF16 boundary",
                      13, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def current_inference_profile_svg() -> str:
    summary = json.loads((CURRENT_INFERENCE_PROFILE_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    models = summary["models"]
    width, height = 1500, 760
    chart_x, chart_y, chart_w, chart_h = 170, 150, 1160, 410
    palette = {
        "hipBLASLt GEMM": "#2563eb", "softmax": "#e11d48",
        "other kernels": "#94a3b8", "FP32/BF16 cast": "#f59e0b",
        "RMSNorm forward/backward": "#16a34a", "GQA repeat": "#8b5cf6",
        "gradient/elementwise add": "#64748b",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 232 · Current T1024 Inference Profile", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "4 rocprof processes · (6-step − 1-step) / 5 · retained default path",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]
    for tick in (0, 20, 40, 60, 80, 100):
        y_pos = chart_y + chart_h * (100 - tick) / 100
        parts.append(f'<line x1="{chart_x}" y1="{y_pos:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{y_pos:.1f}" stroke="#e5e9f0"/>')
        parts.append(text(chart_x - 12, y_pos + 5, f"{tick}%", 13,
                          "#5b6474", anchor="end"))
    centers = (chart_x + chart_w * 0.3, chart_x + chart_w * 0.7)
    for model, center in zip(models, centers):
        bottom = chart_y + chart_h
        by_category = {row["category"]: row for row in model["categories"]}
        for category in palette:
            row = by_category.get(category)
            if row is None:
                continue
            height_value = chart_h * row["kernel_share"]
            bottom -= height_value
            parts.append(f'<rect x="{center-110:.1f}" y="{bottom:.1f}" '
                         f'width="220" height="{height_value:.1f}" '
                         f'fill="{palette[category]}"/>')
            if row["kernel_share"] >= 0.04:
                parts.append(text(center, bottom + height_value / 2 + 5,
                                  f'{category} {row["kernel_share"]*100:.1f}%',
                                  13, "#ffffff", anchor="middle", weight=700))
        label = "Qwen2.5-0.5B" if model["model"].startswith("qwen") else "DeepSeek-Distill-1.5B"
        parts.append(text(center, chart_y + chart_h + 36, label, 17,
                          anchor="middle", weight=700))
        parts.append(text(center, chart_y + chart_h + 62,
                          f'{model["total_kernel_ns_per_step"]/1e6:.3f} ms Kernel',
                          14, "#5b6474", anchor="middle"))
    parts.append(text(width / 2, 650,
                      "GEMM remains 59.7% / 66.8%; softmax is 14.8% / 9.2% but its local thread track is saturated",
                      17, "#172033", anchor="middle", weight=700))
    parts.append(text(width / 2, 695,
                      "Next bounded experiment: exact T1024 QK/PV hipBLASLt solution screening",
                      15, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def fp32_attention_t1024_svg() -> str:
    operator = json.loads((FP32_ATTENTION_T1024_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    model = json.loads((FP32_ATTENTION_T1024_MODEL_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    operator_rows = operator["comparisons"]
    model_rows = model["comparisons"]
    width, height = 1600, 760
    chart_x, chart_y, chart_w, chart_h = 160, 150, 1280, 390
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 233 · T1024 Exact Attention Solutions", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "12 operator processes + PV descriptor counterexample + 12 full-model processes",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]

    def y(value: float) -> float:
        return chart_y + chart_h * (1.6 - value) / 0.65

    for tick in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
        y_pos = y(tick)
        color = "#2563eb" if tick == 1.0 else "#e5e9f0"
        parts.append(f'<line x1="{chart_x}" y1="{y_pos:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{y_pos:.1f}" stroke="{color}"/>')
        parts.append(text(chart_x - 12, y_pos + 5, f"{tick:.1f}×", 13,
                          "#5b6474", anchor="end"))
    items = []
    for row in operator_rows:
        items.append((f'{row["model"]} {row["operation"].upper()}',
                      row["recommended_event_speedup"], "#16a34a", "operator"))
    for row in model_rows:
        label = "Qwen model" if row["model"].startswith("qwen") else "Deep model"
        items.append((label, row["candidate_speedup"], "#e11d48", "model"))
    group_width = chart_w / len(items)
    for index, (label, value, color, kind) in enumerate(items):
        center = chart_x + group_width * (index + 0.5)
        top = y(value)
        base = y(0.95)
        parts.append(f'<rect x="{center-55:.1f}" y="{top:.1f}" width="110" '
                     f'height="{base-top:.1f}" fill="{color}" rx="6"/>')
        parts.append(text(center, top - 10, f"{value:.3f}×", 14, color,
                          anchor="middle", weight=700))
        parts.append(text(center, chart_y + chart_h + 28, label, 13,
                          "#5b6474", anchor="middle"))
        parts.append(text(center, chart_y + chart_h + 49, kind, 11,
                          "#5b6474", anchor="middle"))
    parts.append(text(width / 2, 630,
                      "PV indices do not match interleaved BTHD descriptors: 175 misses, 0 dispatch",
                      17, "#9a4f00", anchor="middle", weight=700))
    parts.append(text(width / 2, 674,
                      "Qwen QK model logits Max/RMS 0.0733/0.0157; DeepSeek model only 1.002×",
                      17, "#b42335", anchor="middle", weight=700))
    parts.append(text(width / 2, 716,
                      "All local winners retained as evidence; no default solution policy",
                      13, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_swiglu_vector_svg() -> str:
    operator = json.loads((BF16_SWIGLU_VECTOR_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    model = json.loads((BF16_SWIGLU_VECTOR_MODEL_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    values = [
        ("Qwen operator", operator["comparisons"][0]["speedup"], "#16a34a"),
        ("Deep operator", operator["comparisons"][1]["speedup"], "#16a34a"),
        ("Qwen model", model["comparisons"][0]["candidate_speedup"], "#e11d48"),
        ("Deep model", model["comparisons"][1]["candidate_speedup"], "#e11d48"),
    ]
    width, height = 1500, 720
    chart_x, chart_y, chart_w, chart_h = 180, 150, 1140, 350
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 234 · BF16 SwiGLU: Operator vs Model", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "12 operator processes + 12 full-model processes · complete outputs checked",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]
    def y(value: float) -> float:
        return chart_y + chart_h * (1.30 - value) / 0.32
    for tick in (1.0, 1.05, 1.10, 1.15, 1.20, 1.25):
        y_pos = y(tick)
        color = "#2563eb" if tick == 1.0 else "#e5e9f0"
        parts.append(f'<line x1="{chart_x}" y1="{y_pos:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{y_pos:.1f}" stroke="{color}"/>')
        parts.append(text(chart_x - 12, y_pos + 5, f"{tick:.2f}×", 13,
                          "#5b6474", anchor="end"))
    slot = chart_w / len(values)
    for index, (label, value, color) in enumerate(values):
        center = chart_x + slot * (index + 0.5)
        top, base = y(value), y(0.98)
        parts.append(f'<rect x="{center-65:.1f}" y="{top:.1f}" width="130" '
                     f'height="{base-top:.1f}" fill="{color}" rx="6"/>')
        parts.append(text(center, top - 10, f"{value:.3f}×", 15, color,
                          anchor="middle", weight=700))
        parts.append(text(center, chart_y + chart_h + 30, label, 14,
                          "#5b6474", anchor="middle"))
    parts.append(text(width / 2, 610,
                      "Operator: 1.249× / 1.190×, bit-identical",
                      18, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 652,
                      "Model: 1.007× / 1.001×; DeepSeek fails the 1.005× gate",
                      18, "#b42335", anchor="middle", weight=700))
    parts.append(text(width / 2, 690,
                      "Explicit vector operator retained; Auto remains scalar",
                      13, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_grouped_swish_svg() -> str:
    operator = json.loads((BF16_GROUPED_SWISH_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    model = json.loads((BF16_GROUPED_SWISH_MODEL_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    values = [
        ("Qwen operator", operator["comparisons"][0][
            "user_arguments_event_speedup_median"], "#16a34a"),
        ("Deep operator", operator["comparisons"][1][
            "user_arguments_event_speedup_median"], "#16a34a"),
        ("Qwen model", model["comparisons"][0]["candidate_speedup"], "#e11d48"),
        ("Deep model", model["comparisons"][1]["candidate_speedup"], "#e11d48"),
    ]
    width, height = 1500, 720
    chart_x, chart_y, chart_w, chart_h = 180, 150, 1140, 350
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 235 · Grouped Swish Epilogue", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "6 capability processes + 12 full-model processes · same binary A/B",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]
    def y(value: float) -> float:
        return chart_y + chart_h * (1.13 - value) / 0.16
    for tick in (0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12):
        y_pos = y(tick)
        color = "#2563eb" if tick == 1.0 else "#e5e9f0"
        parts.append(f'<line x1="{chart_x}" y1="{y_pos:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{y_pos:.1f}" stroke="{color}"/>')
        parts.append(text(chart_x - 12, y_pos + 5, f"{tick:.2f}×", 13,
                          "#5b6474", anchor="end"))
    slot = chart_w / len(values)
    for index, (label, value, color) in enumerate(values):
        center = chart_x + slot * (index + 0.5)
        top, base = y(value), y(0.97)
        parts.append(f'<rect x="{center-65:.1f}" y="{top:.1f}" width="130" '
                     f'height="{base-top:.1f}" fill="{color}" rx="6"/>')
        parts.append(text(center, top - 10, f"{value:.3f}×", 15, color,
                          anchor="middle", weight=700))
        parts.append(text(center, chart_y + chart_h + 30, label, 14,
                          "#5b6474", anchor="middle"))
    parts.append(text(width / 2, 610,
                      "Pointer-stable operator: 1.097× / 1.069×; 64/64 candidates pass",
                      18, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 652,
                      "Model: 1.000× / 0.991×; logits Max 0.0973 / 0.0362",
                      18, "#b42335", anchor="middle", weight=700))
    parts.append(text(width / 2, 690,
                      "Explicit research switch retained default-off; model track closed",
                      13, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_rms_norm_output_svg() -> str:
    summary = json.loads((BF16_RMS_NORM_OUTPUT_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    values = []
    for row in summary["comparisons"]:
        label = "Qwen" if row["model"] == "qwen" else "DeepSeek"
        values.extend([
            (f"{label} Event", row["event_speedup_median"], "#16a34a"),
            (f"{label} wall", row["wall_speedup_median"], "#2563eb"),
        ])
    width, height = 1500, 700
    chart_x, chart_y, chart_w, chart_h = 180, 140, 1140, 360
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 236 · Direct BF16 RMSNorm Output", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "6 processes · caller Storage · 3 warm-ups + 30 Event measurements",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]
    def y(value: float) -> float:
        return chart_y + chart_h * (2.2 - value) / 1.25
    for tick in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2):
        y_pos = y(tick)
        color = "#e11d48" if tick == 1.0 else "#e5e9f0"
        parts.append(f'<line x1="{chart_x}" y1="{y_pos:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{y_pos:.1f}" stroke="{color}"/>')
        parts.append(text(chart_x - 12, y_pos + 5, f"{tick:.1f}×", 13,
                          "#5b6474", anchor="end"))
    slot = chart_w / len(values)
    for index, (label, value, color) in enumerate(values):
        center = chart_x + slot * (index + 0.5)
        top, base = y(value), y(0.95)
        parts.append(f'<rect x="{center-65:.1f}" y="{top:.1f}" width="130" '
                     f'height="{base-top:.1f}" fill="{color}" rx="6"/>')
        parts.append(text(center, top - 10, f"{value:.3f}×", 15, color,
                          anchor="middle", weight=700))
        parts.append(text(center, chart_y + chart_h + 30, label, 14,
                          "#5b6474", anchor="middle"))
    parts.append(text(width / 2, 610,
                      "Complete BF16 output bit-identical · timed payload transfers = 0",
                      18, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 654,
                      "Operator admitted; model route remains disabled until a separate gate",
                      14, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_ffn_norm_model_svg() -> str:
    summary = json.loads((BF16_FFN_NORM_MODEL_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    rows = summary["comparisons"]
    width, height = 1500, 700
    chart_x, chart_y, chart_w, chart_h = 210, 155, 1080, 320
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 237 · BF16 FFN Norm Model Gate", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "12 same-binary processes · B1T1024 · complete vocab logits",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]
    def y(value: float) -> float:
        return chart_y + chart_h * (1.016 - value) / 0.018
    for tick in (1.0, 1.005, 1.01, 1.015):
        y_pos = y(tick)
        color = "#e11d48" if tick == 1.0 else "#e5e9f0"
        parts.append(f'<line x1="{chart_x}" y1="{y_pos:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{y_pos:.1f}" stroke="{color}"/>')
        parts.append(text(chart_x - 12, y_pos + 5, f"{tick:.3f}×", 13,
                          "#5b6474", anchor="end"))
    for index, row in enumerate(rows):
        center = chart_x + chart_w * (index + 0.5) / 2
        value = row["candidate_speedup"]
        top, base = y(value), y(0.998)
        parts.append(f'<rect x="{center-110:.1f}" y="{top:.1f}" width="220" '
                     f'height="{base-top:.1f}" fill="#16a34a" rx="8"/>')
        label = "Qwen" if row["model"].startswith("qwen") else "DeepSeek"
        parts.append(text(center, top - 12, f"{value:.4f}×", 18, "#166534",
                          anchor="middle", weight=700))
        parts.append(text(center, chart_y + chart_h + 32, label, 16,
                          "#5b6474", anchor="middle"))
    qwen_reduction = rows[0]["baseline_engine_allocation_calls"] - \
        rows[0]["candidate_engine_allocation_calls"]
    deep_reduction = rows[1]["baseline_engine_allocation_calls"] - \
        rows[1]["candidate_engine_allocation_calls"]
    parts.append(text(width / 2, 565,
                      f"Measured allocations: -{qwen_reduction} / -{deep_reduction}; peak unchanged",
                      18, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 612,
                      "Complete logits Max/RMS = 0 / 0 for both models",
                      18, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 658,
                      "BF16 FFN Arena now enables the route by default; explicit false remains",
                      14, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def post_bf16_ffn_norm_profile_svg() -> str:
    verification = json.loads((POST_BF16_FFN_NORM_PROFILE_ROOT /
                               "verification.json").read_text(encoding="utf-8"))
    width, height = 1500, 720
    chart_x, chart_y, chart_w, chart_h = 180, 150, 1140, 330
    values = [
        ("Qwen before", verification["qwen_previous_total_kernel_ms"], "#94a3b8"),
        ("Qwen after", verification["qwen_total_kernel_ms"], "#16a34a"),
        ("Deep before", verification["deepseek_previous_total_kernel_ms"], "#94a3b8"),
        ("Deep after", verification["deepseek_total_kernel_ms"], "#16a34a"),
    ]
    maximum = 16.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 238 · Profile After FFN Norm Fusion", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "load-subtracted B1T1024 · four rocprof processes · milliseconds/forward",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]
    def y(value: float) -> float:
        return chart_y + chart_h * (maximum - value) / maximum
    for tick in (0, 4, 8, 12, 16):
        y_pos = y(float(tick))
        parts.append(f'<line x1="{chart_x}" y1="{y_pos:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{y_pos:.1f}" stroke="#e5e9f0"/>')
        parts.append(text(chart_x - 12, y_pos + 5, f"{tick} ms", 13,
                          "#5b6474", anchor="end"))
    slot = chart_w / len(values)
    for index, (label, value, color) in enumerate(values):
        center = chart_x + slot * (index + 0.5)
        top, base = y(value), y(0.0)
        parts.append(f'<rect x="{center-70:.1f}" y="{top:.1f}" width="140" '
                     f'height="{base-top:.1f}" fill="{color}" rx="6"/>')
        parts.append(text(center, top - 10, f"{value:.3f}", 15, color,
                          anchor="middle", weight=700))
        parts.append(text(center, chart_y + chart_h + 28, label, 14,
                          "#5b6474", anchor="middle"))
    parts.append(text(width / 2, 590,
                      "FP32/BF16 casts per step: 96→72 (Qwen), 112→84 (DeepSeek)",
                      18, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 635,
                      "GEMM share is now 60.9% / 68.2%; next boundary is Attention input",
                      18, "#5b6474", anchor="middle", weight=700))
    parts.append(text(width / 2, 680,
                      "Profile selects the next experiment; it does not itself enable another route",
                      13, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_attention_norm_model_svg() -> str:
    summary = json.loads((BF16_ATTENTION_NORM_MODEL_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    rows = summary["comparisons"]
    width, height = 1500, 700
    chart_x, chart_y, chart_w, chart_h = 210, 155, 1080, 320
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 239 · BF16 Attention Norm Model Gate", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "12 same-binary processes · FFN Norm retained on both sides · B1T1024",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]
    def y(value: float) -> float:
        return chart_y + chart_h * (1.016 - value) / 0.018
    for tick in (1.0, 1.005, 1.01, 1.015):
        y_pos = y(tick)
        color = "#e11d48" if tick == 1.0 else "#e5e9f0"
        parts.append(f'<line x1="{chart_x}" y1="{y_pos:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{y_pos:.1f}" stroke="{color}"/>')
        parts.append(text(chart_x - 12, y_pos + 5, f"{tick:.3f}×", 13,
                          "#5b6474", anchor="end"))
    for index, row in enumerate(rows):
        center = chart_x + chart_w * (index + 0.5) / 2
        value = row["candidate_speedup"]
        top, base = y(value), y(0.998)
        parts.append(f'<rect x="{center-110:.1f}" y="{top:.1f}" width="220" '
                     f'height="{base-top:.1f}" fill="#16a34a" rx="8"/>')
        label = "Qwen" if row["model"].startswith("qwen") else "DeepSeek"
        parts.append(text(center, top - 12, f"{value:.4f}×", 18, "#166534",
                          anchor="middle", weight=700))
        parts.append(text(center, chart_y + chart_h + 32, label, 16,
                          "#5b6474", anchor="middle"))
    qwen_peak = rows[0]["baseline_engine_peak_bytes"] - rows[0]["candidate_engine_peak_bytes"]
    deep_peak = rows[1]["baseline_engine_peak_bytes"] - rows[1]["candidate_engine_peak_bytes"]
    parts.append(text(width / 2, 565,
                      f"Peak bytes: -{qwen_peak} / -{deep_peak}; allocations: -120 / -140",
                      18, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 612,
                      "Complete logits Max/RMS = 0 / 0 for both models",
                      18, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 658,
                      "BF16 QKV Arena enables the route by default; explicit false remains",
                      14, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def post_bf16_attention_norm_profile_svg() -> str:
    data = json.loads((POST_BF16_ATTENTION_NORM_PROFILE_ROOT /
                       "verification.json").read_text(encoding="utf-8"))
    width, height = 1500, 700
    chart_x, chart_y, chart_w, chart_h = 180, 145, 1140, 340
    values = [
        ("Qwen prior", data["qwen_previous_total_kernel_ms"], "#94a3b8"),
        ("Qwen both", data["qwen_total_kernel_ms"], "#16a34a"),
        ("Deep prior", data["deepseek_previous_total_kernel_ms"], "#94a3b8"),
        ("Deep both", data["deepseek_total_kernel_ms"], "#16a34a"),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 240 · Profile After Both Norm Fusions", 30,
             anchor="middle", weight=700),
        text(width / 2, 82, "B1T1024 · four load-subtracted rocprof processes", 16,
             "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]
    def y(value: float) -> float:
        return chart_y + chart_h * (16.0 - value) / 16.0
    for tick in (0, 4, 8, 12, 16):
        y_pos = y(float(tick))
        parts.append(f'<line x1="{chart_x}" y1="{y_pos:.1f}" '
                     f'x2="{chart_x+chart_w}" y2="{y_pos:.1f}" stroke="#e5e9f0"/>')
        parts.append(text(chart_x - 12, y_pos + 5, f"{tick} ms", 13,
                          "#5b6474", anchor="end"))
    slot = chart_w / len(values)
    for index, (label, value, color) in enumerate(values):
        center = chart_x + slot * (index + 0.5)
        top, base = y(value), y(0.0)
        parts.append(f'<rect x="{center-70:.1f}" y="{top:.1f}" width="140" '
                     f'height="{base-top:.1f}" fill="{color}" rx="6"/>')
        parts.append(text(center, top - 10, f"{value:.3f}", 15, color,
                          anchor="middle", weight=700))
        parts.append(text(center, chart_y + chart_h + 28, label, 14,
                          "#5b6474", anchor="middle"))
    parts.append(text(width / 2, 585,
                      "Cast calls: 72→48 (Qwen), 84→56 (DeepSeek)",
                      18, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 628,
                      "Remaining per layer: one FP32→BF16 and one BF16→FP32 boundary",
                      18, "#5b6474", anchor="middle", weight=700))
    parts.append(text(width / 2, 670,
                      "Attribute both boundaries before the next implementation",
                      14, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_pv_output_svg() -> str:
    summary = json.loads((BF16_PV_OUTPUT_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    width, height = 1500, 650
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 241 · Direct BF16 P×V Output Capability", 30,
             anchor="middle", weight=700),
        text(width / 2, 82, "FP32 P/V + FP32 compute · only D changes to BF16", 16,
             "#5b6474", anchor="middle"),
    ]
    labels = ["Interleaved BTHD", "Zero-stride GQA BTHD"]
    for index, label in enumerate(labels):
        x = 170 + index * 610
        parts.extend([
            f'<rect x="{x}" y="170" width="500" height="240" rx="14" '
            'fill="#fff1f2" stroke="#e11d48" stroke-width="3"/>',
            text(x + 250, 225, label, 22, "#9f1239", anchor="middle", weight=700),
            text(x + 250, 285, "hipBLASLt status 6", 30, "#e11d48",
                 anchor="middle", weight=700),
            text(x + 250, 335, "timing not started", 17, "#5b6474", anchor="middle"),
            text(x + 250, 375, "retained FP32 path passed", 17, "#166534", anchor="middle"),
        ])
    parts.append(text(width / 2, 500,
                      f"Supported {summary['supported_cases']} / {summary['raw_cases']} · timed 0 · model route 0",
                      20, "#b42335", anchor="middle", weight=700))
    parts.append(text(width / 2, 555,
                      "Candidate APIs removed; direct mixed-output route closed on this backend",
                      18, "#5b6474", anchor="middle", weight=700))
    parts.append(text(width / 2, 610,
                      "Future work requires a different consumer/kernel, not another solution index",
                      14, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_value_pv_svg() -> str:
    width, height = 1500, 620
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 242 · BF16 V Input for P×V", 30,
             anchor="middle", weight=700),
        text(width / 2, 82, "FP32 probabilities + BF16 V + FP32 compute/output", 16,
             "#5b6474", anchor="middle"),
    ]
    for index, label in enumerate(("Interleaved BTHD", "Zero-stride GQA BTHD")):
        x = 170 + index * 610
        parts.append(f'<rect x="{x}" y="155" width="500" height="235" rx="14" '
                     'fill="#fff1f2" stroke="#e11d48" stroke-width="3"/>')
        parts.append(text(x + 250, 212, label, 22, "#9f1239", anchor="middle", weight=700))
        parts.append(text(x + 250, 282, "status 6", 34, "#e11d48", anchor="middle", weight=700))
        parts.append(text(x + 250, 338, "unsupported before timing", 17,
                          "#5b6474", anchor="middle"))
    parts.append(text(width / 2, 475, "Supported 0 / 2 · timed 0 · model routes 0",
                      20, "#b42335", anchor="middle", weight=700))
    parts.append(text(width / 2, 530,
                      "Both remaining vendor mixed-dtype cast routes are now closed",
                      18, "#5b6474", anchor="middle", weight=700))
    parts.append(text(width / 2, 580,
                      "Candidate APIs removed; future work needs a different kernel architecture",
                      14, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def inference_local_saturation_svg() -> str:
    summary = json.loads((INFERENCE_LOCAL_SATURATION_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    width, height = 1500, 760
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 243 · Current Inference Local Saturation", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "Measured upper bounds and six closed local tracks · MI300X/gfx942",
             16, "#5b6474", anchor="middle"),
    ]
    models = (
        ("Qwen2.5-0.5B", summary["qwen_cast_share"],
         summary["qwen_perfect_cast_deletion_upper_bound"]),
        ("DeepSeek-Distill-1.5B", summary["deepseek_cast_share"],
         summary["deepseek_perfect_cast_deletion_upper_bound"]),
    )
    for index, (label, share, upper) in enumerate(models):
        x = 150 + index * 650
        parts.extend([
            f'<rect x="{x}" y="135" width="550" height="170" rx="16" '
            'fill="#eef6ff" stroke="#2563eb" stroke-width="2"/>',
            text(x + 275, 180, label, 22, "#1e3a8a", anchor="middle", weight=700),
            text(x + 275, 230, f"remaining cast share {share * 100:.2f}%", 19,
                 "#334155", anchor="middle"),
            text(x + 275, 275, f"free-deletion ceiling {upper:.4f}x", 27,
                 "#2563eb", anchor="middle", weight=700),
        ])
    closed = [
        "online Attention model", "exact Attention solution",
        "vectorized SwiGLU", "grouped Swish epilogue",
        "direct BF16 P×V output", "BF16 V into P×V",
    ]
    for index, label in enumerate(closed):
        row, column = divmod(index, 3)
        x, y = 95 + column * 470, 370 + row * 105
        parts.extend([
            f'<rect x="{x}" y="{y}" width="400" height="72" rx="12" '
            'fill="#fff1f2" stroke="#e11d48" stroke-width="2"/>',
            text(x + 200, y + 31, label, 17, "#9f1239", anchor="middle", weight=700),
            text(x + 200, y + 57, "closed by measured counterexample", 13,
                 "#5b6474", anchor="middle"),
        ])
    parts.append(text(width / 2, 625,
                      "Decision: stop retuning local default-policy knobs",
                      23, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 672,
                      "Next contract: a new custom-kernel or graph-wide architecture",
                      20, "#172033", anchor="middle", weight=700))
    parts.append(text(width / 2, 716,
                      "This closes one search scale, not the whole inference roadmap",
                      15, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def current_training_profile_svg() -> str:
    summary = json.loads((CURRENT_TRAINING_PROFILE_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    rows = {row["model"]: row for row in summary["models"]}
    width, height = 1500, 720
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 244 · Current B1T512 Training Profile", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "BF16 Linear + BF16 AdamW moments · (three-step − one-step) / 2",
             16, "#5b6474", anchor="middle"),
    ]
    models = (
        ("Qwen2.5-0.5B", rows["qwen2.5-0.5b"], 1.025195919807247),
        ("DeepSeek-Distill-1.5B", rows["deepseek-r1-distill-qwen-1.5b"],
         1.014371378731901),
    )
    colors = {"hipBLASLt GEMM": "#2563eb", "AdamW": "#7c3aed",
              "other": "#cbd5e1"}
    for index, (label, model, speedup) in enumerate(models):
        x = 130 + index * 680
        total_ms = model["total_kernel_ns_per_step"] / 1.0e6
        categories = {row["category"]: row for row in model["categories"]}
        gemm = categories["hipBLASLt GEMM"]["kernel_share"]
        adamw = categories["AdamW"]["kernel_share"]
        other = 1.0 - gemm - adamw
        parts.extend([
            text(x + 275, 150, label, 23, "#172033", anchor="middle", weight=700),
            text(x + 275, 190, f"{total_ms:.3f} ms / stable step", 22,
                 "#172033", anchor="middle", weight=700),
        ])
        cursor = x
        for name, share in (("hipBLASLt GEMM", gemm), ("AdamW", adamw),
                            ("other", other)):
            bar_width = 550 * share
            parts.append(f'<rect x="{cursor:.1f}" y="235" width="{bar_width:.1f}" '
                         f'height="72" fill="{colors[name]}"/>')
            cursor += bar_width
        parts.extend([
            text(x, 340, f"GEMM {gemm * 100:.2f}%", 18, "#2563eb", weight=700),
            text(x + 205, 340, f"AdamW {adamw * 100:.2f}%", 18,
                 "#7c3aed", weight=700),
            text(x + 550, 390, f"{speedup:.4f}x vs Experiment 216", 19,
                 "#166534", anchor="end", weight=700),
        ])
    parts.append(text(width / 2, 505,
                      "Hotspot order is unchanged on the current retained binary",
                      24, "#172033", anchor="middle", weight=700))
    parts.append(text(width / 2, 560,
                      "GEMM remains 58.56% / 63.43% · AdamW threshold search stays closed",
                      19, "#5b6474", anchor="middle"))
    parts.append(text(width / 2, 625,
                      "Next contract: new training GEMM or graph-wide architecture",
                      22, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 675,
                      "Kernel phase attribution is not an end-to-end speed claim",
                      15, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_weight_gradient_shapes_svg() -> str:
    summary = json.loads((BF16_WGRAD_SHAPE_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    width, height = 1500, 800
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 245 · Cast-inclusive BF16 Weight Gradient", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "Three fresh MI300X processes per B1T512 shape · operator gate 1.05x",
             16, "#5b6474", anchor="middle"),
    ]
    labels = {"query": "Query", "kv": "KV", "gate": "Gate / Up"}
    for index, row in enumerate(summary["shapes"]):
        group = index // 3
        column = index % 3
        x, y = 90 + column * 470, 145 + group * 255
        passing = row["passes_operator_performance_gate"]
        fill = "#ecfdf3" if passing else "#fff1f2"
        stroke = "#16a34a" if passing else "#e11d48"
        model = "Qwen2.5-0.5B" if group == 0 else "DeepSeek-Distill-1.5B"
        parts.extend([
            f'<rect x="{x}" y="{y}" width="400" height="205" rx="14" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="3"/>',
            text(x + 200, y + 40, model, 16, "#5b6474", anchor="middle"),
            text(x + 200, y + 76, labels[row["family"]], 22, "#172033",
                 anchor="middle", weight=700),
            text(x + 200, y + 126, f"{row['event_speedup_median']:.3f}x",
                 32, stroke, anchor="middle", weight=700),
            text(x + 200, y + 160,
                 f"minimum {row['event_speedup_minimum']:.3f}x", 15,
                 "#5b6474", anchor="middle"),
            text(x + 200, y + 190,
                 "MODEL GATE" if passing else "REJECT", 15, stroke,
                 anchor="middle", weight=700),
        ])
    parts.append(text(width / 2, 685,
                      "2 / 6 shapes pass · no universal BF16 gradient policy",
                      24, "#172033", anchor="middle", weight=700))
    parts.append(text(width / 2, 735,
                      "Only gate/up enters an explicit, default-off official-model A/B",
                      19, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 775,
                      "Candidate time includes both casts; FP32 Max/RMS remains visible",
                      14, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_weight_gradient_model_svg() -> str:
    summary = json.loads((BF16_WGRAD_MODEL_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    width, height = 1500, 720
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 246 · Gate/Up BF16 Weight Gradient", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "Three alternating-order same-binary pairs · B1T512 · two measured steps",
             16, "#5b6474", anchor="middle"),
    ]
    for index, row in enumerate(summary["comparisons"]):
        x = 145 + index * 680
        model = "Qwen2.5-0.5B" if index == 0 else "DeepSeek-Distill-1.5B"
        speedup = row["throughput_speedup"]
        route = row["diagnostics"]["bf16_gate_up_weight_gradient_assignments"]
        parts.extend([
            f'<rect x="{x}" y="145" width="540" height="330" rx="16" '
            'fill="#ecfdf3" stroke="#16a34a" stroke-width="3"/>',
            text(x + 270, 195, model, 23, "#172033", anchor="middle", weight=700),
            text(x + 270, 270, f"{speedup:.4f}x", 42, "#16a34a",
                 anchor="middle", weight=700),
            text(x + 270, 310, "end-to-end training throughput", 16,
                 "#5b6474", anchor="middle"),
            text(x + 60, 370, f"route assignments  {route}", 17, "#172033"),
            text(x + 60, 405, f"peak ratio  {row['peak_ratio']:.3f}", 17,
                 "#172033"),
            text(x + 60, 440,
                 f"final loss diff  {row['final_loss_relative_difference'] * 100:.4f}%",
                 17, "#172033"),
        ])
    parts.append(text(width / 2, 555,
                      "All six short model gates pass · peak memory unchanged",
                      24, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 615,
                      "Candidate adds 192 / 224 logical allocations over two steps",
                      18, "#5b6474", anchor="middle"))
    parts.append(text(width / 2, 670,
                      "Decision: keep explicit; require a longer trajectory before default",
                      20, "#b45309", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_weight_gradient_trajectory_svg() -> str:
    summary = json.loads((BF16_WGRAD_TRAJECTORY_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    width, height = 1500, 790
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 247 · 20-Step BF16 Weight-Gradient Gate", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "Three process pairs · 240 losses · 979,894,272 complete parameter values",
             16, "#5b6474", anchor="middle"),
    ]
    metrics = (
        ("Throughput ≥ 1.01x", "throughput"),
        ("Peak ≤ 1.01x", "peak_memory"),
        ("Loss diff ≤ 0.5%", "loss_trajectory"),
        ("Parameter Max ≤ 5e-5", "parameter_maximum"),
        ("Parameter RMS ≤ 1e-6", "parameter_rms"),
    )
    for index, (label, key) in enumerate(metrics):
        passed = summary["gate_results"][key]
        x = 95 + index * 270
        fill = "#ecfdf3" if passed else "#fff1f2"
        stroke = "#16a34a" if passed else "#e11d48"
        parts.extend([
            f'<rect x="{x}" y="145" width="230" height="130" rx="14" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="3"/>',
            text(x + 115, 190, label, 15, "#172033", anchor="middle", weight=700),
            text(x + 115, 245, "PASS" if passed else "FAIL", 27,
                 stroke, anchor="middle", weight=700),
        ])
    for index, row in enumerate(summary["comparisons"]):
        x = 150 + index * 690
        model = "Qwen2.5-0.5B" if index == 0 else "DeepSeek-Distill-1.5B"
        parameter = row["parameter_comparison"]
        parts.extend([
            f'<rect x="{x}" y="340" width="520" height="250" rx="16" '
            'fill="#ffffff" stroke="#cbd5e1" stroke-width="2"/>',
            text(x + 260, 385, model, 22, "#172033", anchor="middle", weight=700),
            text(x + 45, 435, f"throughput  {row['throughput_speedup']:.4f}x", 18),
            text(x + 45, 475,
                 f"loss max relative  {row['loss_relative_difference_maximum']:.6g}", 18),
            text(x + 45, 515,
                 f"parameter Max  {parameter['maximum_absolute_difference']:.3e}", 18),
            text(x + 45, 555,
                 f"parameter RMS  {parameter['rms_difference']:.3e}", 18),
        ])
    parts.append(text(width / 2, 660,
                      "Only 1 / 5 aggregate gates passes · model route rejected",
                      26, "#e11d48", anchor="middle", weight=700))
    parts.append(text(width / 2, 715,
                      "Retain the standalone operator and complete evidence tools",
                      20, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 760,
                      "Autograd/CLI candidate wiring and candidate runners removed",
                      15, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_weight_gradient_allocation_svg() -> str:
    summary = json.loads((BF16_WGRAD_ALLOCATION_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    width, height = 1500, 720
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 248 · Allocation Source Identity", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "20-step retained evidence · rejected model route is not restored",
             16, "#5b6474", anchor="middle"),
    ]
    for index, row in enumerate(summary["models"]):
        x = 145 + index * 680
        model = "Qwen2.5-0.5B" if index == 0 else "DeepSeek-Distill-1.5B"
        parts.extend([
            f'<rect x="{x}" y="145" width="540" height="365" rx="16" '
            'fill="#eef6ff" stroke="#2563eb" stroke-width="3"/>',
            text(x + 270, 195, model, 23, "#172033", anchor="middle", weight=700),
            text(x + 45, 255, f"routes  {row['total_routes']}", 18),
            text(x + 45, 295,
                 f"allocation delta  {row['allocation_calls_delta']}", 18),
            text(x + 45, 335, "exactly 2 logical allocations / route", 18,
                 "#2563eb", weight=700),
            text(x + 45, 390,
                 f"bytes / route  {row['bytes_per_route']:,}", 18),
            text(x + 45, 430,
                 f"two cast buffers  {row['expected_cast_bytes_per_route']:,}", 18),
            text(x + 270, 480, "EXACT IDENTITY", 23, "#166534",
                 anchor="middle", weight=700),
        ])
    parts.append(text(width / 2, 575,
                      "Backend allocations +0 · peak bytes +0 · cached bytes +0",
                      22, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 635,
                      "Attribution is complete; workspace speedup is not yet proven",
                      22, "#b45309", anchor="middle", weight=700))
    parts.append(text(width / 2, 685,
                      "Next: allocating vs preallocated wall/Event gate before any API",
                      16, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def bf16_weight_gradient_workspace_svg() -> str:
    summary = json.loads((BF16_WGRAD_WORKSPACE_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    width, height = 1500, 700
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 249 · Workspace Cost Gate", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "Preallocated / allocating · cache already primed · three fresh processes",
             16, "#5b6474", anchor="middle"),
    ]
    for index, row in enumerate(summary["models"]):
        x = 145 + index * 680
        model = "Qwen2.5-0.5B" if index == 0 else "DeepSeek-Distill-1.5B"
        parts.extend([
            f'<rect x="{x}" y="145" width="540" height="315" rx="16" '
            'fill="#fff1f2" stroke="#e11d48" stroke-width="3"/>',
            text(x + 270, 195, model, 23, "#172033", anchor="middle", weight=700),
            text(x + 70, 265, "Event", 18, "#5b6474"),
            text(x + 470, 265, f"{row['event_speedup_median']:.3f}x", 27,
                 "#172033", anchor="end", weight=700),
            text(x + 70, 325, "Wall", 18, "#5b6474"),
            text(x + 470, 325, f"{row['wall_speedup_median']:.3f}x", 31,
                 "#e11d48", anchor="end", weight=700),
            text(x + 70, 375, "Minimum wall", 18, "#5b6474"),
            text(x + 470, 375, f"{row['wall_speedup_minimum']:.3f}x", 23,
                 "#e11d48", anchor="end", weight=700),
            text(x + 270, 430, "REJECT", 22, "#e11d48",
                 anchor="middle", weight=700),
        ])
    parts.append(text(width / 2, 535,
                      "0 / 2 shapes pass the 1.01 wall gate",
                      26, "#e11d48", anchor="middle", weight=700))
    parts.append(text(width / 2, 590,
                      "3 cache reuses / public call · 0 backend allocations",
                      20, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 645,
                      "Decision: do not add a workspace API",
                      22, "#172033", anchor="middle", weight=700))
    parts.append("</svg>\n")
    return "\n".join(parts)


def training_local_saturation_svg() -> str:
    summary = json.loads((TRAINING_LOCAL_SATURATION_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    width, height = 1500, 820
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 250 · Current Training Local Saturation", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "Current B1T512 profile · six adjacent model-policy tracks closed",
             16, "#5b6474", anchor="middle"),
    ]
    models = (
        ("Qwen2.5-0.5B", summary["qwen_kernel_ms"],
         summary["qwen_gemm_share"], summary["qwen_adamw_share"],
         summary["qwen_perfect_cast_deletion_ceiling"]),
        ("DeepSeek-Distill-1.5B", summary["deepseek_kernel_ms"],
         summary["deepseek_gemm_share"], summary["deepseek_adamw_share"],
         summary["deepseek_perfect_cast_deletion_ceiling"]),
    )
    for index, (model, kernel, gemm, adamw, cast_ceiling) in enumerate(models):
        x = 145 + index * 680
        parts.extend([
            f'<rect x="{x}" y="130" width="540" height="220" rx="16" '
            'fill="#eef6ff" stroke="#2563eb" stroke-width="2"/>',
            text(x + 270, 175, model, 22, "#172033", anchor="middle", weight=700),
            text(x + 55, 225, f"Kernel / step  {kernel:.3f} ms", 18),
            text(x + 55, 265, f"GEMM  {gemm * 100:.2f}%", 18, "#2563eb", weight=700),
            text(x + 300, 265, f"AdamW  {adamw * 100:.2f}%", 18,
                 "#7c3aed", weight=700),
            text(x + 270, 320, f"free cast deletion ≤ {cast_ceiling:.4f}x", 19,
                 "#5b6474", anchor="middle"),
        ])
    labels = (
        "grouped dW", "packed dW", "exact dW index",
        "optimizer Graph", "BF16 dW trajectory", "BF16 dW workspace",
    )
    for index, label in enumerate(labels):
        row, column = divmod(index, 3)
        x, y = 95 + column * 470, 410 + row * 105
        parts.extend([
            f'<rect x="{x}" y="{y}" width="400" height="72" rx="12" '
            'fill="#fff1f2" stroke="#e11d48" stroke-width="2"/>',
            text(x + 200, y + 31, label, 18, "#9f1239", anchor="middle", weight=700),
            text(x + 200, y + 57, "closed by operator/model/trajectory gate", 13,
                 "#5b6474", anchor="middle"),
        ])
    parts.append(text(width / 2, 660,
                      "Decision: stop local default-policy retuning",
                      25, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 720,
                      "Next scale: new custom kernel / graph architecture / production reducer",
                      20, "#172033", anchor="middle", weight=700))
    parts.append(text(width / 2, 775,
                      "This closes one search scale, not the training roadmap",
                      15, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def current_data_parallel_svg() -> str:
    audit = json.loads((CURRENT_DATA_PARALLEL_ROOT / "gap-audit.json").read_text(
        encoding="utf-8"))
    width, height = 1500, 760
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 251 · Current Two-GPU Data Parallel", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "20 steps · 4 MiB bucket limit · rank parameters remain identical",
             16, "#5b6474", anchor="middle"),
    ]
    shares = (
        ("Forward + backward", audit["forward_backward_share"], "#2563eb"),
        ("Communication", audit["communication_share"], "#7c3aed"),
        ("Optimizer", audit["optimizer_share"], "#16a34a"),
        ("Parameter audit", audit["unattributed_verification_share"], "#f59e0b"),
    )
    cursor = 120.0
    for label, share, color in shares:
        bar_width = 1260.0 * share
        parts.append(f'<rect x="{cursor:.1f}" y="145" width="{bar_width:.1f}" '
                     f'height="95" fill="{color}"/>')
        if bar_width >= 105:
            parts.append(text(cursor + bar_width / 2, 185,
                              f"{share * 100:.1f}%", 18, "#ffffff",
                              anchor="middle", weight=700))
            parts.append(text(cursor + bar_width / 2, 216, label, 13,
                              "#ffffff", anchor="middle"))
        cursor += bar_width
    parts.append(text(width / 2, 290,
                      "Median total 2.290 ms · communication 0.350 ms · parameter audit residual 0.305 ms",
                      19, "#172033", anchor="middle", weight=700))
    gaps = (
        ("Complete backward sync before communication", False),
        ("Real gradient-ready overlap", False),
        ("Persistent gradient buckets / zero-copy views", False),
        ("One process per GPU", False),
        ("Two-rank global-batch equivalence", True),
        ("RCCL current validation 14 / 14", True),
    )
    for index, (label, available) in enumerate(gaps):
        row, column = divmod(index, 2)
        x, y = 120 + column * 660, 355 + row * 90
        color = "#16a34a" if available else "#e11d48"
        fill = "#ecfdf3" if available else "#fff1f2"
        parts.extend([
            f'<rect x="{x}" y="{y}" width="600" height="62" rx="11" '
            f'fill="{fill}" stroke="{color}" stroke-width="2"/>',
            text(x + 35, y + 39, "✓" if available else "×", 25,
                 color, anchor="middle", weight=700),
            text(x + 70, y + 38, label, 17, "#172033", weight=700),
        ])
    parts.append(text(width / 2, 660,
                      "First contract: separate parameter-audit timing and interval",
                      23, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 710,
                      "One tiny-model bucket cannot prove backward / communication overlap",
                      16, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def data_parallel_verification_svg() -> str:
    summary = json.loads((DATA_PARALLEL_VERIFICATION_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    width, height = 1500, 720
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 252 · Parameter Verification Interval", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "Three rotated fresh processes · steady medians use steps 2–20",
             16, "#5b6474", anchor="middle"),
    ]
    order = (("every_step", "Every step", "#2563eb"),
             ("final_step", "Final step", "#16a34a"),
             ("disabled", "Disabled", "#f59e0b"))
    maximum = max(row["median_total_ms"] for row in summary["policies"].values())
    for index, (key, label, color) in enumerate(order):
        row = summary["policies"][key]
        x = 150 + index * 450
        height = 330 * row["median_total_ms"] / maximum
        y = 485 - height
        parts.extend([
            f'<rect x="{x}" y="{y:.1f}" width="280" height="{height:.1f}" '
            f'rx="10" fill="{color}"/>',
            text(x + 140, y - 18, f"{row['median_total_ms']:.3f} ms", 22,
                 color, anchor="middle", weight=700),
            text(x + 140, 525, label, 20, "#172033", anchor="middle", weight=700),
            text(x + 140, 558,
                 f"checks {row['parameter_checks_per_process']} / 20", 15,
                 "#5b6474", anchor="middle"),
        ])
        if key != "every_step":
            parts.append(text(x + 140, 600,
                              f"{row['speedup_vs_every_step']:.3f}x", 23,
                              "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 645,
                      "180 / 180 loss values exact · default interval remains 1",
                      21, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 690,
                      "Optimizer completion is explicit; skipped verification is not a correctness claim",
                      15, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def data_parallel_bucket_svg() -> str:
    summary = json.loads((DATA_PARALLEL_BUCKET_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    width, height = 1500, 740
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 253 · Tiny Real Bucket-Count Matrix", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "Final-step parameter audit · three rotated processes · steps 2–20",
             16, "#5b6474", anchor="middle"),
    ]
    order = (("4b", "4 B"), ("64b", "64 B"),
             ("4kib", "4 KiB"), ("4mib", "4 MiB"))
    max_total = max(row["median_total_ms"] for row in summary["policies"].values())
    for index, (key, label) in enumerate(order):
        row = summary["policies"][key]
        x = 100 + index * 350
        bar_height = 320 * row["median_total_ms"] / max_total
        y = 465 - bar_height
        color = "#e11d48" if row["bucket_count"] > 1 else "#2563eb"
        parts.extend([
            f'<rect x="{x}" y="{y:.1f}" width="240" height="{bar_height:.1f}" '
            f'rx="9" fill="{color}"/>',
            text(x + 120, y - 18, f"{row['median_total_ms']:.2f} ms", 21,
                 color, anchor="middle", weight=700),
            text(x + 120, 510, label, 20, "#172033", anchor="middle", weight=700),
            text(x + 120, 545, f"{row['bucket_count']} bucket(s)", 16,
                 "#5b6474", anchor="middle"),
            text(x + 120, 578, f"comm {row['median_communication_ms']:.2f} ms", 16,
                 "#5b6474", anchor="middle"),
        ])
    parts.append(text(width / 2, 635,
                      "4 KiB and 4 MiB are the same one-bucket workload",
                      22, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 685,
                      "Twelve tiny buckets are slower · next: Model-S multi-bucket workload",
                      20, "#172033", anchor="middle", weight=700))
    parts.append(text(width / 2, 720,
                      "Synthetic or artificial tiny buckets are not readiness-overlap evidence",
                      14, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def data_parallel_model_s_svg() -> str:
    summary = json.loads((DATA_PARALLEL_MODEL_S_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    width, height = 1500, 750
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 254 · Model-S Natural Buckets", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "15,586,176 parameters · B1T32 · final-step rank audit",
             16, "#5b6474", anchor="middle"),
    ]
    order = (("1mib", "1 MiB"), ("4mib", "4 MiB"), ("25mib", "25 MiB"))
    max_total = max(row["median_total_ms"] for row in summary["policies"].values())
    for index, (key, label) in enumerate(order):
        row = summary["policies"][key]
        x = 150 + index * 450
        bar_height = 320 * row["median_total_ms"] / max_total
        y = 465 - bar_height
        color = "#16a34a" if key == summary["best_policy"] else "#2563eb"
        parts.extend([
            f'<rect x="{x}" y="{y:.1f}" width="280" height="{bar_height:.1f}" '
            f'rx="10" fill="{color}"/>',
            text(x + 140, y - 18, f"{row['median_total_ms']:.2f} ms", 22,
                 color, anchor="middle", weight=700),
            text(x + 140, 510, label, 20, "#172033", anchor="middle", weight=700),
            text(x + 140, 545, f"{row['bucket_count']} buckets", 17,
                 "#5b6474", anchor="middle"),
            text(x + 140, 578, f"comm {row['median_communication_ms']:.3f} ms", 16,
                 "#5b6474", anchor="middle"),
            text(x + 140, 610,
                 f"peak {row['maximum_engine_peak_bytes'] / (1024**2):.1f} MiB", 16,
                 "#5b6474", anchor="middle"),
        ])
    parts.append(text(width / 2, 665,
                      "25 MiB / 3 buckets is the current reducer baseline",
                      23, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 710,
                      "Peak tradeoff +54,294,528 bytes · overlap not implemented yet",
                      15, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def data_parallel_copy_attribution_svg() -> str:
    audit = json.loads((DATA_PARALLEL_COPY_ROOT / "attribution.json").read_text(
        encoding="utf-8"))
    width, height = 1500, 730
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 255 · Reducer Temporary Identity", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "Model-S · 25 MiB · 3 buckets · current clean commit",
             16, "#5b6474", anchor="middle"),
    ]
    groups = (
        ("Bucket", audit["bucket_tensor_count"], "#2563eb"),
        ("Average", audit["average_tensor_count"], "#7c3aed"),
        ("Unpacked gradients", audit["unpacked_tensor_count"], "#e11d48"),
    )
    maximum = max(value for _, value, _ in groups)
    for index, (label, value, color) in enumerate(groups):
        x = 150 + index * 450
        height = 290 * value / maximum
        y = 440 - height
        parts.extend([
            f'<rect x="{x}" y="{y:.1f}" width="280" height="{height:.1f}" '
            f'rx="9" fill="{color}"/>',
            text(x + 140, y - 18, str(value), 26, color,
                 anchor="middle", weight=700),
            text(x + 140, 480, label, 18, "#172033", anchor="middle", weight=700),
        ])
    parts.append(text(width / 2, 545,
                      "126 tensors = 126 backend allocations · cache reuse 0",
                      23, "#e11d48", anchor="middle", weight=700))
    parts.append(text(width / 2, 590,
                      "228 D2D copies · 374,068,224 temporary bytes / step",
                      21, "#172033", anchor="middle", weight=700))
    parts.append(text(width / 2, 640,
                      "Communication 7.26 ms = 32.31% of steady total",
                      21, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 690,
                      "Persistent design must cover bucket + average + unpacked gradients",
                      16, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def data_parallel_inplace_average_svg() -> str:
    summary = json.loads((DATA_PARALLEL_INPLACE_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    width, height = 1500, 720
    baseline = summary["policies"]["allocating"]
    candidate = summary["policies"]["inplace"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 256 · In-Place Bucket Average", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "Same binary · Model-S · 25 MiB / 3 buckets · final parameter audit",
             16, "#5b6474", anchor="middle"),
    ]
    for index, (label, row, color) in enumerate((
            ("Allocating", baseline, "#64748b"),
            ("In-place", candidate, "#16a34a"))):
        x = 220 + index * 650
        parts.extend([
            f'<rect x="{x}" y="145" width="430" height="330" rx="16" '
            f'fill="#ffffff" stroke="{color}" stroke-width="3"/>',
            text(x + 215, 195, label, 25, color, anchor="middle", weight=700),
            text(x + 55, 260,
                 f"communication  {row['median_communication_ms']:.2f} ms", 19),
            text(x + 55, 310, f"total  {row['median_total_ms']:.2f} ms", 22,
                 color, weight=700),
            text(x + 55, 360,
                 f"peak  {row['maximum_engine_peak_bytes'] / (1024**2):.1f} MiB", 18),
            text(x + 55, 410,
                 "6 average tensors" if index == 0 else "0 average tensors", 18),
        ])
    parts.append(text(width / 2, 545,
                      f"Communication {summary['communication_speedup']:.3f}x · total {summary['total_speedup']:.3f}x",
                      25, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 600,
                      "30 / 30 losses exact · peak unchanged · RCCL 22 / 22",
                      20, "#172033", anchor="middle", weight=700))
    parts.append(text(width / 2, 655,
                      "Default kept; 120 backend allocations and 228 copies remain",
                      18, "#b45309", anchor="middle", weight=700))
    parts.append(text(width / 2, 695,
                      "Next: persistent bucket + unpacked-gradient storage",
                      15, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def data_parallel_persistent_bucket_svg() -> str:
    summary = json.loads((DATA_PARALLEL_PERSISTENT_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    width, height = 1500, 760
    baseline = summary["policies"]["transient"]
    candidate = summary["policies"]["persistent"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 257 · Persistent Gradient Buckets", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "Same binary · Model-S · 25 MiB / 3 buckets · step 2–5 medians",
             16, "#5b6474", anchor="middle"),
    ]
    for index, (label, row, color, allocations) in enumerate((
            ("Transient each step", baseline, "#64748b", 120),
            ("Persistent plan", candidate, "#16a34a", 0))):
        x = 170 + index * 680
        parts.extend([
            f'<rect x="{x}" y="135" width="480" height="385" rx="16" '
            f'fill="#ffffff" stroke="{color}" stroke-width="3"/>',
            text(x + 240, 185, label, 25, color, anchor="middle", weight=700),
            text(x + 55, 245,
                 f"later backend allocs  {allocations}", 19),
            text(x + 55, 295,
                 f"communication  {row['median_communication_ms']:.3f} ms", 19),
            text(x + 55, 345, f"total  {row['median_total_ms']:.3f} ms", 22,
                 color, weight=700),
            text(x + 55, 400,
                 f"live  {row['median_engine_current_bytes'] / (1024**2):.1f} MiB", 18),
            text(x + 55, 450,
                 f"peak  {row['maximum_engine_peak_bytes'] / (1024**2):.1f} MiB", 18),
        ])
    parts.append(text(width / 2, 585,
                      f"Communication {summary['communication_speedup']:.3f}x · total {summary['total_speedup']:.3f}x",
                      25, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 635,
                      "30 / 30 losses exact · later allocations 120 → 0 · rank parameters exact",
                      20, "#172033", anchor="middle", weight=700))
    parts.append(text(width / 2, 682,
                      f"Live +{summary['current_bytes_added'] / (1024**2):.1f} MiB · peak +{summary['peak_bytes_added'] / (1024**2):.1f} MiB · explicit, not default",
                      19, "#b45309", anchor="middle", weight=700))
    parts.append(text(width / 2, 726,
                      "Next: parameter gradients become views into reduced bucket Storage",
                      15, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def data_parallel_gradient_view_svg() -> str:
    summary = json.loads((DATA_PARALLEL_GRADIENT_VIEW_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    width, height = 1600, 780
    policies = summary["policies"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 258 · Gradients as Bucket Views", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "Same binary · 3 rotated policies · Model-S B1T32 · step 2–5 medians",
             16, "#5b6474", anchor="middle"),
    ]
    cards = (
        ("Transient", policies["transient"], "#64748b", "114 unpack copies"),
        ("Persistent copy", policies["persistent_copy"], "#b45309",
         "114 unpack copies"),
        ("Bucket views", policies["bucket_views"], "#16a34a",
         "0 unpack copies"),
    )
    for index, (label, row, color, copy_label) in enumerate(cards):
        x = 80 + index * 510
        parts.extend([
            f'<rect x="{x}" y="135" width="420" height="390" rx="16" '
            f'fill="#ffffff" stroke="{color}" stroke-width="3"/>',
            text(x + 210, 185, label, 24, color, anchor="middle", weight=700),
            text(x + 45, 245,
                 f"communication  {row['median_communication_ms']:.3f} ms", 18),
            text(x + 45, 300, f"total  {row['median_total_ms']:.3f} ms", 22,
                 color, weight=700),
            text(x + 45, 360,
                 f"live  {row['median_engine_current_bytes'] / (1024**2):.1f} MiB", 18),
            text(x + 45, 410,
                 f"peak  {row['maximum_engine_peak_bytes'] / (1024**2):.1f} MiB", 18),
            text(x + 45, 470, copy_label, 18, color, weight=700),
        ])
    parts.append(text(width / 2, 590,
                      f"View vs copy: communication {summary['view_vs_copy_communication_speedup']:.3f}x · total {summary['view_vs_copy_total_speedup']:.3f}x",
                      23, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 635,
                      f"View vs transient: communication {summary['view_vs_transient_communication_speedup']:.3f}x · total {summary['view_vs_transient_total_speedup']:.3f}x",
                      23, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 682,
                      "45 / 45 losses exact · live equals transient · peak still +31.7 MiB",
                      19, "#b45309", anchor="middle", weight=700))
    parts.append(text(width / 2, 730,
                      "Explicit, not default · next: backward accumulates directly into bucket views",
                      16, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def data_parallel_direct_gradient_svg() -> str:
    summary = json.loads((DATA_PARALLEL_DIRECT_GRADIENT_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    width, height = 1600, 790
    policies = summary["policies"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 259 · Zero Copies, Slower Step", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "Model-S · 3 rotated policies · forward/backward and communication reported separately",
             16, "#5b6474", anchor="middle"),
    ]
    cards = (
        ("Transient", policies["transient"], "#64748b", "114 pack + 114 unpack"),
        ("Bucket views", policies["bucket_views"], "#16a34a", "114 pack + 0 unpack"),
        ("Direct targets", policies["direct"], "#e11d48", "0 pack + 0 unpack"),
    )
    for index, (label, row, color, copies) in enumerate(cards):
        x = 80 + index * 510
        parts.extend([
            f'<rect x="{x}" y="135" width="420" height="405" rx="16" '
            f'fill="#ffffff" stroke="{color}" stroke-width="3"/>',
            text(x + 210, 185, label, 24, color, anchor="middle", weight=700),
            text(x + 45, 245,
                 f"forward/backward  {row['median_forward_backward_ms']:.3f} ms", 18),
            text(x + 45, 300,
                 f"communication  {row['median_communication_ms']:.3f} ms", 18),
            text(x + 45, 355, f"total  {row['median_total_ms']:.3f} ms", 22,
                 color, weight=700),
            text(x + 45, 415,
                 f"peak  {row['maximum_engine_peak_bytes'] / (1024**2):.1f} MiB", 18),
            text(x + 45, 480, copies, 17, color, weight=700),
        ])
    parts.append(text(width / 2, 602,
                      f"Direct communication {summary['communication_speedup_vs_views']:.3f}x · peak -{summary['peak_bytes_saved_vs_views'] / (1024**2):.1f} MiB vs views",
                      23, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 650,
                      f"But forward/backward {summary['forward_backward_speedup_vs_views']:.3f}x · total {summary['total_speedup_vs_views']:.3f}x",
                      24, "#b42335", anchor="middle", weight=700))
    parts.append(text(width / 2, 700,
                      "Producer still allocates gradient, then leaf target adds it again · model route rejected",
                      19, "#b42335", anchor="middle", weight=700))
    parts.append(text(width / 2, 748,
                      "Next: one producer out-kernel must remove both temporary output and leaf add",
                      16, "#5b6474", anchor="middle"))
    parts.append("</svg>\n")
    return "\n".join(parts)


def gradient_producer_out_svg() -> str:
    summary = json.loads((GRADIENT_PRODUCER_OUT_ROOT / "summary.json").read_text(
        encoding="utf-8"))
    order = (
        ("model_s_head_t32", "Head T32"),
        ("model_s_ffn_t32", "FFN T32"),
        ("model_s_attention_t32", "Attn T32"),
        ("model_s_head_t512", "Head T512"),
        ("tiny_counterexample", "Tiny"),
    )
    width, height = 1600, 760
    chart_x, chart_y, chart_w, chart_h = 130, 145, 1340, 390
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        text(width / 2, 48, "Experiment 260 · Caller-Owned Gradient Producer", 30,
             anchor="middle", weight=700),
        text(width / 2, 82,
             "15 exact outputs · 3 fresh processes / shape · allocating + leaf add vs direct output",
             16, "#5b6474", anchor="middle"),
        f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" '
        'fill="#ffffff" stroke="#cbd3df" rx="10"/>',
    ]

    def y(value: float) -> float:
        return chart_y + chart_h * (2.0 - value) / 1.0

    for tick in (1.0, 1.25, 1.5, 1.75, 2.0):
        position = y(tick)
        color = "#2563eb" if tick == 1.0 else "#e5e9f0"
        parts.append(f'<line x1="{chart_x}" y1="{position:.1f}" '
                     f'x2="{chart_x + chart_w}" y2="{position:.1f}" stroke="{color}"/>')
        parts.append(text(chart_x - 12, position + 5, f"{tick:.2f}x", 13,
                          "#5b6474", anchor="end"))
    group_width = chart_w / len(order)
    for index, (key, label) in enumerate(order):
        row = summary["shapes"][key]
        center = chart_x + group_width * (index + 0.5)
        for offset, (value, color, name) in enumerate((
                (row["event_speedup"], "#2563eb", "Event"),
                (row["wall_speedup"], "#16a34a", "Wall"))):
            x_pos = center - 66 + offset * 70
            top = y(value)
            base = y(1.0)
            parts.append(f'<rect x="{x_pos:.1f}" y="{top:.1f}" width="60" '
                         f'height="{base - top:.1f}" fill="{color}" rx="5"/>')
            parts.append(text(x_pos + 30, top - 8, f"{value:.3f}x", 12,
                              color, anchor="middle", weight=700))
            parts.append(text(x_pos + 30, chart_y + chart_h + 22, name, 11,
                              "#5b6474", anchor="middle"))
        parts.append(text(center, chart_y + chart_h + 52, label, 14,
                          "#172033", anchor="middle", weight=700))
    parts.append(text(width / 2, 635,
                      "Every shape passes Event + Wall 1.05x · logical allocation 1 → 0",
                      22, "#166534", anchor="middle", weight=700))
    parts.append(text(width / 2, 682,
                      "CPU / HIP / PyTorch exact · admitted only to scoped Autograd right-leaf gate",
                      18, "#172033", anchor="middle", weight=700))
    parts.append(text(width / 2, 725,
                      "No model or DDP route yet",
                      15, "#5b6474", anchor="middle"))
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
                BATCHED_GEMM_CHART: strided_batched_hipblaslt_svg(),
                BATCHED_BACKWARD_CHART: batched_attention_backward_svg(),
                SAVED_ATTENTION_CHART: saved_attention_probabilities_svg(),
                BF16_ADAMW_CHART: bf16_adamw_moments_svg(),
                HYBRID_ADAMW_CHART: hybrid_bf16_adamw_svg(),
                POST_HYBRID_PROFILE_CHART: post_hybrid_training_profile_svg(),
                GROUPED_WGRAD_CHART: grouped_weight_gradient_discard_svg(),
                PACKED_WGRAD_CHART: packed_weight_gradient_discard_svg(),
                FP32_WGRAD_SOLUTION_CHART: fp32_weight_gradient_solutions_svg(),
                TRAINING_GRAPH_CHART: training_graph_capture_svg(),
                ADAMW_GRAPH_CHART: adamw_graph_replay_svg(),
                ADAMW_GRAPH_MULTI_CHART: adamw_graph_multi_svg(),
                GRADIENT_ADDRESS_CHART: gradient_address_stability_svg(),
                OPTIMIZER_GRAPH_PREFLIGHT_CHART:
                    optimizer_graph_model_preflight_svg(),
                QUIESCENT_HANDOFF_CHART: quiescent_allocator_handoff_svg(),
                OPTIMIZER_GRAPH_MODEL_CHART: optimizer_graph_model_gate_svg(),
                ROCWMMA_QK_CHART: rocwmma_qk_tile_svg(),
                ROCWMMA_ONLINE_CHART: rocwmma_online_attention_svg(),
                ROCWMMA_OPERATOR_CHART: rocwmma_online_operator_svg(),
                ROCWMMA_MODEL_CHART: rocwmma_online_model_svg(),
                ROCWMMA_DIRECT_MODEL_CHART: rocwmma_direct_bf16_model_svg(),
                CURRENT_INFERENCE_PROFILE_CHART: current_inference_profile_svg(),
                FP32_ATTENTION_T1024_CHART: fp32_attention_t1024_svg(),
                BF16_SWIGLU_VECTOR_CHART: bf16_swiglu_vector_svg(),
                BF16_GROUPED_SWISH_CHART: bf16_grouped_swish_svg(),
                BF16_RMS_NORM_OUTPUT_CHART: bf16_rms_norm_output_svg(),
                BF16_FFN_NORM_MODEL_CHART: bf16_ffn_norm_model_svg(),
                POST_BF16_FFN_NORM_PROFILE_CHART: post_bf16_ffn_norm_profile_svg(),
                BF16_ATTENTION_NORM_MODEL_CHART: bf16_attention_norm_model_svg(),
                POST_BF16_ATTENTION_NORM_PROFILE_CHART: post_bf16_attention_norm_profile_svg(),
                BF16_PV_OUTPUT_CHART: bf16_pv_output_svg(),
                BF16_VALUE_PV_CHART: bf16_value_pv_svg(),
                INFERENCE_LOCAL_SATURATION_CHART: inference_local_saturation_svg(),
                CURRENT_TRAINING_PROFILE_CHART: current_training_profile_svg(),
                BF16_WGRAD_SHAPE_CHART: bf16_weight_gradient_shapes_svg(),
                BF16_WGRAD_MODEL_CHART: bf16_weight_gradient_model_svg(),
                BF16_WGRAD_TRAJECTORY_CHART: bf16_weight_gradient_trajectory_svg(),
                BF16_WGRAD_ALLOCATION_CHART: bf16_weight_gradient_allocation_svg(),
                BF16_WGRAD_WORKSPACE_CHART: bf16_weight_gradient_workspace_svg(),
                TRAINING_LOCAL_SATURATION_CHART: training_local_saturation_svg(),
                CURRENT_DATA_PARALLEL_CHART: current_data_parallel_svg(),
                DATA_PARALLEL_VERIFICATION_CHART: data_parallel_verification_svg(),
                DATA_PARALLEL_BUCKET_CHART: data_parallel_bucket_svg(),
                DATA_PARALLEL_MODEL_S_CHART: data_parallel_model_s_svg(),
                DATA_PARALLEL_COPY_CHART: data_parallel_copy_attribution_svg(),
                DATA_PARALLEL_INPLACE_CHART: data_parallel_inplace_average_svg(),
                DATA_PARALLEL_PERSISTENT_CHART: data_parallel_persistent_bucket_svg(),
                DATA_PARALLEL_GRADIENT_VIEW_CHART: data_parallel_gradient_view_svg(),
                DATA_PARALLEL_DIRECT_GRADIENT_CHART: data_parallel_direct_gradient_svg(),
                GRADIENT_PRODUCER_OUT_CHART: gradient_producer_out_svg()}
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
