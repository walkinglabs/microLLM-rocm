#!/usr/bin/env python3
"""Render the complete Qwen3 one-step training alignment decision."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def metric_rows(summary: dict) -> list[tuple[str, float, float, bool]]:
    limits = summary["limits"]
    gradients = summary["gradients"]
    parameters = summary["parameters"]
    return [
        ("Gradient Max", gradients["maximum_absolute_difference"],
         limits["gradient_max"], summary["gates"]["gradient_maximum"]),
        ("Gradient RMS", gradients["rms_difference"],
         limits["gradient_rms"], summary["gates"]["gradient_rms"]),
        ("Parameter Max", parameters["maximum_absolute_difference"],
         limits["parameter_max"], summary["gates"]["parameter_maximum"]),
        ("Parameter RMS", parameters["rms_difference"],
         limits["parameter_rms"], summary["gates"]["parameter_rms"]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fp32 = json.loads(args.fp32.read_text(encoding="utf-8"))
    bf16 = json.loads(args.bf16.read_text(encoding="utf-8"))
    if fp32["status"] != "pass" or bf16["status"] != "precision_mismatch":
        raise RuntimeError("unexpected complete-training audit decision")

    width, height = 1440, 850
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="title description">',
        '<title id="title">Qwen3 complete one-step training alignment</title>',
        '<desc id="description">FP32 passes every complete-model gradient and '
        'parameter gate. BF16 fails gradient maximum and parameter RMS.</desc>',
        f'<rect width="{width}" height="{height}" fill="#f7f9fc"/>',
        '<text x="720" y="48" text-anchor="middle" '
        'font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" '
        'fill="#172033">Qwen3 Full Training Alignment: FP32 Pass · BF16 Rejected</text>',
        '<text x="720" y="80" text-anchor="middle" '
        'font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">'
        '310 independent Tensors · 596,049,920 values per state · gradients before AdamW + parameters after one step</text>',
        '<rect x="50" y="112" width="650" height="350" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<rect x="740" y="112" width="650" height="350" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="80" y="154" font-family="Inter,Arial,sans-serif" font-size="22" font-weight="700" fill="#198754">FP32 · PASS</text>',
        '<text x="770" y="154" font-family="Inter,Arial,sans-serif" font-size="22" font-weight="700" fill="#c0392b">BF16 · REJECT</text>',
    ]
    for panel_x, summary in ((80, fp32), (770, bf16)):
        for index, (label, value, limit, passed) in enumerate(metric_rows(summary)):
            y = 205 + index * 58
            color = "#198754" if passed else "#c0392b"
            symbol = "PASS" if passed else "FAIL"
            lines.extend([
                f'<text x="{panel_x}" y="{y}" font-family="Inter,Arial,sans-serif" font-size="16" fill="#172033">{label}</text>',
                f'<text x="{panel_x + 205}" y="{y}" font-family="ui-monospace,SFMono-Regular,monospace" font-size="15" fill="#172033">{value:.3e}</text>',
                f'<text x="{panel_x + 350}" y="{y}" font-family="ui-monospace,SFMono-Regular,monospace" font-size="15" fill="#5b6474">limit {limit:.1e}</text>',
                f'<rect x="{panel_x + 500}" y="{y - 20}" width="92" height="28" rx="14" fill="{color}" opacity="0.13"/>',
                f'<text x="{panel_x + 546}" y="{y}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" font-weight="700" fill="{color}">{symbol}</text>',
            ])
    lines.extend([
        '<text x="80" y="430" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">Worst gradient: token_embedding.weight · 5.746e-4</text>',
        '<text x="770" y="430" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">Worst gradient: token_embedding.weight · 3.641e-1</text>',
        '<rect x="50" y="495" width="1340" height="285" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="80" y="535" font-family="Inter,Arial,sans-serif" font-size="21" font-weight="700" fill="#172033">BF16 gradient Max by parameter family</text>',
        '<text x="1120" y="535" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">red line = fixed 5e-2 gate</text>',
    ])
    families = bf16["gradient_families"]
    maximum_ratio = 8.0
    chart_x, chart_width = 315, 985
    gate_x = chart_x + chart_width / maximum_ratio
    lines.append(
        f'<line x1="{gate_x:.1f}" y1="552" x2="{gate_x:.1f}" y2="758" '
        'stroke="#c0392b" stroke-width="2" stroke-dasharray="5 4"/>')
    for index, family in enumerate(families):
        y = 570 + index * 22
        value = family["maximum_absolute_difference"]
        ratio = value / bf16["limits"]["gradient_max"]
        bar_width = chart_width * min(ratio, maximum_ratio) / maximum_ratio
        color = "#c0392b" if ratio > 1.0 else "#4776c5"
        label = html.escape(family["family"].replace("_", " "))
        lines.extend([
            f'<text x="80" y="{y + 12}" font-family="Inter,Arial,sans-serif" font-size="14" fill="#172033">{label}</text>',
            f'<rect x="{chart_x}" y="{y}" width="{bar_width:.1f}" height="15" rx="4" fill="{color}" opacity="0.86"/>',
            f'<text x="1320" y="{y + 12}" text-anchor="end" font-family="ui-monospace,SFMono-Regular,monospace" font-size="13" fill="#172033">{value:.3e} · {ratio:.2f}× gate</text>',
        ])
    lines.extend([
        '<text x="720" y="822" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">Source checkpoint: 311 stored Tensors; tied lm_head reuses token embedding, so runtime exports 310 once · temporary 9.54 GB exports removed</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
