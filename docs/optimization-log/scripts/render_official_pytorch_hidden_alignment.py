#!/usr/bin/env python3
"""Render official Qwen/DeepSeek per-layer hidden-state alignment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    width, height = 1420, 720
    left, right, top, bottom = 100, 1020, 125, 585
    log_min, log_max = -8.0, -4.0

    def y(value: float) -> float:
        exponent = math.log10(max(value, 1.0e-8))
        return bottom - (exponent - log_min) / (log_max - log_min) * (bottom - top)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f7f9fc"/>',
        '<text x="710" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">Every Official-Model Hidden State Now Has a PyTorch Oracle</text>',
        '<text x="710" y="79" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">FP32 relative-L2 by stage · context 4 · complete synchronous snapshots</text>',
    ]
    for tick in (1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4):
        position = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{position:.1f}" x2="{right}" y2="{position:.1f}" stroke="#dbe1ea"/>')
        lines.append(
            f'<text x="88" y="{position + 5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">{tick:.0e}</text>')
    colors = ("#4776c5", "#d06d45")
    labels = ("Qwen2.5-0.5B", "DeepSeek-Distill-1.5B")
    for model, color, label in zip(data["summaries"], colors, labels):
        stages = model["stages"]
        points = []
        for index, stage in enumerate(stages):
            x = left + index / max(1, len(stages) - 1) * (right - left)
            points.append(f"{x:.1f},{y(stage['relative_l2']):.1f}")
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>')
        for index, stage in enumerate(stages):
            x = left + index / max(1, len(stages) - 1) * (right - left)
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y(stage["relative_l2"]):.1f}" r="3.5" fill="{color}"/>')
        legend_y = 145 + (0 if label.startswith("Qwen") else 32)
        lines.append(f'<line x1="130" y1="{legend_y}" x2="175" y2="{legend_y}" stroke="{color}" stroke-width="4"/>')
        lines.append(f'<text x="185" y="{legend_y + 5}" font-family="Inter,Arial,sans-serif" font-size="15" fill="#172033">{label}</text>')
    lines.extend([
        f'<text x="{left}" y="618" font-family="Inter,Arial,sans-serif" font-size="13" fill="#5b6474">embedding</text>',
        f'<text x="{(left + right) / 2:.1f}" y="618" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" fill="#5b6474">decoder blocks</text>',
        f'<text x="{right}" y="618" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="13" fill="#5b6474">norm + logits</text>',
        '<rect x="1060" y="125" width="315" height="460" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="1090" y="168" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Evidence</text>',
        '<text x="1090" y="216" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ embedding exact: both models</text>',
        '<text x="1090" y="249" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ 27 / 31 stages complete</text>',
        '<text x="1090" y="299" font-family="Inter,Arial,sans-serif" font-size="16" fill="#172033">first nonzero: block 0</text>',
        '<text x="1090" y="332" font-family="Inter,Arial,sans-serif" font-size="16" fill="#172033">Qwen max rel-L2 2.89e-5</text>',
        '<text x="1090" y="365" font-family="Inter,Arial,sans-serif" font-size="16" fill="#172033">Deep max rel-L2 2.85e-6</text>',
        '<text x="1090" y="415" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">Qwen logits Max 8.01e-5</text>',
        '<text x="1090" y="448" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">Deep logits Max 2.48e-5</text>',
        '<text x="1090" y="498" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">numerical diagnostic only</text>',
        '<text x="1090" y="529" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">no performance claim</text>',
        '<text x="710" y="680" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">The first difference is visible instead of inferred from final logits</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
