#!/usr/bin/env python3
"""Render the width-4096 typed Softmax shared-exp-cache result."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def medians(path: Path) -> dict[str, dict[str, float]]:
    records = []
    for line in (path / "raw.jsonl").read_text(encoding="utf-8").splitlines():
        records.extend(json.loads(line)["records"])
    output = {}
    for dtype in ("bf16", "fp16"):
        selected = [row for row in records
                    if row["dtype"] == dtype and row["width"] == 4096]
        output[dtype] = {
            "torch_event_us": statistics.median(
                row["torch_event_ms"] for row in selected) * 1000.0,
            "micro_event_us": statistics.median(
                row["microllm_event_ms"] for row in selected) * 1000.0,
            "micro_wall_us": statistics.median(
                row["microllm_wall_ms"] for row in selected) * 1000.0,
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = medians(args.baseline)
    candidate = medians(args.candidate)

    width, height = 1320, 700
    plot_left, plot_right = 110, 880
    plot_top, plot_bottom = 125, 565
    maximum = 12.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f7f9fc"/>',
        '<text x="660" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">Caching FP32 Exponentials Helps Wide Typed Softmax</text>',
        '<text x="660" y="79" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">width4096 Event time · lower is better · six fresh MI300X processes</text>',
    ]
    for tick in (0, 3, 6, 9, 12):
        y = plot_bottom - tick / maximum * (plot_bottom - plot_top)
        lines.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" stroke="#dbe1ea"/>')
        lines.append(
            f'<text x="96" y="{y + 5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">{tick} μs</text>')

    colors = {"torch": "#4776c5", "block": "#aab3c2", "cache": "#198754"}
    for group_index, dtype in enumerate(("bf16", "fp16")):
        center = 310 + group_index * 390
        values = (
            ("torch", candidate[dtype]["torch_event_us"]),
            ("block", baseline[dtype]["micro_event_us"]),
            ("cache", candidate[dtype]["micro_event_us"]),
        )
        for index, (label, value) in enumerate(values):
            x = center + (index - 1) * 92
            y = plot_bottom - value / maximum * (plot_bottom - plot_top)
            lines.append(
                f'<rect x="{x - 29}" y="{y:.1f}" width="58" height="{plot_bottom - y:.1f}" rx="7" fill="{colors[label]}"/>')
            lines.append(
                f'<text x="{x}" y="{y - 9:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" font-weight="700" fill="#172033">{value:.2f}</text>')
            lines.append(
                f'<text x="{x}" y="592" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" fill="#5b6474">{label}</text>')
        lines.append(
            f'<text x="{center}" y="628" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">{dtype.upper()}</text>')

    bf16_event = baseline["bf16"]["micro_event_us"] / candidate["bf16"]["micro_event_us"]
    fp16_event = baseline["fp16"]["micro_event_us"] / candidate["fp16"]["micro_event_us"]
    bf16_wall = baseline["bf16"]["micro_wall_us"] / candidate["bf16"]["micro_wall_us"]
    fp16_wall = baseline["fp16"]["micro_wall_us"] / candidate["fp16"]["micro_wall_us"]
    lines.extend([
        '<rect x="925" y="125" width="345" height="440" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="955" y="168" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Evidence</text>',
        '<text x="955" y="214" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ 10 / 10 precision + pointers</text>',
        '<text x="955" y="247" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ zero Tensor / allocator bytes</text>',
        f'<text x="955" y="298" font-family="Inter,Arial,sans-serif" font-size="16" font-weight="700" fill="#172033">BF16 Event / wall {bf16_event:.3f}× / {bf16_wall:.3f}×</text>',
        f'<text x="955" y="331" font-family="Inter,Arial,sans-serif" font-size="16" font-weight="700" fill="#172033">FP16 Event / wall {fp16_event:.3f}× / {fp16_wall:.3f}×</text>',
        '<text x="955" y="382" font-family="Inter,Arial,sans-serif" font-size="16" fill="#e28b22">△ still 0.550× / 0.576× Torch</text>',
        '<text x="955" y="429" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">cache only width 2048–8192</text>',
        '<text x="955" y="456" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">outside range keeps no-cache block</text>',
        '<text x="955" y="503" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">next: replace 8 full-block barriers</text>',
        '<text x="955" y="528" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">with wave-level reduction</text>',
        '<text x="660" y="670" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">Keep bounded cache · wide-row parity remains open</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
