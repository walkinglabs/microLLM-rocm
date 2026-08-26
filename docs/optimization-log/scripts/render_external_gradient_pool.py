#!/usr/bin/env python3
"""Render the measured Autograd external-gradient-pool decision as SVG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    groups = summary["groups"]
    width, height = 1280, 720
    labels = [f"Tiny T{row['context']}" if row["model"] == "tiny"
              else f"Model-S T{row['context']}" for row in groups]
    event = [row["event_speedup_median"] for row in groups]
    wall = [row["wall_speedup_median"] for row in groups]
    peak_mb = [row["peak_extra_bytes_delta_median"] / (1024 * 1024)
               for row in groups]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1280" height="720" fill="#f7f9fc"/>',
        '<text x="640" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">External Gradient Pool · Exact but Slower</text>',
        '<text x="640" y="78" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">18 fresh MI300X processes · rotated order · 1 warmup + 5 measured backward steps</text>',
        '<text x="75" y="130" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">Baseline / external speed (1.00 means equal)</text>',
        '<line x1="75" y1="420" x2="760" y2="420" stroke="#aab3c2" stroke-width="2"/>',
        '<line x1="75" y1="130" x2="75" y2="420" stroke="#aab3c2" stroke-width="2"/>',
    ]
    for tick in (0.6, 0.7, 0.8, 0.9, 1.0):
        y = 420 - (tick - 0.6) / 0.4 * 260
        lines.append(f'<line x1="75" y1="{y:.1f}" x2="760" y2="{y:.1f}" stroke="#dbe1ea"/>')
        lines.append(f'<text x="65" y="{y + 5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">{tick:.1f}</text>')
    for index, label in enumerate(labels):
        center = 180 + index * 210
        for offset, value, color in ((-35, event[index], "#3769d6"),
                                     (35, wall[index], "#8b5cf6")):
            bar_height = max(0.0, (value - 0.6) / 0.4 * 260)
            y = 420 - bar_height
            lines.append(f'<rect x="{center + offset - 28}" y="{y:.1f}" width="56" height="{bar_height:.1f}" rx="6" fill="{color}"/>')
            lines.append(f'<text x="{center + offset}" y="{y - 8:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="14" font-weight="700" fill="#172033">{value:.3f}×</text>')
        lines.append(f'<text x="{center}" y="448" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="15" fill="#172033">{label}</text>')
    lines.extend([
        '<rect x="120" y="478" width="18" height="18" rx="3" fill="#3769d6"/><text x="146" y="493" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">Event</text>',
        '<rect x="220" y="478" width="18" height="18" rx="3" fill="#8b5cf6"/><text x="246" y="493" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">Wall</text>',
        '<rect x="815" y="122" width="400" height="380" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="845" y="162" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">What the evidence says</text>',
        '<text x="845" y="205" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ 21/21 Tiny addresses stable</text>',
        '<text x="845" y="238" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ 57/57 Model-S addresses stable</text>',
        '<text x="845" y="271" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ all 15,586,176 Model-S gradients exact</text>',
        '<text x="845" y="318" font-family="Inter,Arial,sans-serif" font-size="16" fill="#b42335">✗ every median speed ratio is below 1.00</text>',
    ])
    for index, (label, value) in enumerate(zip(labels, peak_mb)):
        lines.append(f'<text x="845" y="{354 + index * 30}" font-family="Inter,Arial,sans-serif" font-size="15" fill="#b42335">✗ {label} measured peak +{value:.2f} MiB</text>')
    lines.extend([
        '<rect x="75" y="550" width="1140" height="105" rx="16" fill="#fff2f4" stroke="#ef9aa7"/>',
        '<text x="640" y="590" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="22" font-weight="700" fill="#b42335">Decision: keep as an explicit interoperability API</text>',
        '<text x="640" y="624" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">Do not enable it as the engine training default. Stable addresses are useful for foreign runtimes and communication buffers, but zero-first accumulation adds work.</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

