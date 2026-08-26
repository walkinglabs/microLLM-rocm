#!/usr/bin/env python3
"""Render representative PyTorch ROCm Custom Op speed ratios."""

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

    def select(kind: str, operation: str, dtype: str, shape: str) -> dict:
        return next(row for row in groups if row["kind"] == kind and
                    row["operation"] == operation and row["dtype"] == dtype and
                    row["shape"] == shape)

    rows = [
        ("FP32 add 16M", select("forward", "add", "fp32", "bandwidth")),
        ("FP32 multiply 16M", select("forward", "multiply", "fp32", "bandwidth")),
        ("FP16 add 16M", select("forward", "add", "fp16", "bandwidth")),
        ("FP16 multiply 16M", select("forward", "multiply", "fp16", "bandwidth")),
        ("BF16 add 16M", select("forward", "add", "bf16", "bandwidth")),
        ("BF16 multiply 16M", select("forward", "multiply", "bf16", "bandwidth")),
        ("FP32 branch F+B 1M", select(
            "forward_backward", "add_multiply_branch", "fp32", "large")),
    ]
    width, height = 1420, 760
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1420" height="760" fill="#f7f9fc"/>',
        '<text x="710" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">PyTorch ROCm Custom Ops · Correct, Not Yet Faster</text>',
        '<text x="710" y="78" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">6 fresh MI300X processes · Torch/microLLM ratio · 5 warmups + 25 measured calls</text>',
        '<line x1="90" y1="520" x2="1020" y2="520" stroke="#aab3c2" stroke-width="2"/>',
        '<line x1="90" y1="130" x2="90" y2="520" stroke="#aab3c2" stroke-width="2"/>',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = 520 - tick * 360
        lines.append(f'<line x1="90" y1="{y:.1f}" x2="1020" y2="{y:.1f}" stroke="#dbe1ea"/>')
        lines.append(f'<text x="78" y="{y + 5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">{tick:.2f}</text>')
    colors = ["#3769d6", "#3769d6", "#13a37f", "#13a37f",
              "#8b5cf6", "#8b5cf6", "#d97706"]
    for index, ((label, row), color) in enumerate(zip(rows, colors)):
        center = 155 + index * 125
        event = row["event_speedup_median"]
        wall = row["wall_speedup_median"]
        for offset, value, opacity in ((-22, event, 1.0), (22, wall, 0.55)):
            y = 520 - value * 360
            lines.append(f'<rect x="{center + offset - 17}" y="{y:.1f}" width="34" height="{value * 360:.1f}" rx="5" fill="{color}" opacity="{opacity}"/>')
        lines.append(f'<text x="{center}" y="{520 - max(event, wall) * 360 - 9:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" font-weight="700" fill="#172033">{event:.3f}×</text>')
        lines.append(f'<text x="{center}" y="550" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" fill="#172033" transform="rotate(28 {center} 550)">{label}</text>')
    lines.extend([
        '<rect x="1070" y="125" width="300" height="390" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="1100" y="167" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Contract evidence</text>',
        '<text x="1100" y="211" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ 20 / 20 cases exact</text>',
        '<text x="1100" y="244" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ FP32 / FP16 / BF16</text>',
        '<text x="1100" y="277" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ current HIP Stream</text>',
        '<text x="1100" y="310" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ Autograd branch</text>',
        '<text x="1100" y="343" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ torch.compile Meta path</text>',
        '<text x="1100" y="392" font-family="Inter,Arial,sans-serif" font-size="16" fill="#b42335">✗ 0 / 20 speed medians ≥ 1</text>',
        '<text x="1100" y="425" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">FP32 16M reaches 0.933–0.973×</text>',
        '<text x="1100" y="458" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">low precision needs vectorized kernels</text>',
        '<rect x="150" y="660" width="18" height="18" rx="3" fill="#3769d6"/><text x="177" y="675" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">solid = Event</text>',
        '<rect x="300" y="660" width="18" height="18" rx="3" fill="#3769d6" opacity="0.55"/><text x="327" y="675" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">light = wall</text>',
        '<text x="710" y="718" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#b42335">Decision: ship the optional integration; do not claim operator speedup</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

