#!/usr/bin/env python3
"""Render the rejected Softmax-out Autograd fallthrough."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path) -> dict[tuple[str, int], dict]:
    data = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    return {(row["dtype"], row["width"]): row for row in data["groups"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before, after = rows(args.baseline), rows(args.candidate)
    width, height = 1060, 620
    left, right, top, bottom = 105, 650, 125, 500
    minimum, maximum, gate = 0.98, 1.05, 1.05
    gains = {
        dtype: before[(dtype, 4096)]["custom_event_ms_median"] /
               after[(dtype, 4096)]["custom_event_ms_median"]
        for dtype in ("bf16", "fp16")
    }

    def y(value: float) -> float:
        return bottom - (value - minimum) / (maximum - minimum) * (bottom - top)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f7f9fc"/>',
        '<text x="530" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">Autograd Fallthrough Is Not the Remaining Hotspot</text>',
        '<text x="530" y="79" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">explicit Autograd kernel / fallthrough Custom Event · width4096</text>',
    ]
    for tick in (0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05):
        position = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{position:.1f}" x2="{right}" y2="{position:.1f}" stroke="#dbe1ea"/>')
        lines.append(
            f'<text x="92" y="{position + 5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">{tick:.2f}×</text>')
    gate_y = y(gate)
    lines.append(
        f'<line x1="{left}" y1="{gate_y:.1f}" x2="{right}" y2="{gate_y:.1f}" stroke="#b42335" stroke-width="2" stroke-dasharray="8 7"/>')
    for index, dtype in enumerate(("bf16", "fp16")):
        center = 250 + index * 260
        value = gains[dtype]
        position = y(value)
        lines.append(
            f'<rect x="{center - 52}" y="{position:.1f}" width="104" height="{bottom - position:.1f}" rx="9" fill="#b42335"/>')
        lines.append(
            f'<text x="{center}" y="{position - 10:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">{value:.3f}×</text>')
        lines.append(
            f'<text x="{center}" y="535" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="17" fill="#172033">{dtype.upper()}</text>')
    lines.extend([
        '<rect x="700" y="125" width="310" height="375" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="730" y="168" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Decision</text>',
        '<text x="730" y="216" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ precision / pointer / peak</text>',
        '<text x="730" y="266" font-family="Inter,Arial,sans-serif" font-size="16" fill="#b42335">FP16 gain ≈ 1.008×</text>',
        '<text x="730" y="299" font-family="Inter,Arial,sans-serif" font-size="16" fill="#b42335">BF16 gain ≈ 0.998×</text>',
        '<text x="730" y="349" font-family="Inter,Arial,sans-serif" font-size="16" font-weight="700" fill="#b42335">fallthrough removed</text>',
        '<text x="730" y="399" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">explicit rejection stays central</text>',
        '<text x="730" y="432" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">adapter local line closed</text>',
        '<text x="530" y="590" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">Correct architecture changes still need a speed gate</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
