#!/usr/bin/env python3
"""Render caller-owned native/custom Softmax evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    groups = sorted(data["groups"], key=lambda row: (row["dtype"], row["width"]))
    width, height = 1360, 700
    left, right, top, bottom = 100, 980, 125, 565
    minimum, maximum = 0.4, 1.2

    def y(value: float) -> float:
        return bottom - (value - minimum) / (maximum - minimum) * (bottom - top)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f7f9fc"/>',
        '<text x="680" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">Caller-Owned Softmax Wins at Model Width, Not Every Width</text>',
        '<text x="680" y="79" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">native out / microLLM Custom out Event ratio · zero peak extra on both sides</text>',
    ]
    for tick in (0.4, 0.6, 0.8, 1.0, 1.2):
        position = y(tick)
        stroke = "#758195" if tick == 1.0 else "#dbe1ea"
        lines.append(
            f'<line x1="{left}" y1="{position:.1f}" x2="{right}" y2="{position:.1f}" stroke="{stroke}"/>')
        lines.append(
            f'<text x="88" y="{position + 5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">{tick:.1f}×</text>')
    spacing = (right - left) / len(groups)
    for index, row in enumerate(groups):
        center = left + spacing * (index + 0.5)
        value = row["event_speedup_median"]
        position = y(value)
        color = "#198754" if value >= 1.0 else "#e28b22"
        lines.append(
            f'<rect x="{center - 26:.1f}" y="{position:.1f}" width="52" height="{bottom - position:.1f}" rx="7" fill="{color}"/>')
        lines.append(
            f'<text x="{center:.1f}" y="{position - 8:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" font-weight="700" fill="#172033">{value:.3f}</text>')
        lines.append(
            f'<text x="{center:.1f}" y="594" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" fill="#172033" transform="rotate(38 {center:.1f} 594)">{row["dtype"]} w{row["width"]}</text>')
    lines.extend([
        '<rect x="1020" y="125" width="295" height="440" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="1050" y="168" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Contract</text>',
        '<text x="1050" y="216" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ 10 / 10 precision rows</text>',
        '<text x="1050" y="249" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ returned pointer = caller</text>',
        '<text x="1050" y="282" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ native/custom peak = 0</text>',
        '<text x="1050" y="332" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">w1024 FP16/BF16 1.116×/1.087×</text>',
        '<text x="1050" y="382" font-family="Inter,Arial,sans-serif" font-size="16" fill="#e28b22">w4096 FP16/BF16 0.813×/0.467×</text>',
        '<text x="1050" y="429" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">inference-only mutation schema</text>',
        '<text x="1050" y="460" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">functional op owns Autograd</text>',
        '<text x="1050" y="507" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">keep API · keep wide failure</text>',
        '<text x="680" y="665" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">Explicit ownership removes allocation ambiguity</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
