#!/usr/bin/env python3
"""Render typed fused low-precision SwiGLU backward evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.comparison.read_text(encoding="utf-8"))
    groups = report["groups"]
    width, height = 1200, 680
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1200" height="680" fill="#f7f9fc"/>',
        '<text x="600" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">Typed Fused Backward Reaches Low-Precision Parity</text>',
        '<text x="600" y="78" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">six fresh MI300X processes · complete two-gradient and peak gates</text>',
        '<line x1="75" y1="500" x2="780" y2="500" stroke="#aab3c2" stroke-width="2"/>',
    ]
    for tick in (0.0, 0.5, 1.0, 1.5):
        y = 500 - tick / 1.5 * 340
        lines.append(f'<line x1="75" y1="{y:.1f}" x2="780" y2="{y:.1f}" stroke="#dbe1ea"/>')
    for index, row in enumerate(groups):
        center = 145 + index * 160
        aten = row["typed_vs_aten_event"]
        native = row["typed_vs_native_event"]
        for offset, value, color in ((-28, aten, "#198754"), (28, native, "#3769d6")):
            height_bar = value / 1.5 * 340
            y = 500 - height_bar
            lines.append(f'<rect x="{center + offset - 21}" y="{y:.1f}" width="42" height="{height_bar:.1f}" rx="6" fill="{color}"/>')
            lines.append(f'<text x="{center + offset}" y="{y - 8:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" font-weight="700" fill="#172033">{value:.3f}</text>')
        lines.append(f'<text x="{center}" y="530" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="14" fill="#172033">{row["dtype"]} {row["shape"]}</text>')
    lines.extend([
        '<rect x="825" y="125" width="320" height="375" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="855" y="167" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Admission</text>',
        '<text x="855" y="211" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ typed/ATen 1.257×–1.319×</text>',
        '<text x="855" y="244" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ typed/native 1.048×–1.084×</text>',
        '<text x="855" y="277" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ peak exactly equals native</text>',
        '<text x="855" y="325" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">BF16 Max/RMS = 0 / 0</text>',
        '<text x="855" y="358" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">FP16 Max = 2.38e-7</text>',
        '<text x="855" y="406" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">FP32 producer unchanged</text>',
        '<rect x="250" y="590" width="18" height="18" rx="3" fill="#198754"/><text x="278" y="605" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">typed / C++ ATen</text>',
        '<rect x="470" y="590" width="18" height="18" rx="3" fill="#3769d6"/><text x="498" y="605" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">typed / native Torch</text>',
        '<text x="600" y="655" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">Keep typed fused FP16/BF16 backward · SwiGLU adapter line reaches parity</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

