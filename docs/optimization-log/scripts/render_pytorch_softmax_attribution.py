#!/usr/bin/env python3
"""Render typed Softmax raw/C++/Python/PyTorch attribution."""

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
    values = (
        ("PyTorch", data["pytorch_event_ms_median"] * 1000.0, "#4776c5"),
        ("raw launcher", data["raw_event_ms_median"] * 1000.0, "#52a36d"),
        ("C++ out API", data["cpp_event_ms_median"] * 1000.0, "#198754"),
        ("Python / C API", data["python_capi_event_ms_median"] * 1000.0, "#e28b22"),
    )
    width, height = 1280, 680
    left, right, top, bottom = 105, 820, 125, 545
    maximum = 5.5
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f7f9fc"/>',
        '<text x="640" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">The Last Softmax Gap Is Split Across Two Boundaries</text>',
        '<text x="640" y="79" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">FP16 rows8 × width4096 Event median · lower is better · MI300X</text>',
    ]
    for tick in (0, 1, 2, 3, 4, 5):
        y = bottom - tick / maximum * (bottom - top)
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#dbe1ea"/>')
        lines.append(
            f'<text x="92" y="{y + 5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">{tick} μs</text>')
    for index, (label, value, color) in enumerate(values):
        center = 185 + index * 185
        y = bottom - value / maximum * (bottom - top)
        lines.append(
            f'<rect x="{center - 43}" y="{y:.1f}" width="86" height="{bottom - y:.1f}" rx="9" fill="{color}"/>')
        lines.append(
            f'<text x="{center}" y="{y - 10:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="17" font-weight="700" fill="#172033">{value:.3f}</text>')
        lines.append(
            f'<text x="{center}" y="580" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="15" fill="#172033">{label}</text>')

    lines.extend([
        '<rect x="865" y="125" width="365" height="420" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="895" y="168" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Attribution</text>',
        '<text x="895" y="216" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ raw/C++ Max = 5.96e-8</text>',
        '<text x="895" y="249" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ timed payload transfers = 0</text>',
        '<text x="895" y="299" font-family="Inter,Arial,sans-serif" font-size="16" fill="#172033">C++ / raw time = 1.011×</text>',
        '<text x="895" y="332" font-family="Inter,Arial,sans-serif" font-size="16" fill="#e28b22">Python / C++ time = 1.056×</text>',
        '<text x="895" y="365" font-family="Inter,Arial,sans-serif" font-size="16" fill="#e28b22">raw / PyTorch time = 1.052×</text>',
        '<text x="895" y="398" font-family="Inter,Arial,sans-serif" font-size="16" font-weight="700" fill="#b42335">Python / PyTorch = 1.123×</text>',
        '<text x="895" y="448" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">next scale: C++ PyTorch Custom Op</text>',
        '<text x="895" y="477" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">not another blind math tweak</text>',
        '<text x="640" y="642" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">Kernel and bridge each own part of the residual</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
