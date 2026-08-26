#!/usr/bin/env python3
"""Render SwiGLU scalar-seed time and peak deltas."""

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
    width, height = 1220, 680
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1220" height="680" fill="#f7f9fc"/>',
        '<text x="610" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">Zero-Stride Scalar Seed Removes the Materialization</text>',
        '<text x="610" y="78" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">FP32 SwiGLU sum backward · six fresh processes before and after</text>',
        '<rect x="70" y="125" width="510" height="420" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="325" y="165" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Candidate / baseline Event</text>',
        '<rect x="640" y="125" width="510" height="420" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="895" y="165" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Measured temporary peak</text>',
    ]
    for index, row in enumerate(groups):
        center = 235 + index * 180
        speed = row["candidate_vs_baseline_event"]
        height_bar = speed / 1.3 * 270
        y = 475 - height_bar
        lines.append(f'<rect x="{center - 45}" y="{y:.1f}" width="90" height="{height_bar:.1f}" rx="8" fill="#198754"/>')
        lines.append(f'<text x="{center}" y="{y - 10:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">{speed:.3f}×</text>')
        lines.append(f'<text x="{center}" y="510" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">{row["shape"]} · {row["elements"]}</text>')

        peak_center = 805 + index * 180
        before_mib = row["baseline_peak_extra_bytes"] / (1024 * 1024)
        after_kib = row["candidate_peak_extra_bytes"] / 1024
        before_height = min(270, before_mib / 4.1 * 270)
        after_height = max(3, after_kib / 4096 * 270)
        lines.append(f'<rect x="{peak_center - 50}" y="{475 - before_height:.1f}" width="42" height="{before_height:.1f}" rx="5" fill="#b42335"/>')
        lines.append(f'<rect x="{peak_center + 8}" y="{475 - after_height:.1f}" width="42" height="{after_height:.1f}" rx="5" fill="#3769d6"/>')
        lines.append(f'<text x="{peak_center - 29}" y="{465 - before_height:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" fill="#172033">{before_mib:.3f} MiB</text>')
        lines.append(f'<text x="{peak_center + 29}" y="{465 - after_height:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" fill="#172033">{after_kib:.1f} KiB</text>')
        lines.append(f'<text x="{peak_center}" y="510" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">{row["shape"]}</text>')
    lines.extend([
        '<text x="610" y="590" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="17" fill="#198754">Max error 4.77e-7 · peak reduction 99.42%–99.96% · Event +8.1%–16.4%</text>',
        '<text x="610" y="625" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#b42335">Still only 0.773×–0.781× native Torch: materialization was real, but not the whole gap</text>',
        '<text x="610" y="658" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">Keep exact zero-stride route · preserve general-gradient fallback</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

