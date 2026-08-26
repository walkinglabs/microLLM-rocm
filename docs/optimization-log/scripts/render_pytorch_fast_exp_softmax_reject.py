#!/usr/bin/env python3
"""Render the rejected FP16 fast-exp typed Softmax experiment."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def median(path: Path, field: str) -> float:
    values = []
    for line in (path / "raw.jsonl").read_text(encoding="utf-8").splitlines():
        for row in json.loads(line)["records"]:
            if row["dtype"] == "fp16" and row["width"] == 4096:
                values.append(row[field])
    return statistics.median(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gains = {
        "Event": median(args.baseline, "microllm_event_ms") /
                 median(args.candidate, "microllm_event_ms"),
        "wall": median(args.baseline, "microllm_wall_ms") /
                median(args.candidate, "microllm_wall_ms"),
    }

    width, height = 1120, 620
    left, right, top, bottom = 110, 660, 125, 505
    minimum, maximum, gate = 1.0, 1.06, 1.05

    def y(value: float) -> float:
        return bottom - (value - minimum) / (maximum - minimum) * (bottom - top)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f7f9fc"/>',
        '<text x="560" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">Fast Exp Is Accurate but Below the Keep Gate</text>',
        '<text x="560" y="79" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">FP16 width4096 · retained wave / fast-exp candidate · six MI300X processes</text>',
    ]
    for tick in (1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06):
        position = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{position:.1f}" x2="{right}" y2="{position:.1f}" stroke="#dbe1ea"/>')
        lines.append(
            f'<text x="96" y="{position + 5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">{tick:.2f}×</text>')
    gate_y = y(gate)
    lines.append(
        f'<line x1="{left}" y1="{gate_y:.1f}" x2="{right}" y2="{gate_y:.1f}" stroke="#b42335" stroke-width="2" stroke-dasharray="8 7"/>')
    for index, (label, value) in enumerate(gains.items()):
        x = 275 + index * 220
        position = y(value)
        lines.append(
            f'<rect x="{x - 48}" y="{position:.1f}" width="96" height="{bottom - position:.1f}" rx="9" fill="#b42335"/>')
        lines.append(
            f'<text x="{x}" y="{position - 12:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">{value:.3f}×</text>')
        lines.append(
            f'<text x="{x}" y="540" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">{label}</text>')

    lines.extend([
        '<rect x="710" y="125" width="360" height="380" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="740" y="168" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Decision</text>',
        '<text x="740" y="216" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ 10 / 10 precision + resources</text>',
        '<text x="740" y="251" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ FP16 maximum error = 1.19e-7</text>',
        '<text x="740" y="301" font-family="Inter,Arial,sans-serif" font-size="16" fill="#b42335">✗ Event 1.045× &lt; 1.05×</text>',
        '<text x="740" y="334" font-family="Inter,Arial,sans-serif" font-size="16" fill="#b42335">✗ wall 1.034× &lt; 1.05×</text>',
        '<text x="740" y="384" font-family="Inter,Arial,sans-serif" font-size="16" font-weight="700" fill="#b42335">candidate removed</text>',
        '<text x="740" y="431" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">retained path remains expf</text>',
        '<text x="740" y="462" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">next: thread-count matrix</text>',
        '<text x="560" y="590" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">Do not trade approximation for a sub-threshold gain</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
