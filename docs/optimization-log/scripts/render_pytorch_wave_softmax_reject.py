#!/usr/bin/env python3
"""Render the rejected broad wave-reduction typed Softmax experiment."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def medians(path: Path, field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {"bf16": [], "fp16": []}
    for line in (path / "raw.jsonl").read_text(encoding="utf-8").splitlines():
        for row in json.loads(line)["records"]:
            if row["width"] == 4096:
                grouped[row["dtype"]].append(row[field])
    return {dtype: statistics.median(values)
            for dtype, values in grouped.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_event = medians(args.baseline, "microllm_event_ms")
    candidate_event = medians(args.candidate, "microllm_event_ms")
    baseline_wall = medians(args.baseline, "microllm_wall_ms")
    candidate_wall = medians(args.candidate, "microllm_wall_ms")
    gains = {
        dtype: {
            "event": baseline_event[dtype] / candidate_event[dtype],
            "wall": baseline_wall[dtype] / candidate_wall[dtype],
        }
        for dtype in ("bf16", "fp16")
    }

    width, height = 1260, 650
    left, right, top, bottom = 110, 820, 125, 525
    minimum, maximum, gate = 1.0, 1.09, 1.05

    def y(value: float) -> float:
        return bottom - (value - minimum) / (maximum - minimum) * (bottom - top)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f7f9fc"/>',
        '<text x="630" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">Broad Wave Reduction Misses the Two-DType Gate</text>',
        '<text x="630" y="79" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">cached baseline / wave candidate · width4096 · six fresh MI300X processes</text>',
    ]
    for tick in (1.00, 1.02, 1.04, 1.06, 1.08):
        position = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{position:.1f}" x2="{right}" y2="{position:.1f}" stroke="#dbe1ea"/>')
        lines.append(
            f'<text x="96" y="{position + 5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">{tick:.2f}×</text>')
    gate_y = y(gate)
    lines.append(
        f'<line x1="{left}" y1="{gate_y:.1f}" x2="{right}" y2="{gate_y:.1f}" stroke="#b42335" stroke-width="2" stroke-dasharray="8 7"/>')
    lines.append(
        f'<text x="{right - 4}" y="{gate_y - 8:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" font-weight="700" fill="#b42335">keep gate 1.05×</text>')

    for group_index, dtype in enumerate(("bf16", "fp16")):
        center = 285 + group_index * 350
        for index, field in enumerate(("event", "wall")):
            value = gains[dtype][field]
            x = center + (index - 0.5) * 92
            position = y(value)
            color = "#198754" if value >= gate else "#b42335"
            lines.append(
                f'<rect x="{x - 30:.1f}" y="{position:.1f}" width="60" height="{bottom - position:.1f}" rx="7" fill="{color}"/>')
            lines.append(
                f'<text x="{x:.1f}" y="{position - 9:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="14" font-weight="700" fill="#172033">{value:.3f}×</text>')
            lines.append(
                f'<text x="{x:.1f}" y="551" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" fill="#5b6474">{field}</text>')
        lines.append(
            f'<text x="{center}" y="587" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">{dtype.upper()}</text>')

    lines.extend([
        '<rect x="865" y="125" width="345" height="400" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="895" y="168" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Decision</text>',
        '<text x="895" y="216" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ precision / pointer / peak pass</text>',
        '<text x="895" y="261" font-family="Inter,Arial,sans-serif" font-size="16" fill="#b42335">✗ BF16 wall = 1.033×</text>',
        '<text x="895" y="294" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ FP16 Event/wall ≈ 1.071×</text>',
        '<text x="895" y="344" font-family="Inter,Arial,sans-serif" font-size="16" font-weight="700" fill="#b42335">broad candidate removed</text>',
        '<text x="895" y="391" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">do not average dtypes</text>',
        '<text x="895" y="424" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">cached tree remains default</text>',
        '<text x="895" y="471" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">next falsifiable scope:</text>',
        '<text x="895" y="497" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">FP16-only wave predicate</text>',
        '<text x="630" y="625" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">Reject broad route · retain failure evidence</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
