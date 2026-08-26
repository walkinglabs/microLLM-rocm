#!/usr/bin/env python3
"""Render the C++ PyTorch Custom Op Softmax inference-gate result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def groups(path: Path) -> dict[tuple[str, int], dict]:
    data = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    return {(row["dtype"], row["width"]): row for row in data["groups"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = groups(args.baseline)
    candidate = groups(args.candidate)
    keys = sorted(candidate)

    width, height = 1400, 720
    left, right, top, bottom = 100, 1010, 125, 575
    minimum, maximum = 0.45, 1.10

    def y(value: float) -> float:
        return bottom - (value - minimum) / (maximum - minimum) * (bottom - top)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f7f9fc"/>',
        '<text x="700" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">C++ Custom Op Reaches Model-Width Parity, Not Wide Parity</text>',
        '<text x="700" y="79" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">native Torch / microLLM Custom Op Event ratio · six MI300X processes</text>',
    ]
    for tick in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1):
        position = y(tick)
        stroke = "#758195" if tick == 1.0 else "#dbe1ea"
        lines.append(
            f'<line x1="{left}" y1="{position:.1f}" x2="{right}" y2="{position:.1f}" stroke="{stroke}"/>')
        lines.append(
            f'<text x="88" y="{position + 5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">{tick:.1f}×</text>')
    spacing = (right - left) / len(keys)
    for index, key in enumerate(keys):
        center = left + spacing * (index + 0.5)
        value = candidate[key]["event_speedup_median"]
        position = y(value)
        color = "#198754" if value >= 0.98 else "#e28b22"
        lines.append(
            f'<rect x="{center - 27:.1f}" y="{position:.1f}" width="54" height="{bottom - position:.1f}" rx="7" fill="{color}"/>')
        lines.append(
            f'<text x="{center:.1f}" y="{position - 8:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" font-weight="700" fill="#172033">{value:.3f}</text>')
        lines.append(
            f'<text x="{center:.1f}" y="604" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" fill="#172033" transform="rotate(38 {center:.1f} 604)">{key[0]} w{key[1]}</text>')

    fp16 = ("fp16", 4096)
    inference_gain = (baseline[fp16]["custom_event_ms_median"] /
                      candidate[fp16]["custom_event_ms_median"])
    lines.extend([
        '<rect x="1050" y="125" width="305" height="450" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="1080" y="168" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Decision</text>',
        '<text x="1080" y="216" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ 10 / 10 precision rows</text>',
        '<text x="1080" y="249" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ peak equals native in all rows</text>',
        f'<text x="1080" y="299" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">no-grad gate = {inference_gain:.3f}× wide</text>',
        '<text x="1080" y="332" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">w1024 FP16/BF16 = 1.026×/0.993×</text>',
        '<text x="1080" y="382" font-family="Inter,Arial,sans-serif" font-size="16" fill="#e28b22">w4096 FP16/BF16 = 0.795×/0.529×</text>',
        '<text x="1080" y="429" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">functional adapter retained</text>',
        '<text x="1080" y="460" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">not a universal speed claim</text>',
        '<text x="1080" y="507" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">next: explicit caller-owned schema</text>',
        '<text x="700" y="680" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">Keep integration · keep the wide counterexample visible</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
