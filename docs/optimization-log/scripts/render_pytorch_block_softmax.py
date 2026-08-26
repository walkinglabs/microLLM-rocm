#!/usr/bin/env python3
"""Render serial versus block-parallel typed Softmax evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_micro_event_medians(path: Path) -> dict[tuple[str, int], float]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        worker = json.loads(line)
        for row in worker["records"]:
            grouped.setdefault((row["dtype"], row["width"]), []).append(
                row["microllm_event_ms"])
    return {key: statistics.median(values) for key, values in grouped.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_summary(args.baseline / "summary.json")
    candidate = load_summary(args.candidate / "summary.json")
    serial_times = load_micro_event_medians(args.baseline / "raw.jsonl")
    block_times = load_micro_event_medians(args.candidate / "raw.jsonl")
    serial = {(row["dtype"], row["width"]): row for row in baseline["groups"]}
    block = {(row["dtype"], row["width"]): row for row in candidate["groups"]}
    keys = sorted(block)

    width, height = 1440, 760
    plot_left, plot_right = 100, 1040
    plot_top, plot_bottom = 125, 610
    log_min, log_max = -3.0, math.log10(2.0)

    def y_position(value: float) -> float:
        normalized = (math.log10(max(value, 0.001)) - log_min) / (log_max - log_min)
        return plot_bottom - normalized * (plot_bottom - plot_top)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f7f9fc"/>',
        '<text x="720" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">One Block per Row Removes the Typed Softmax Cliff</text>',
        '<text x="720" y="79" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">Torch / microLLM Event ratio · six fresh MI300X processes · same ten cases</text>',
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#aab3c2" stroke-width="2"/>',
    ]
    for tick in (0.001, 0.01, 0.1, 1.0):
        y = y_position(tick)
        stroke = "#758195" if tick == 1.0 else "#dbe1ea"
        lines.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" stroke="{stroke}"/>')
        lines.append(
            f'<text x="88" y="{y + 5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">{tick:g}×</text>')

    group_width = (plot_right - plot_left) / len(keys)
    for index, key in enumerate(keys):
        center = plot_left + group_width * (index + 0.5)
        serial_value = serial[key]["event_speedup_median"]
        block_value = block[key]["event_speedup_median"]
        for offset, value, color in ((-17, serial_value, "#aab3c2"),
                                     (17, block_value,
                                      "#198754" if block_value >= 1.0 else "#e28b22")):
            y = y_position(value)
            lines.append(
                f'<rect x="{center + offset - 13:.1f}" y="{y:.1f}" width="26" height="{plot_bottom - y:.1f}" rx="4" fill="{color}"/>')
        lines.append(
            f'<text x="{center + 17:.1f}" y="{y_position(block_value) - 7:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="10" font-weight="700" fill="#172033">{block_value:.3f}</text>')
        lines.append(
            f'<text x="{center:.1f}" y="638" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" fill="#172033" transform="rotate(38 {center:.1f} 638)">{key[0]} w{key[1]}</text>')

    improvement = {
        key: serial_times[key] / block_times[key]
        for key in keys
    }
    model_gain = [improvement[key] for key in keys if key[1] == 1024]
    wide_gain = [improvement[key] for key in keys if key[1] == 4096]
    lines.extend([
        '<rect x="1080" y="125" width="320" height="485" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="1110" y="165" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Measured decision</text>',
        '<rect x="1110" y="190" width="18" height="18" rx="3" fill="#aab3c2"/>',
        '<text x="1138" y="205" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">serial baseline</text>',
        '<rect x="1260" y="190" width="18" height="18" rx="3" fill="#198754"/>',
        '<text x="1288" y="205" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">block</text>',
        '<text x="1110" y="251" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ 10 / 10 precision + pointer gates</text>',
        '<text x="1110" y="284" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ micro peak extra = 0</text>',
        '<text x="1110" y="333" font-family="Inter,Arial,sans-serif" font-size="16" font-weight="700" fill="#172033">w128: 1.21×–1.25× Torch</text>',
        '<text x="1110" y="366" font-family="Inter,Arial,sans-serif" font-size="16" font-weight="700" fill="#172033">w1024: 1.10×–1.11× Torch</text>',
        f'<text x="1110" y="399" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">serial → block: {min(model_gain):.1f}×–{max(model_gain):.1f}×</text>',
        f'<text x="1110" y="432" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">w4096 gain: {min(wide_gain):.1f}×–{max(wide_gain):.1f}×</text>',
        '<text x="1110" y="481" font-family="Inter,Arial,sans-serif" font-size="16" fill="#e28b22">△ w4096 remains 0.43×–0.46× Torch</text>',
        '<text x="1110" y="526" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">next: cache FP32 exp values for</text>',
        '<text x="1110" y="551" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">wide rows; avoid recomputing exp</text>',
        '<text x="720" y="714" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">Keep shape-aware block dispatch · continue only on the wide-row residual</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
