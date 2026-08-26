#!/usr/bin/env python3
"""Render eager/compiled/native/manual SwiGLU F+B evidence."""

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
    width, height = 1260, 700
    policies = ("native", "eager", "compiled", "manual")
    colors = {"native": "#d97706", "eager": "#3769d6",
              "compiled": "#b42335", "manual": "#198754"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1260" height="700" fill="#f7f9fc"/>',
        '<text x="630" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">torch.compile Makes the Opaque Custom Op Slower</text>',
        '<text x="630" y="78" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">eight fresh MI300X processes · steady Event ms · compile cold start excluded from bars</text>',
        '<line x1="75" y1="510" x2="870" y2="510" stroke="#aab3c2" stroke-width="2"/>',
    ]
    maximum = max(row["compiled_event_ms_median"] for row in groups) * 1.15
    for index, row in enumerate(groups):
        center = 260 + index * 390
        for policy_index, policy in enumerate(policies):
            x = center + (policy_index - 1.5) * 75
            value = row[f"{policy}_event_ms_median"]
            bar_height = value / maximum * 330
            y = 510 - bar_height
            lines.append(f'<rect x="{x - 26:.1f}" y="{y:.1f}" width="52" height="{bar_height:.1f}" rx="6" fill="{colors[policy]}"/>')
            lines.append(f'<text x="{x:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" font-weight="700" fill="#172033">{value:.4f}</text>')
            lines.append(f'<text x="{x:.1f}" y="535" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" fill="#5b6474">{policy}</text>')
        lines.append(f'<text x="{center}" y="568" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" font-weight="700" fill="#172033">{row["shape"]} · cold {row["compile_cold_ms_median"]:.1f} ms</text>')
    lines.extend([
        '<rect x="910" y="125" width="300" height="385" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="940" y="167" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Rejection evidence</text>',
        '<text x="940" y="211" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ gradients Max 4.77e-7</text>',
        '<text x="940" y="244" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">loss reduction delta ≤0.00390625</text>',
        '<text x="940" y="292" font-family="Inter,Arial,sans-serif" font-size="16" fill="#b42335">compiled/eager 0.584×–0.610×</text>',
        '<text x="940" y="325" font-family="Inter,Arial,sans-serif" font-size="16" fill="#b42335">compiled/native 0.462×–0.476×</text>',
        '<text x="940" y="358" font-family="Inter,Arial,sans-serif" font-size="16" fill="#b42335">manual/compiled 7.696×–8.635×</text>',
        '<text x="940" y="406" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">cold start 55.8–1160.3 ms</text>',
        '<text x="940" y="449" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">opaque ops are not fused away</text>',
        '<text x="630" y="655" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">Reject compiled recommendation · C++ Autograd is the last adjacent candidate</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

