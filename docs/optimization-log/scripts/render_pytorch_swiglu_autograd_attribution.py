#!/usr/bin/env python3
"""Render native/custom/manual SwiGLU F+B attribution."""

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
    width, height = 1220, 680
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1220" height="680" fill="#f7f9fc"/>',
        '<text x="610" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">SwiGLU F+B Gap Is in Autograd Submission</text>',
        '<text x="610" y="78" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">same FP32 math · six fresh MI300X processes · Event milliseconds per iteration</text>',
        '<line x1="80" y1="500" x2="820" y2="500" stroke="#aab3c2" stroke-width="2"/>',
    ]
    maximum = max(row["custom_event_ms_median"] for row in groups) * 1.15
    colors = {"native": "#d97706", "custom": "#b42335", "manual": "#198754"}
    for index, row in enumerate(groups):
        center = 250 + index * 350
        for offset, policy in ((-80, "native"), (0, "custom"), (80, "manual")):
            value = row[f"{policy}_event_ms_median"]
            bar_height = value / maximum * 330
            y = 500 - bar_height
            lines.append(f'<rect x="{center + offset - 30}" y="{y:.1f}" width="60" height="{bar_height:.1f}" rx="7" fill="{colors[policy]}"/>')
            lines.append(f'<text x="{center + offset}" y="{y - 8:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" font-weight="700" fill="#172033">{value:.4f}</text>')
            lines.append(f'<text x="{center + offset}" y="525" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" fill="#5b6474">{policy}</text>')
        lines.append(f'<text x="{center}" y="560" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" font-weight="700" fill="#172033">{row["shape"]} · {row["elements"]}</text>')
    lines.extend([
        '<rect x="865" y="125" width="300" height="375" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="895" y="167" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Attribution</text>',
        '<text x="895" y="211" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ all losses/gradients match</text>',
        '<text x="895" y="244" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ same fused GPU producers</text>',
        '<text x="895" y="292" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">manual/custom 4.855×–5.271×</text>',
        '<text x="895" y="325" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">manual/native 3.859×–4.105×</text>',
        '<text x="895" y="373" font-family="Inter,Arial,sans-serif" font-size="16" fill="#b42335">Python Autograd submission gap</text>',
        '<text x="895" y="406" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">not a HIP arithmetic bottleneck</text>',
        '<text x="895" y="449" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">next: C++ Autograd or compiled graph</text>',
        '<text x="610" y="635" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">Stop changing math kernels · optimize the framework boundary</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

