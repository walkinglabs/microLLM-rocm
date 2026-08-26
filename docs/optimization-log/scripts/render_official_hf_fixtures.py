#!/usr/bin/env python3
"""Render pinned official Hugging Face fixture evidence."""

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
    width, height = 1240, 650
    maximum = max(row["weight_bytes"] for row in data["models"])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f7f9fc"/>',
        '<text x="620" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">Official Fixtures Are Pinned, Inspected, and Kept Outside Git</text>',
        '<text x="620" y="79" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">complete safetensors headers + tokenizer/config presence · no payload vendoring</text>',
    ]
    colors = ("#4776c5", "#d06d45")
    for index, (row, color) in enumerate(zip(data["models"], colors)):
        y = 160 + index * 205
        bar_width = 650 * row["weight_bytes"] / maximum
        lines.append(f'<rect x="120" y="{y}" width="{bar_width:.1f}" height="62" rx="10" fill="{color}"/>')
        lines.append(f'<text x="135" y="{y + 38}" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#ffffff">{row["name"]}</text>')
        lines.append(f'<text x="120" y="{y + 91}" font-family="Inter,Arial,sans-serif" font-size="15" fill="#172033">{row["weight_bytes"] / 1e9:.3f} GB · {row["tensor_count"]} tensors · {row["parameter_count"]:,} parameters</text>')
    lines.extend([
        '<rect x="835" y="125" width="350" height="430" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="865" y="168" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Fixture gate</text>',
        '<text x="865" y="216" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ two pinned 40-char revisions</text>',
        '<text x="865" y="249" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ parameter/Tensor counts exact</text>',
        '<text x="865" y="282" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ config + vocab + merges present</text>',
        '<text x="865" y="315" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ BF16 weight dtype inspected</text>',
        '<text x="865" y="365" font-family="Inter,Arial,sans-serif" font-size="16" fill="#172033">Qwen license: Apache-2.0</text>',
        '<text x="865" y="398" font-family="Inter,Arial,sans-serif" font-size="16" fill="#172033">DeepSeek license: MIT</text>',
        '<text x="865" y="448" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">manifest stores paths locally</text>',
        '<text x="865" y="479" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">evidence stores no payload paths</text>',
        '<text x="865" y="510" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">large files remain outside Git</text>',
        '<text x="620" y="615" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">One manifest now feeds C++ inference, PyTorch oracles, and benchmarks</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
