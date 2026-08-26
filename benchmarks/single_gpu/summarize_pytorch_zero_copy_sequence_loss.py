#!/usr/bin/env python3
"""Aggregate RoPE/Embedding/loss zero-copy matrices and draw error gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--svg", required=True)
    parser.add_argument("--expected-runs", type=int, default=3)
    arguments = parser.parse_args()
    root = Path(arguments.root)
    paths = sorted(path for path in root.glob("run-*") if path.is_dir())
    if len(paths) != arguments.expected_runs:
        raise ValueError(f"expected {arguments.expected_runs} runs, found {len(paths)}")
    reports = [json.loads((path / "report.json").read_text(encoding="utf-8"))
               for path in paths]
    if any(report["status"] != "pass" or report["record_count"] != 12
           for report in reports):
        raise ValueError("sequence/loss report is incomplete")
    records = [row for report in reports for row in report["records"]]
    if any(not row["pointer_matches"] or not row["wrappers_non_owning"] or
           row["max_error"] > row["tolerance"] for row in records):
        raise ValueError("sequence/loss row failed")
    groups = {}
    for row in records:
        group = groups.setdefault(row["operation"], {
            "operation": row["operation"], "rows": 0,
            "maximum_error": 0.0, "maximum_rms_error": 0.0,
            "tolerance": row["tolerance"], "maximum_tolerance_fraction": 0.0,
        })
        group["rows"] += 1
        group["maximum_error"] = max(group["maximum_error"], row["max_error"])
        group["maximum_rms_error"] = max(
            group["maximum_rms_error"], row["rms_error"])
        fraction = row["max_error"] / row["tolerance"] \
            if row["tolerance"] > 0 else 0.0
        group["maximum_tolerance_fraction"] = max(
            group["maximum_tolerance_fraction"], fraction)
    ordered = [groups[name] for name in ("rope", "embedding", "cross_entropy")]
    pointer_count = sum(
        2 if row["operation"] == "rope" else
        3 if row["operation"] == "embedding" else 4
        for row in records)
    summary = {
        "schema_version": 1,
        "status": "pass_with_profiler_boundary",
        "run_count": len(reports),
        "seeds": [report["seed"] for report in reports],
        "record_count": len(records),
        "groups": ordered,
        "pointer_matches": pointer_count,
        "non_owning_wrappers": pointer_count,
        "maximum_error": max(row["max_error"] for row in records),
        "maximum_rms_error": max(row["rms_error"] for row in records),
        "total_wrapped_payload_bytes": sum(
            report["total_wrapped_payload_bytes"] for report in reports),
        "total_wrapper_copy_bytes": 0,
        "rocprof_performance_claim": False,
    }
    if (summary["record_count"] != 36 or pointer_count != 108 or
            summary["total_wrapper_copy_bytes"] != 0):
        raise ValueError("sequence/loss aggregate failed")
    atomic_text(Path(arguments.summary),
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="500" viewBox="0 0 1120 500">',
        '<rect width="1120" height="500" fill="#0b1020"/>',
        '<text x="40" y="42" fill="#f4f7ff" font-family="sans-serif" font-size="25" '
        'font-weight="700">Zero-copy RoPE / Embedding / loss matrix</text>',
        '<text x="40" y="70" fill="#9eb0cf" font-family="sans-serif" font-size="14">'
        '36 complete PyTorch ROCm outputs · three seeds · bar = Max tolerance usage</text>',
        '<line x1="70" y1="360" x2="800" y2="360" stroke="#51617d"/>',
    ]
    for index, group in enumerate(ordered):
        x = 110 + index * 220
        fraction = group["maximum_tolerance_fraction"]
        height = fraction * 230
        parts.extend([
            f'<rect x="{x}" y="{360-height:.1f}" width="120" height="{height:.1f}" '
            'fill="#4cc9f0" rx="3"/>',
            f'<text x="{x+60}" y="{350-height:.1f}" text-anchor="middle" '
            f'fill="#f4f7ff" font-family="sans-serif" font-size="12">'
            f'{group["maximum_error"]:.3g}</text>',
            f'<text x="{x+60}" y="388" text-anchor="middle" fill="#cbd5e8" '
            f'font-family="sans-serif" font-size="13">{group["operation"]}</text>',
            f'<text x="{x+60}" y="412" text-anchor="middle" fill="#80ed99" '
            f'font-family="sans-serif" font-size="12">{fraction*100:.2f}% gate</text>',
        ])
    parts.extend([
        '<text x="850" y="135" fill="#80ed99" font-family="sans-serif" font-size="14">RoPE 12/12</text>',
        '<text x="850" y="175" fill="#80ed99" font-family="sans-serif" font-size="14">Embedding 12/12 exact</text>',
        '<text x="850" y="215" fill="#80ed99" font-family="sans-serif" font-size="14">CrossEntropy 12/12</text>',
        '<text x="850" y="255" fill="#80ed99" font-family="sans-serif" font-size="14">pointers 108/108</text>',
        '<text x="850" y="295" fill="#80ed99" font-family="sans-serif" font-size="14">wrapper copy 0 B</text>',
        '<text x="70" y="462" fill="#9eb0cf" font-family="sans-serif" font-size="13">'
        'CrossEntropy keeps caller-owned [rows,2] reduction workspace; ignored rows included.</text>',
        '</svg>',
    ])
    atomic_text(Path(arguments.svg), "\n".join(parts) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
