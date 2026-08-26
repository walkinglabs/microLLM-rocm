#!/usr/bin/env python3
"""Validate repeated Python/ROCTX/GPU captures and draw their calibration quality."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_run(path: Path, expected_iterations: int) -> dict:
    calibration = json.loads((path / "calibration.json").read_text(encoding="utf-8"))
    timeline = json.loads((path / "unified.json").read_text(encoding="utf-8"))
    profile_rows = [json.loads(line) for line in
                    (path / "profile.jsonl").read_text(encoding="utf-8").splitlines()
                    if line]
    with (path / "python-unified_marker_api_trace.csv").open(
            newline="", encoding="utf-8") as stream:
        markers = list(csv.DictReader(stream))
    with (path / "python-unified_kernel_trace.csv").open(
            newline="", encoding="utf-8") as stream:
        kernels = list(csv.DictReader(stream))
    with (path / "python-unified_hip_api_trace.csv").open(
            newline="", encoding="utf-8") as stream:
        hip_apis = list(csv.DictReader(stream))
    events = timeline.get("traceEvents", [])
    correlated = [event for event in events
                  if (event.get("cat") == "gpu_kernel" and
                      event.get("args", {}).get("roctx_range"))]
    correlated_adds = [event for event in correlated
                       if "add_typed_kernel" in event["name"]]
    python_events = [event for event in events
                     if str(event.get("cat", "")).startswith("python_")]
    flows = [event for event in events if event.get("cat") == "correlation"]
    if (calibration.get("status") != "pass" or
            calibration.get("matched_spans") != expected_iterations or
            len(profile_rows) != expected_iterations * 3 or
            len(correlated_adds) != expected_iterations or
            len(flows) != len(correlated) * 2 or
            len(python_events) != len(profile_rows)):
        raise ValueError(f"{path.name} does not satisfy the capture contract")
    return {
        "run": path.name,
        "scale": float(calibration["scale"]),
        "scale_error_ppm": abs(float(calibration["scale"]) - 1.0) * 1_000_000.0,
        "max_abs_residual_ns": float(calibration["max_abs_residual_ns"]),
        "rms_residual_ns": float(calibration["rms_residual_ns"]),
        "max_boundary_width_ns": int(calibration["max_boundary_width_ns"]),
        "matched_spans": int(calibration["matched_spans"]),
        "profile_rows": len(profile_rows),
        "marker_events": len(markers),
        "kernel_events": len(kernels),
        "hip_api_events": len(hip_apis),
        "correlated_pairs": len(correlated),
        "correlated_adds": len(correlated_adds),
        "trace_events": len(events),
    }


def render_svg(runs: list[dict]) -> str:
    width, height = 1120, 500
    colors = ["#4cc9f0", "#80ed99", "#f9c74f", "#f9844a"]
    max_quality_us = max(max(run["max_abs_residual_ns"],
                             run["max_boundary_width_ns"]) / 1000.0
                         for run in runs) * 1.15
    max_quality_us = max(max_quality_us, 1.0)
    max_ppm = max(run["scale_error_ppm"] for run in runs) * 1.2
    max_ppm = max(max_ppm, 1.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="1120" height="500" fill="#0b1020"/>',
        '<text x="40" y="42" fill="#f4f7ff" font-family="sans-serif" '
        'font-size="25" font-weight="700">Python → ROCTX → GPU clock calibration</text>',
        '<text x="40" y="68" fill="#9eb0cf" font-family="sans-serif" '
        'font-size="14">Lower is better; every run contains 8 marked spans and 8 correlated HIP adds.</text>',
        '<text x="40" y="108" fill="#f4f7ff" font-family="sans-serif" '
        'font-size="16">Boundary and fit error (µs)</text>',
        '<line x1="55" y1="320" x2="650" y2="320" stroke="#51617d"/>',
        '<text x="710" y="108" fill="#f4f7ff" font-family="sans-serif" '
        'font-size="16">Clock scale error (ppm)</text>',
        '<line x1="725" y1="320" x2="1070" y2="320" stroke="#51617d"/>',
    ]
    group_width = 560.0 / len(runs)
    ppm_width = 320.0 / len(runs)
    for index, run in enumerate(runs):
        x = 75.0 + index * group_width
        residual_us = run["max_abs_residual_ns"] / 1000.0
        boundary_us = run["max_boundary_width_ns"] / 1000.0
        residual_height = residual_us / max_quality_us * 185.0
        boundary_height = boundary_us / max_quality_us * 185.0
        parts.extend([
            f'<rect x="{x:.1f}" y="{320-boundary_height:.1f}" width="42" '
            f'height="{boundary_height:.1f}" fill="{colors[0]}" rx="3"/>',
            f'<rect x="{x+48:.1f}" y="{320-residual_height:.1f}" width="42" '
            f'height="{residual_height:.1f}" fill="{colors[1]}" rx="3"/>',
            f'<text x="{x+45:.1f}" y="344" text-anchor="middle" fill="#cbd5e8" '
            f'font-family="sans-serif" font-size="13">{run["run"]}</text>',
            f'<text x="{x+21:.1f}" y="{310-boundary_height:.1f}" text-anchor="middle" '
            f'fill="#f4f7ff" font-family="sans-serif" font-size="11">{boundary_us:.2f}</text>',
            f'<text x="{x+69:.1f}" y="{310-residual_height:.1f}" text-anchor="middle" '
            f'fill="#f4f7ff" font-family="sans-serif" font-size="11">{residual_us:.2f}</text>',
        ])
        ppm_x = 740.0 + index * ppm_width
        ppm_height = run["scale_error_ppm"] / max_ppm * 185.0
        parts.extend([
            f'<rect x="{ppm_x:.1f}" y="{320-ppm_height:.1f}" width="62" '
            f'height="{ppm_height:.1f}" fill="{colors[2]}" rx="3"/>',
            f'<text x="{ppm_x+31:.1f}" y="344" text-anchor="middle" fill="#cbd5e8" '
            f'font-family="sans-serif" font-size="13">{run["run"]}</text>',
            f'<text x="{ppm_x+31:.1f}" y="{310-ppm_height:.1f}" text-anchor="middle" '
            f'fill="#f4f7ff" font-family="sans-serif" font-size="11">'
            f'{run["scale_error_ppm"]:.2f}</text>',
        ])
    parts.extend([
        '<rect x="58" y="380" width="16" height="16" fill="#4cc9f0" rx="2"/>',
        '<text x="82" y="393" fill="#cbd5e8" font-family="sans-serif" '
        'font-size="13">widest ctypes boundary</text>',
        '<rect x="250" y="380" width="16" height="16" fill="#80ed99" rx="2"/>',
        '<text x="274" y="393" fill="#cbd5e8" font-family="sans-serif" '
        'font-size="13">maximum affine residual</text>',
        '<text x="58" y="440" fill="#9eb0cf" font-family="sans-serif" font-size="13">'
        'Acceptance gates: boundary ≤100 µs, residual ≤50 µs, scale within 1% of nanoseconds.</text>',
        '<text x="58" y="466" fill="#9eb0cf" font-family="sans-serif" font-size="13">'
        'A warm-up range is intentionally excluded from the fit; unmarked nested Python spans use the measured map.</text>',
        '</svg>',
    ])
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--svg", required=True)
    parser.add_argument("--expected-runs", type=int, default=3)
    parser.add_argument("--expected-iterations", type=int, default=8)
    arguments = parser.parse_args()
    root = Path(arguments.root)
    run_paths = sorted(path for path in root.glob("run-*") if path.is_dir())
    if len(run_paths) != arguments.expected_runs:
        raise ValueError(
            f"expected {arguments.expected_runs} run directories, found {len(run_paths)}")
    runs = [load_run(path, arguments.expected_iterations) for path in run_paths]
    summary = {
        "schema_version": 1,
        "status": "pass",
        "runs": runs,
        "run_count": len(runs),
        "iterations_per_run": arguments.expected_iterations,
        "max_scale_error_ppm": max(run["scale_error_ppm"] for run in runs),
        "max_abs_residual_ns": max(run["max_abs_residual_ns"] for run in runs),
        "max_boundary_width_ns": max(run["max_boundary_width_ns"] for run in runs),
        "total_correlated_adds": sum(run["correlated_adds"] for run in runs),
        "total_profile_rows": sum(run["profile_rows"] for run in runs),
    }
    if (summary["max_scale_error_ppm"] > 10_000.0 or
            summary["max_abs_residual_ns"] > 50_000.0 or
            summary["max_boundary_width_ns"] > 100_000 or
            summary["total_correlated_adds"] !=
            arguments.expected_runs * arguments.expected_iterations):
        raise ValueError("repeated clock-correlation gate failed")
    atomic_text(Path(arguments.summary),
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_text(Path(arguments.svg), render_svg(runs))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
