#!/usr/bin/env python3
"""Validate repeated Python HIP Event observations and draw the evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_run(path: Path) -> dict:
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    profile_rows = [json.loads(line) for line in
                    (path / "profile.jsonl").read_text(encoding="utf-8").splitlines()
                    if line]
    apis = load_csv(path / "event_hip_api_trace.csv")
    markers = load_csv(path / "event_marker_api_trace.csv")
    kernels = load_csv(path / "event_kernel_trace.csv")
    if len(profile_rows) != 1:
        raise ValueError(f"{path.name} must contain one formal Event span")
    profile = profile_rows[0]
    formal_markers = [row for row in markers if row["Function"].endswith("softmax.async")]
    if len(formal_markers) != 1:
        raise ValueError(f"{path.name} has no unique formal ROCTX range")
    marker = formal_markers[0]
    marker_start = int(marker["Start_Timestamp"])
    marker_end = int(marker["End_Timestamp"])
    inside = [row for row in apis
              if (int(row["Start_Timestamp"]) >= marker_start and
                  int(row["End_Timestamp"]) <= marker_end)]
    inside_counts = Counter(row["Function"] for row in inside)
    all_counts = Counter(row["Function"] for row in apis)
    launch_rows = [row for row in inside if row["Function"] == "hipLaunchKernel"]
    if len(launch_rows) != 1:
        raise ValueError(f"{path.name} has no unique softmax launch API")
    launch = launch_rows[0]
    matching_kernels = [row for row in kernels
                        if row["Correlation_Id"] == launch["Correlation_Id"]]
    observer_thread = str(profile["completion_observer_native_thread_id"])
    observer_syncs = [row for row in apis
                      if (row["Function"] == "hipEventSynchronize" and
                          row["Thread_Id"] == observer_thread)]
    if (report["status"] != "pass" or report["event_ready_at_submit"] or
            report["maximum_output_error"] != 0.0 or
            not report["observer_thread_is_distinct"] or
            report["host_work_before_completion_observed_ns"] <= 0 or
            profile["kind"] != "hip_event_completion_span" or
            profile["synchronization_scope"] != "hip_event_default_stream" or
            inside_counts["hipEventRecord"] != 2 or
            inside_counts["hipEventQuery"] != 1 or
            inside_counts["hipEventSynchronize"] != 0 or
            all_counts["hipDeviceSynchronize"] != 0 or
            all_counts["hipStreamSynchronize"] != 0 or
            len(observer_syncs) != 1 or len(matching_kernels) != 1 or
            "softmax_kernel" not in matching_kernels[0]["Kernel_Name"]):
        raise ValueError(f"{path.name} failed the Event/API evidence contract")
    return {
        "run": path.name,
        "submission_ms": report["submission_duration_ns"] / 1_000_000.0,
        "device_elapsed_ms": report["device_elapsed_ns"] / 1_000_000.0,
        "completion_observed_ms": report["completion_duration_ns"] / 1_000_000.0,
        "host_work_before_observation_ms":
            report["host_work_before_completion_observed_ns"] / 1_000_000.0,
        "event_ready_at_submit": False,
        "observer_thread_is_distinct": True,
        "formal_event_records": inside_counts["hipEventRecord"],
        "formal_event_queries": inside_counts["hipEventQuery"],
        "observer_event_synchronizes": len(observer_syncs),
        "device_synchronizes": all_counts["hipDeviceSynchronize"],
        "stream_synchronizes": all_counts["hipStreamSynchronize"],
        "launch_kernel_correlation_exact": True,
        "marker_kernel_ids_equal": (
            marker["Correlation_Id"] == matching_kernels[0]["Correlation_Id"]),
        "maximum_output_error": report["maximum_output_error"],
    }


def render_svg(runs: list[dict]) -> str:
    maximum = max(run["completion_observed_ms"] for run in runs) * 1.15
    colors = ["#4cc9f0", "#80ed99", "#f9c74f"]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="500" '
        'viewBox="0 0 1120 500">',
        '<rect width="1120" height="500" fill="#0b1020"/>',
        '<text x="40" y="42" fill="#f4f7ff" font-family="sans-serif" '
        'font-size="25" font-weight="700">Async HIP Event completion: measured, not globally synchronized</text>',
        '<text x="40" y="70" fill="#9eb0cf" font-family="sans-serif" '
        'font-size="14">4096×1024 softmax; three fresh MI300X processes</text>',
        '<text x="45" y="112" fill="#f4f7ff" font-family="sans-serif" '
        'font-size="16">Host observation and device Event durations (ms)</text>',
        '<line x1="55" y1="340" x2="730" y2="340" stroke="#51617d"/>',
        '<text x="790" y="112" fill="#f4f7ff" font-family="sans-serif" '
        'font-size="16">Formal route API calls</text>',
    ]
    group = 640.0 / len(runs)
    for index, run in enumerate(runs):
        base = 75.0 + index * group
        values = [run["submission_ms"], run["device_elapsed_ms"],
                  run["completion_observed_ms"]]
        for value_index, value in enumerate(values):
            bar_height = value / maximum * 190.0
            x = base + value_index * 48.0
            parts.extend([
                f'<rect x="{x:.1f}" y="{340-bar_height:.1f}" width="40" '
                f'height="{bar_height:.1f}" fill="{colors[value_index]}" rx="3"/>',
                f'<text x="{x+20:.1f}" y="{330-bar_height:.1f}" '
                f'text-anchor="middle" fill="#f4f7ff" font-family="sans-serif" '
                f'font-size="11">{value:.3f}</text>',
            ])
        parts.append(
            f'<text x="{base+68:.1f}" y="365" text-anchor="middle" '
            f'fill="#cbd5e8" font-family="sans-serif" font-size="13">{run["run"]}</text>')
    api_rows = [
        ("hipEventRecord", 2, "#4cc9f0"),
        ("hipEventQuery", 1, "#80ed99"),
        ("observer hipEventSynchronize", 1, "#f9c74f"),
        ("hipDeviceSynchronize", 0, "#f94144"),
        ("hipStreamSynchronize", 0, "#f94144"),
    ]
    for index, (name, count, color) in enumerate(api_rows):
        y = 150 + index * 42
        parts.extend([
            f'<rect x="790" y="{y-18}" width="{max(4, count*42)}" height="24" '
            f'fill="{color}" rx="3"/>',
            f'<text x="{810+max(4, count*42)}" y="{y}" fill="#cbd5e8" '
            f'font-family="sans-serif" font-size="13">{name}: {count}</text>',
        ])
    parts.extend([
        '<rect x="60" y="400" width="15" height="15" fill="#4cc9f0"/>',
        '<text x="82" y="413" fill="#cbd5e8" font-family="sans-serif" font-size="13">submission</text>',
        '<rect x="180" y="400" width="15" height="15" fill="#80ed99"/>',
        '<text x="202" y="413" fill="#cbd5e8" font-family="sans-serif" font-size="13">device Event elapsed</text>',
        '<rect x="382" y="400" width="15" height="15" fill="#f9c74f"/>',
        '<text x="404" y="413" fill="#cbd5e8" font-family="sans-serif" font-size="13">completion observed</text>',
        '<text x="60" y="454" fill="#9eb0cf" font-family="sans-serif" font-size="13">'
        'Completion observation is an upper bound because Python thread scheduling may delay the observer.</text>',
        '<text x="60" y="478" fill="#9eb0cf" font-family="sans-serif" font-size="13">'
        'The exact device duration comes from HIP Events; no device-wide or stream-wide synchronize is called.</text>',
        '</svg>',
    ])
    return "\n".join(parts) + "\n"


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
    runs = [load_run(path) for path in paths]
    summary = {
        "schema_version": 1,
        "status": "pass",
        "run_count": len(runs),
        "runs": runs,
        "all_pending_at_submit": all(not run["event_ready_at_submit"] for run in runs),
        "all_observers_distinct": all(run["observer_thread_is_distinct"] for run in runs),
        "all_launch_kernel_correlations_exact": all(
            run["launch_kernel_correlation_exact"] for run in runs),
        "marker_kernel_id_equal_count": sum(
            run["marker_kernel_ids_equal"] for run in runs),
        "total_device_synchronizes": sum(run["device_synchronizes"] for run in runs),
        "total_stream_synchronizes": sum(run["stream_synchronizes"] for run in runs),
        "minimum_host_work_before_observation_ms": min(
            run["host_work_before_observation_ms"] for run in runs),
        "maximum_device_elapsed_ms": max(run["device_elapsed_ms"] for run in runs),
        "maximum_output_error": max(run["maximum_output_error"] for run in runs),
    }
    if (not summary["all_pending_at_submit"] or
            not summary["all_observers_distinct"] or
            not summary["all_launch_kernel_correlations_exact"] or
            summary["total_device_synchronizes"] != 0 or
            summary["total_stream_synchronizes"] != 0 or
            summary["minimum_host_work_before_observation_ms"] <= 0.0 or
            summary["maximum_output_error"] != 0.0):
        raise ValueError("repeated asynchronous Event gate failed")
    atomic_text(Path(arguments.summary),
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_text(Path(arguments.svg), render_svg(runs))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
