#!/usr/bin/env python3
"""Validate explicit Python Stream isolation and render repeated evidence."""

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


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_run(path: Path) -> dict:
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    profile = json.loads((path / "profile.jsonl").read_text(encoding="utf-8"))
    apis = rows(path / "stream_hip_api_trace.csv")
    kernels = rows(path / "stream_kernel_trace.csv")
    markers = rows(path / "stream_marker_api_trace.csv")
    counts = Counter(row["Function"] for row in apis)
    formal_markers = [row for row in markers
                      if row["Function"].endswith("stream.a.matmul")]
    if len(formal_markers) != 1:
        raise ValueError(f"{path.name} has no unique target marker")
    marker = formal_markers[0]
    begin, end = int(marker["Start_Timestamp"]), int(marker["End_Timestamp"])
    inside = [row for row in apis
              if int(row["Start_Timestamp"]) >= begin and
              int(row["End_Timestamp"]) <= end]
    kernel_by_correlation = {row["Correlation_Id"]: row for row in kernels}
    target_pairs = [(api, kernel_by_correlation[api["Correlation_Id"]])
                    for api in inside if api["Correlation_Id"] in kernel_by_correlation]
    if len(target_pairs) != 1:
        raise ValueError(f"{path.name} target launch/kernel pair is ambiguous")
    target_api, target_kernel = target_pairs[0]
    target_stream = target_kernel["Stream_Id"]
    stream_counts = Counter(row["Stream_Id"] for row in kernels)
    busy_candidates = [(stream_id, count) for stream_id, count in stream_counts.items()
                       if stream_id not in {"0", target_stream}]
    if len(busy_candidates) != 1:
        raise ValueError(f"{path.name} busy Stream is ambiguous")
    busy_stream, busy_kernels = busy_candidates[0]
    observer = str(profile["completion_observer_native_thread_id"])
    submitter = str(profile["native_thread_id"])
    observer_syncs = [row for row in apis
                      if row["Function"] == "hipEventSynchronize" and
                      row["Thread_Id"] == observer]
    submitter_syncs = sorted(
        [row for row in apis
         if row["Function"] == "hipEventSynchronize" and
         row["Thread_Id"] == submitter],
        key=lambda row: int(row["Start_Timestamp"]))
    if (report["status"] != "pass" or
            not report["target_pending_at_submit"] or
            not report["busy_pending_before_target_wait"] or
            not report["busy_pending_after_target_wait"] or
            report["synchronization_scope"] != "hip_event_explicit_stream" or
            not report["observer_thread_is_distinct"] or
            report["maximum_output_error"] > 1.0e-3 or
            busy_kernels != report["busy_iterations"] or
            target_stream == busy_stream or len(observer_syncs) != 1 or
            len(submitter_syncs) != 2 or
            int(observer_syncs[0]["End_Timestamp"]) >=
            int(submitter_syncs[-1]["Start_Timestamp"]) or
            counts["hipDeviceSynchronize"] != 0 or
            counts["hipStreamSynchronize"] != 0 or
            target_api["Correlation_Id"] != target_kernel["Correlation_Id"] or
            marker["Correlation_Id"] == target_kernel["Correlation_Id"]):
        raise ValueError(f"{path.name} failed explicit Stream isolation")
    busy_wait_ms = (int(submitter_syncs[-1]["End_Timestamp"]) -
                    int(submitter_syncs[-1]["Start_Timestamp"])) / 1_000_000.0
    return {
        "run": path.name,
        "target_stream_id": int(target_stream),
        "busy_stream_id": int(busy_stream),
        "busy_kernels": busy_kernels,
        "target_device_elapsed_ms": report["target_device_elapsed_ns"] / 1_000_000.0,
        "target_submission_ms": report["target_submission_duration_ns"] / 1_000_000.0,
        "busy_wait_after_target_ms": busy_wait_ms,
        "target_pending_at_submit": True,
        "busy_pending_after_target_wait": True,
        "observer_event_synchronizes": 1,
        "device_synchronizes": counts["hipDeviceSynchronize"],
        "stream_synchronizes": counts["hipStreamSynchronize"],
        "launch_kernel_correlation_exact": True,
        "marker_kernel_ids_equal": False,
        "maximum_output_error": report["maximum_output_error"],
    }


def render_svg(runs: list[dict]) -> str:
    maximum = max(run["busy_wait_after_target_ms"] for run in runs) * 1.15
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="500" viewBox="0 0 1120 500">',
        '<rect width="1120" height="500" fill="#0b1020"/>',
        '<text x="40" y="42" fill="#f4f7ff" font-family="sans-serif" font-size="25" '
        'font-weight="700">Explicit Python Stream isolation</text>',
        '<text x="40" y="70" fill="#9eb0cf" font-family="sans-serif" font-size="14">'
        'Wait target Event; 64-GEMM independent Stream must remain pending</text>',
        '<line x1="55" y1="340" x2="770" y2="340" stroke="#51617d"/>',
        '<text x="55" y="110" fill="#f4f7ff" font-family="sans-serif" font-size="16">'
        'Target Event and remaining busy-Stream wait (ms)</text>',
    ]
    group = 680.0 / len(runs)
    for index, run in enumerate(runs):
        base = 85 + index * group
        target_height = run["target_device_elapsed_ms"] / maximum * 190
        busy_height = run["busy_wait_after_target_ms"] / maximum * 190
        parts.extend([
            f'<rect x="{base:.1f}" y="{340-target_height:.1f}" width="52" '
            f'height="{target_height:.1f}" fill="#4cc9f0" rx="3"/>',
            f'<rect x="{base+62:.1f}" y="{340-busy_height:.1f}" width="52" '
            f'height="{busy_height:.1f}" fill="#f9c74f" rx="3"/>',
            f'<text x="{base+26:.1f}" y="{330-target_height:.1f}" text-anchor="middle" '
            f'fill="#f4f7ff" font-family="sans-serif" font-size="11">'
            f'{run["target_device_elapsed_ms"]:.3f}</text>',
            f'<text x="{base+88:.1f}" y="{330-busy_height:.1f}" text-anchor="middle" '
            f'fill="#f4f7ff" font-family="sans-serif" font-size="11">'
            f'{run["busy_wait_after_target_ms"]:.3f}</text>',
            f'<text x="{base+57:.1f}" y="365" text-anchor="middle" fill="#cbd5e8" '
            f'font-family="sans-serif" font-size="13">{run["run"]}</text>',
        ])
    parts.extend([
        '<text x="820" y="112" fill="#f4f7ff" font-family="sans-serif" font-size="16">Gate</text>',
        '<text x="820" y="150" fill="#80ed99" font-family="sans-serif" font-size="14">3/3 target pending</text>',
        '<text x="820" y="182" fill="#80ed99" font-family="sans-serif" font-size="14">3/3 busy still pending</text>',
        '<text x="820" y="214" fill="#80ed99" font-family="sans-serif" font-size="14">192/192 busy GEMMs</text>',
        '<text x="820" y="246" fill="#80ed99" font-family="sans-serif" font-size="14">device sync: 0</text>',
        '<text x="820" y="278" fill="#80ed99" font-family="sans-serif" font-size="14">Stream sync: 0</text>',
        '<rect x="60" y="405" width="15" height="15" fill="#4cc9f0"/>',
        '<text x="83" y="418" fill="#cbd5e8" font-family="sans-serif" font-size="13">target Event device time</text>',
        '<rect x="285" y="405" width="15" height="15" fill="#f9c74f"/>',
        '<text x="308" y="418" fill="#cbd5e8" font-family="sans-serif" font-size="13">busy work remaining after target wait</text>',
        '<text x="60" y="462" fill="#9eb0cf" font-family="sans-serif" font-size="13">'
        'The independent Stream uses preallocated output: no allocation is allowed to fake serialization.</text>',
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
        "all_targets_pending_at_submit": all(
            run["target_pending_at_submit"] for run in runs),
        "all_busy_streams_pending_after_target_wait": all(
            run["busy_pending_after_target_wait"] for run in runs),
        "total_busy_kernels": sum(run["busy_kernels"] for run in runs),
        "total_device_synchronizes": sum(run["device_synchronizes"] for run in runs),
        "total_stream_synchronizes": sum(run["stream_synchronizes"] for run in runs),
        "minimum_busy_wait_after_target_ms": min(
            run["busy_wait_after_target_ms"] for run in runs),
        "maximum_output_error": max(run["maximum_output_error"] for run in runs),
        "marker_kernel_id_equal_count": sum(
            run["marker_kernel_ids_equal"] for run in runs),
    }
    if (not summary["all_targets_pending_at_submit"] or
            not summary["all_busy_streams_pending_after_target_wait"] or
            summary["total_busy_kernels"] != len(runs) * 64 or
            summary["total_device_synchronizes"] != 0 or
            summary["total_stream_synchronizes"] != 0 or
            summary["minimum_busy_wait_after_target_ms"] <= 0.0 or
            summary["maximum_output_error"] > 1.0e-3):
        raise ValueError("repeated explicit Stream isolation failed")
    atomic_text(Path(arguments.summary),
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_text(Path(arguments.svg), render_svg(runs))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
