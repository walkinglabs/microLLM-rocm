#!/usr/bin/env python3
"""Validate repeated PyTorch zero-copy Tensor descriptors and draw evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BOOLEAN_GATES = (
    "pointers_match",
    "wrappers_non_owning",
    "first_event_pending",
    "second_event_pending",
    "noncontiguous_rejected",
    "short_storage_rejected",
    "owner_retained_by_wrapper",
    "owner_released_after_close",
    "wrapper_destroy_preserved_torch",
)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_run(path: Path) -> dict:
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    spans = [json.loads(line) for line in
             (path / "profile.jsonl").read_text(encoding="utf-8").splitlines()
             if line]
    if (report["status"] != "pass" or
            not all(report[name] for name in BOOLEAN_GATES) or
            report["first_output_max_error"] != 0.0 or
            report["mutated_input_output_max_error"] != 0.0 or
            report["wrapper_copy_bytes"] != 0 or len(spans) != 2):
        raise ValueError(f"{path.name} failed zero-copy Tensor gates")
    return {
        "run": path.name,
        "torch_version": report["torch_version"],
        "torch_hip_version": report["torch_hip_version"],
        "iterations": report["iterations"],
        "wrapped_payload_bytes": report["wrapped_payload_bytes"],
        "wrapper_copy_bytes": report["wrapper_copy_bytes"],
        "first_output_max_error": report["first_output_max_error"],
        "mutated_input_output_max_error": report["mutated_input_output_max_error"],
        **{name: report[name] for name in BOOLEAN_GATES},
    }


def render_svg(runs: list[dict]) -> str:
    wrapped_mib = runs[0]["wrapped_payload_bytes"] / (1024.0 * 1024.0)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="500" viewBox="0 0 1120 500">',
        '<rect width="1120" height="500" fill="#0b1020"/>',
        '<text x="40" y="42" fill="#f4f7ff" font-family="sans-serif" font-size="25" '
        'font-weight="700">PyTorch ROCm zero-copy Tensor descriptors</text>',
        '<text x="40" y="70" fill="#9eb0cf" font-family="sans-serif" font-size="14">'
        'Three 16 MiB tensors per process; pointers and ownership remain external</text>',
        '<text x="55" y="115" fill="#f4f7ff" font-family="sans-serif" font-size="16">'
        'Payload exposed per run (MiB)</text>',
        '<line x1="55" y1="340" x2="660" y2="340" stroke="#51617d"/>',
    ]
    for index, run in enumerate(runs):
        x = 95 + index * 185
        height = wrapped_mib / 55.0 * 190.0
        parts.extend([
            f'<rect x="{x}" y="{340-height:.1f}" width="70" height="{height:.1f}" '
            'fill="#4cc9f0" rx="3"/>',
            f'<text x="{x+35}" y="{330-height:.1f}" text-anchor="middle" '
            f'fill="#f4f7ff" font-family="sans-serif" font-size="12">{wrapped_mib:.1f}</text>',
            f'<text x="{x+35}" y="365" text-anchor="middle" fill="#cbd5e8" '
            f'font-family="sans-serif" font-size="13">{run["run"]}</text>',
            f'<text x="{x+35}" y="388" text-anchor="middle" fill="#80ed99" '
            'font-family="sans-serif" font-size="12">copied: 0 B</text>',
        ])
    gates = [
        "pointer identity 3/3", "non-owning 3/3", "Torch mutation visible 3/3",
        "micro write visible 3/3", "owner retained/released 3/3",
        "wrapper destroy safe 3/3", "bad stride rejected 3/3",
        "short storage rejected 3/3", "output Max 0",
    ]
    parts.append('<text x="720" y="115" fill="#f4f7ff" font-family="sans-serif" font-size="16">Gates</text>')
    for index, gate in enumerate(gates):
        y = 150 + index * 31
        parts.append(
            f'<text x="720" y="{y}" fill="#80ed99" font-family="sans-serif" '
            f'font-size="13">✓ {gate}</text>')
    parts.extend([
        '<text x="55" y="448" fill="#9eb0cf" font-family="sans-serif" font-size="13">'
        '384 add submissions use caller pointers; this is correctness/lifetime evidence, not a timed speedup.</text>',
        '<text x="55" y="474" fill="#9eb0cf" font-family="sans-serif" font-size="13">'
        'Non-contiguous descriptors are representable but low-level add rejects them instead of copying.</text>',
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
        "status": "pass_with_profiler_boundary",
        "run_count": len(runs),
        "runs": runs,
        "torch_version": runs[0]["torch_version"],
        "torch_hip_version": runs[0]["torch_hip_version"],
        "all_gates_passed": all(all(run[name] for name in BOOLEAN_GATES)
                                for run in runs),
        "total_wrapped_payload_bytes": sum(
            run["wrapped_payload_bytes"] for run in runs),
        "total_wrapper_copy_bytes": sum(run["wrapper_copy_bytes"] for run in runs),
        "submitted_zero_copy_adds": sum(run["iterations"] * 2 for run in runs),
        "maximum_output_error": max(
            max(run["first_output_max_error"],
                run["mutated_input_output_max_error"]) for run in runs),
        "rocprof_performance_claim": False,
        "rocprof_boundary_source": "2026-08-26-pytorch-native-stream-interop",
    }
    if (not summary["all_gates_passed"] or
            summary["total_wrapper_copy_bytes"] != 0 or
            summary["submitted_zero_copy_adds"] != len(runs) * 128 or
            summary["maximum_output_error"] != 0.0):
        raise ValueError("repeated PyTorch zero-copy Tensor gate failed")
    atomic_text(Path(arguments.summary),
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_text(Path(arguments.svg), render_svg(runs))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
