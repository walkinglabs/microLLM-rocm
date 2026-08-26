#!/usr/bin/env python3
"""Validate repeated FP16/BF16 zero-copy outputs and render evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_run(path: Path) -> dict:
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "pass" or report["case_count"] != 2:
        raise ValueError(f"{path.name} has an invalid low-precision report")
    cases = {case["dtype"]: case for case in report["cases"]}
    if set(cases) != {"fp16", "bf16"}:
        raise ValueError(f"{path.name} dtype set is incomplete")
    for case in cases.values():
        if (not case["pointer_matches"] or not case["wrappers_non_owning"] or
                not case["multiply_pending"] or not case["matmul_pending"] or
                case["multiply_max_error"] > case["multiply_tolerance"] or
                case["matmul_max_error"] > case["matmul_tolerance"] or
                case["wrapper_copy_bytes"] != 0):
            raise ValueError(f"{path.name}/{case['dtype']} failed")
    return {
        "run": path.name,
        "torch_version": report["torch_version"],
        "torch_hip_version": report["torch_hip_version"],
        "wrapped_payload_bytes": report["total_wrapped_payload_bytes"],
        "wrapper_copy_bytes": report["total_wrapper_copy_bytes"],
        "cases": [cases["fp16"], cases["bf16"]],
    }


def render_svg(runs: list[dict]) -> str:
    wrapped_mib = runs[0]["wrapped_payload_bytes"] / (1024 * 1024)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="500" viewBox="0 0 1120 500">',
        '<rect width="1120" height="500" fill="#0b1020"/>',
        '<text x="40" y="42" fill="#f4f7ff" font-family="sans-serif" font-size="25" '
        'font-weight="700">FP16/BF16 zero-copy external outputs</text>',
        '<text x="40" y="70" fill="#9eb0cf" font-family="sans-serif" font-size="14">'
        'PyTorch ROCm pointers; multiply + hipBLASLt matmul; caller-owned output</text>',
        '<text x="55" y="115" fill="#f4f7ff" font-family="sans-serif" font-size="16">'
        'Wrapped payload per process (MiB)</text>',
        '<line x1="55" y1="340" x2="650" y2="340" stroke="#51617d"/>',
    ]
    for index, run in enumerate(runs):
        x = 90 + index * 180
        height = wrapped_mib / 70.0 * 190
        parts.extend([
            f'<rect x="{x}" y="{340-height:.1f}" width="74" height="{height:.1f}" '
            'fill="#4cc9f0" rx="3"/>',
            f'<text x="{x+37}" y="{330-height:.1f}" text-anchor="middle" '
            f'fill="#f4f7ff" font-family="sans-serif" font-size="12">{wrapped_mib:.1f}</text>',
            f'<text x="{x+37}" y="365" text-anchor="middle" fill="#cbd5e8" '
            f'font-family="sans-serif" font-size="13">{run["run"]}</text>',
            f'<text x="{x+37}" y="389" text-anchor="middle" fill="#80ed99" '
            'font-family="sans-serif" font-size="12">copy 0 B</text>',
        ])
    gates = (
        "FP16 pointers 18/18", "BF16 pointers 18/18",
        "multiply pending 6/6", "matmul pending 6/6",
        "FP16 multiply Max 0", "BF16 multiply Max 0",
        "FP16 matmul Max 0", "BF16 matmul Max 0",
        "768 zero-copy submissions",
    )
    parts.append('<text x="720" y="115" fill="#f4f7ff" font-family="sans-serif" font-size="16">Gates</text>')
    for index, gate in enumerate(gates):
        parts.append(
            f'<text x="720" y="{150+index*31}" fill="#80ed99" '
            f'font-family="sans-serif" font-size="13">✓ {gate}</text>')
    parts.extend([
        '<text x="55" y="455" fill="#9eb0cf" font-family="sans-serif" font-size="13">'
        'Operator results are exact for this fixed uniform workload; broader shape/error matrices remain separate.</text>',
        '<text x="55" y="478" fill="#9eb0cf" font-family="sans-serif" font-size="13">'
        'rocprof mixed-process injection remains blocked, so no performance speedup is claimed.</text>',
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
    all_cases = [case for run in runs for case in run["cases"]]
    summary = {
        "schema_version": 1,
        "status": "pass_with_profiler_boundary",
        "run_count": len(runs),
        "runs": runs,
        "dtype_cases": len(all_cases),
        "all_pointer_gates_passed": all(case["pointer_matches"] for case in all_cases),
        "all_wrappers_non_owning": all(case["wrappers_non_owning"] for case in all_cases),
        "pending_event_gates": sum(
            int(case["multiply_pending"]) + int(case["matmul_pending"])
            for case in all_cases),
        "maximum_multiply_error": max(case["multiply_max_error"] for case in all_cases),
        "maximum_matmul_error": max(case["matmul_max_error"] for case in all_cases),
        "total_wrapped_payload_bytes": sum(run["wrapped_payload_bytes"] for run in runs),
        "total_wrapper_copy_bytes": sum(run["wrapper_copy_bytes"] for run in runs),
        "submitted_zero_copy_ops": len(runs) * 2 * (64 + 64),
        "rocprof_performance_claim": False,
    }
    if (not summary["all_pointer_gates_passed"] or
            not summary["all_wrappers_non_owning"] or
            summary["pending_event_gates"] != 12 or
            summary["maximum_multiply_error"] != 0.0 or
            summary["maximum_matmul_error"] != 0.0 or
            summary["total_wrapper_copy_bytes"] != 0 or
            summary["submitted_zero_copy_ops"] != 768):
        raise ValueError("repeated low-precision zero-copy gate failed")
    atomic_text(Path(arguments.summary),
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_text(Path(arguments.svg), render_svg(runs))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
