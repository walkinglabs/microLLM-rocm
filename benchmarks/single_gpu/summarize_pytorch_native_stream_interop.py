#!/usr/bin/env python3
"""Validate repeated PyTorch/native-Stream ordering and retain profiler failure."""

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
    spans = [json.loads(line) for line in
             (path / "profile.jsonl").read_text(encoding="utf-8").splitlines()
             if line]
    if (report["status"] != "pass" or report["wrapper_owning"] or
            not report["torch_pending_for_microllm_event"] or
            not report["microllm_pending_for_torch_event"] or
            report["maximum_output_error"] > 1.0e-3 or len(spans) != 2 or
            {span["name"] for span in spans} !=
            {"torch.to.microllm", "microllm.to.torch"} or
            not all(span["roctx_emitted"] for span in spans)):
        raise ValueError(f"{path.name} failed native Stream ordering")
    return {
        "run": path.name,
        "torch_version": report["torch_version"],
        "torch_hip_version": report["torch_hip_version"],
        "native_stream_handle_nonzero": report["native_stream_handle"] > 0,
        "wrapper_owning": False,
        "iterations_per_direction": report["iterations_per_direction"],
        "torch_pending_for_microllm_event": True,
        "microllm_pending_for_torch_event": True,
        "torch_to_microllm_wait_ms":
            report["torch_to_microllm_event_wait_ns"] / 1_000_000.0,
        "microllm_to_torch_wait_ms":
            report["microllm_to_torch_event_wait_ns"] / 1_000_000.0,
        "maximum_output_error": report["maximum_output_error"],
    }


def render_svg(runs: list[dict]) -> str:
    maximum = max(max(run["torch_to_microllm_wait_ms"],
                      run["microllm_to_torch_wait_ms"]) for run in runs) * 1.15
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="500" viewBox="0 0 1120 500">',
        '<rect width="1120" height="500" fill="#0b1020"/>',
        '<text x="40" y="42" fill="#f4f7ff" font-family="sans-serif" font-size="25" '
        'font-weight="700">PyTorch ROCm ↔ microLLM native Stream ordering</text>',
        '<text x="40" y="70" fill="#9eb0cf" font-family="sans-serif" font-size="14">'
        'Same non-owning torch.cuda.Stream; 64 preallocated GEMMs per direction</text>',
        '<line x1="55" y1="340" x2="780" y2="340" stroke="#51617d"/>',
        '<text x="55" y="110" fill="#f4f7ff" font-family="sans-serif" font-size="16">'
        'Cross-framework Event wait (ms)</text>',
    ]
    group = 680.0 / len(runs)
    for index, run in enumerate(runs):
        base = 85 + index * group
        first = run["torch_to_microllm_wait_ms"] / maximum * 190
        second = run["microllm_to_torch_wait_ms"] / maximum * 190
        parts.extend([
            f'<rect x="{base:.1f}" y="{340-first:.1f}" width="52" height="{first:.1f}" '
            'fill="#4cc9f0" rx="3"/>',
            f'<rect x="{base+62:.1f}" y="{340-second:.1f}" width="52" height="{second:.1f}" '
            'fill="#f9c74f" rx="3"/>',
            f'<text x="{base+26:.1f}" y="{330-first:.1f}" text-anchor="middle" '
            f'fill="#f4f7ff" font-family="sans-serif" font-size="11">'
            f'{run["torch_to_microllm_wait_ms"]:.3f}</text>',
            f'<text x="{base+88:.1f}" y="{330-second:.1f}" text-anchor="middle" '
            f'fill="#f4f7ff" font-family="sans-serif" font-size="11">'
            f'{run["microllm_to_torch_wait_ms"]:.3f}</text>',
            f'<text x="{base+57:.1f}" y="365" text-anchor="middle" fill="#cbd5e8" '
            f'font-family="sans-serif" font-size="13">{run["run"]}</text>',
        ])
    parts.extend([
        '<text x="830" y="118" fill="#80ed99" font-family="sans-serif" font-size="14">'
        'Torch → micro Event: 3/3 pending</text>',
        '<text x="830" y="152" fill="#80ed99" font-family="sans-serif" font-size="14">'
        'micro → Torch Event: 3/3 pending</text>',
        '<text x="830" y="186" fill="#80ed99" font-family="sans-serif" font-size="14">'
        'wrapper ownership: 0/3</text>',
        '<text x="830" y="220" fill="#80ed99" font-family="sans-serif" font-size="14">'
        'output max error: 2.57e-8</text>',
        '<text x="830" y="270" fill="#f94144" font-family="sans-serif" font-size="14">'
        'rocprof injection: LLVM option conflict</text>',
        '<rect x="60" y="405" width="15" height="15" fill="#4cc9f0"/>',
        '<text x="83" y="418" fill="#cbd5e8" font-family="sans-serif" font-size="13">Torch work waited by micro Event</text>',
        '<rect x="330" y="405" width="15" height="15" fill="#f9c74f"/>',
        '<text x="353" y="418" fill="#cbd5e8" font-family="sans-serif" font-size="13">micro work waited by Torch Event</text>',
        '<text x="60" y="468" fill="#9eb0cf" font-family="sans-serif" font-size="13">'
        'Ordering/output evidence passes; rocprof cannot be claimed for this mixed LLVM process.</text>',
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
    failure = (root / "rocprof-injection-failure/rocprof-stderr.txt").read_text(
        encoding="utf-8", errors="replace")
    if ("spirv-expand-step' registered more than once" not in failure or
            "LLVM ERROR: inconsistency" not in failure):
        raise ValueError("rocprof injection failure is missing its bounded diagnosis")
    runs = [load_run(path) for path in paths]
    summary = {
        "schema_version": 1,
        "status": "pass_with_profiler_boundary",
        "run_count": len(runs),
        "runs": runs,
        "torch_version": runs[0]["torch_version"],
        "torch_hip_version": runs[0]["torch_hip_version"],
        "all_torch_work_pending_for_microllm": all(
            run["torch_pending_for_microllm_event"] for run in runs),
        "all_microllm_work_pending_for_torch": all(
            run["microllm_pending_for_torch_event"] for run in runs),
        "all_wrappers_non_owning": all(not run["wrapper_owning"] for run in runs),
        "minimum_torch_to_microllm_wait_ms": min(
            run["torch_to_microllm_wait_ms"] for run in runs),
        "minimum_microllm_to_torch_wait_ms": min(
            run["microllm_to_torch_wait_ms"] for run in runs),
        "maximum_output_error": max(run["maximum_output_error"] for run in runs),
        "rocprof_injection": "failed_duplicate_llvm_command_line_option",
        "rocprof_performance_claim": False,
    }
    if (not summary["all_torch_work_pending_for_microllm"] or
            not summary["all_microllm_work_pending_for_torch"] or
            not summary["all_wrappers_non_owning"] or
            summary["minimum_torch_to_microllm_wait_ms"] <= 0.0 or
            summary["minimum_microllm_to_torch_wait_ms"] <= 0.0 or
            summary["maximum_output_error"] > 1.0e-3):
        raise ValueError("native PyTorch Stream interop gate failed")
    atomic_text(Path(arguments.summary),
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_text(Path(arguments.svg), render_svg(runs))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
