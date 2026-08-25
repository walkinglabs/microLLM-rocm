#!/usr/bin/env python3
"""Contract tests for load-subtracted training Kernel profiles."""

from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/single_gpu/profile_step_delta.py"


def write_stats(path: pathlib.Path, rows: list[tuple[str, int, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("Name", "Calls", "TotalDurationNs"))
        writer.writeheader()
        for name, calls, duration in rows:
            writer.writerow({"Name": name, "Calls": calls,
                             "TotalDurationNs": duration})


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="microllm-profile-delta-") as temporary:
        root = pathlib.Path(temporary)
        one = root / "one.csv"
        many = root / "many.csv"
        output = root / "output"
        write_stats(one, [
            ("Cijk_fixture", 10, 1000),
            ("adamw_update_bf16_moments_kernel", 5, 500),
            ("causal_softmax_kernel", 3, 300),
            ("load_only_cast_transpose_2d", 7, 700),
        ])
        write_stats(many, [
            ("Cijk_fixture", 14, 1800),
            ("adamw_update_bf16_moments_kernel", 9, 900),
            ("causal_softmax_kernel", 5, 500),
            ("load_only_cast_transpose_2d", 7, 680),
        ])
        command = [
            sys.executable, str(SCRIPT), "--one-step", str(one),
            "--many-step", str(many), "--many-step-count", "3",
            "--output-directory", str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise AssertionError(completed.stderr or completed.stdout)
        result = json.loads((output / "profile-delta.json").read_text(
            encoding="utf-8"))
        categories = {row["category"]: row for row in result["categories"]}
        assert result["derived_steps"] == 2
        assert result["negative_call_delta_names"] == []
        assert result["excluded_nonpositive_delta_names"] == [
            "load_only_cast_transpose_2d"]
        assert categories["hipBLASLt GEMM"]["calls_per_step"] == 2
        assert categories["AdamW"]["duration_ns_per_step"] == 200
        assert categories["softmax"]["duration_ns_per_step"] == 100
        assert (output / "one-step-kernel-stats.csv").is_file()
        assert (output / "three-step-kernel-stats.csv").is_file()
        chart = output / "profile-delta.svg"
        assert chart.is_file()
        root = ET.parse(chart).getroot()
        assert root.tag.endswith("svg")
        chart_text = chart.read_text(encoding="utf-8")
        assert "hipBLASLt GEMM" in chart_text
        assert "Load-subtracted GPU kernel profile" in chart_text

        write_stats(many, [("Cijk_fixture", 9, 900)])
        rejected = subprocess.run(command, capture_output=True, text=True)
        assert rejected.returncode == 2
        assert "negative Kernel call deltas" in rejected.stderr
    print("training profile delta contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
