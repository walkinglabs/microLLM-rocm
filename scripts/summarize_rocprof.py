#!/usr/bin/env python3
"""Aggregate rocprofv3 kernel and HIP API trace CSV files without pandas."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-trace", required=True, type=Path)
    parser.add_argument("--hip-api-trace", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    result = parser.parse_args()
    if not result.kernel_trace.is_file() or not result.hip_api_trace.is_file():
        parser.error("both trace CSV files must exist")
    return result


def aggregate(path: Path, name_column: str):
    groups = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            duration = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
            if duration < 0:
                raise RuntimeError(f"negative duration in {path}")
            groups[row[name_column]].append(duration)
    total = sum(sum(values) for values in groups.values())
    rows = []
    for name, values in groups.items():
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        rows.append({
            "Name": name,
            "Calls": len(values),
            "TotalDurationNs": sum(values),
            "AverageNs": mean,
            "Percentage": 100.0 * sum(values) / total if total else 0.0,
            "MinNs": min(values),
            "MaxNs": max(values),
            "StdDev": math.sqrt(variance),
        })
    return sorted(rows, key=lambda row: row["TotalDurationNs"], reverse=True)


def write(path: Path, rows):
    columns = ("Name", "Calls", "TotalDurationNs", "AverageNs", "Percentage",
               "MinNs", "MaxNs", "StdDev")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    kernels = aggregate(args.kernel_trace, "Kernel_Name")
    hip_api = aggregate(args.hip_api_trace, "Function")
    write(args.output_directory / "kernel-stats.csv", kernels)
    write(args.output_directory / "hip-api-stats.csv", hip_api)
    print(f"kernel_dispatches={sum(row['Calls'] for row in kernels)}")
    print(f"hip_api_calls={sum(row['Calls'] for row in hip_api)}")


if __name__ == "__main__":
    main()
