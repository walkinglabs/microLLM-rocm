#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_jsonl  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Compare microLLM and PyTorch alignment traces")
    parser.add_argument("--microllm", required=True, type=Path)
    parser.add_argument("--pytorch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=3.0e-5)
    parser.add_argument("--rtol", type=float, default=3.0e-5)
    parser.add_argument("--allow-truncated", action="store_true")
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0:
        parser.error("tolerances must be non-negative")
    return args


def occurrence_records(records):
    counts = defaultdict(int)
    output = {}
    for record in records:
        base = (record["kind"], record["name"])
        occurrence = counts[base]
        counts[base] += 1
        output[(*base, occurrence)] = record
    return output


def compare_values(left, right, atol, rtol, allow_truncated):
    left_map = occurrence_records(left)
    right_map = occurrence_records(right)
    results = []
    all_keys = sorted(set(left_map) | set(right_map))
    for key in all_keys:
        lhs = left_map.get(key)
        rhs = right_map.get(key)
        result = {
            "kind": key[0],
            "name": key[1],
            "occurrence": key[2],
            "status": "pass",
        }
        if lhs is None or rhs is None:
            result["status"] = "missing"
            result["missing_framework"] = "microllm" if lhs is None else "pytorch"
            results.append(result)
            continue
        result["microllm_shape"] = lhs["shape"]
        result["pytorch_shape"] = rhs["shape"]
        result["microllm_dtype"] = lhs["dtype"]
        result["pytorch_dtype"] = rhs["dtype"]
        if lhs["shape"] != rhs["shape"] or lhs["dtype"] != rhs["dtype"]:
            result["status"] = "metadata_mismatch"
            results.append(result)
            continue
        truncated = lhs["values_truncated"] or rhs["values_truncated"]
        result["partial_values"] = truncated
        if truncated and not allow_truncated:
            result["status"] = "truncated"
            results.append(result)
            continue
        left_values = lhs["values"]
        right_values = rhs["values"]
        if len(left_values) != len(right_values):
            result["status"] = "value_count_mismatch"
            result["microllm_value_count"] = len(left_values)
            result["pytorch_value_count"] = len(right_values)
            results.append(result)
            continue
        if not left_values:
            result.update(max_abs=0.0, max_rel=0.0, mse=0.0, cosine=1.0,
                          maximum_error_index=None)
            results.append(result)
            continue
        numeric = []
        special_mismatches = []
        for index, (actual, reference) in enumerate(zip(left_values, right_values)):
            if isinstance(actual, (int, float)) and isinstance(reference, (int, float)):
                numeric.append((index, float(actual), float(reference)))
            elif actual != reference:
                special_mismatches.append(index)
        differences = [(index, abs(actual - reference))
                       for index, actual, reference in numeric]
        maximum_index, maximum_difference = max(
            differences, key=lambda item: item[1], default=(None, 0.0))
        relative = [difference / max(abs(reference), 1.0e-12)
                    for (_, difference), (_, _, reference) in zip(differences, numeric)]
        mse = (sum(difference * difference for _, difference in differences) /
               len(differences) if differences else 0.0)
        dot = sum(actual * reference for _, actual, reference in numeric)
        left_norm = math.sqrt(sum(actual * actual for _, actual, _ in numeric))
        right_norm = math.sqrt(sum(reference * reference for _, _, reference in numeric))
        cosine = dot / (left_norm * right_norm) if left_norm and right_norm else (
            1.0 if left_norm == right_norm else 0.0)
        within = not special_mismatches and all(
            difference <= atol + rtol * abs(reference)
            for (_, difference), (_, _, reference) in zip(differences, numeric))
        diagnostic_index = special_mismatches[0] if special_mismatches else maximum_index
        result.update(
            status="pass" if within else "numeric_mismatch",
            max_abs=maximum_difference,
            max_rel=max(relative, default=0.0),
            mse=mse,
            cosine=cosine,
            maximum_error_index=diagnostic_index,
            microllm_at_max=(left_values[diagnostic_index]
                             if diagnostic_index is not None else None),
            pytorch_at_max=(right_values[diagnostic_index]
                            if diagnostic_index is not None else None),
            special_value_mismatches=len(special_mismatches),
        )
        results.append(result)
    return results


def percentile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def timing_groups(records):
    per_iteration_counts = defaultdict(int)
    values = defaultdict(list)
    for record in records:
        base = (record["iteration"], record["kind"], record["name"])
        occurrence = per_iteration_counts[base]
        per_iteration_counts[base] += 1
        values[(record["kind"], record["name"], occurrence)].append(record["wall_ms"])
    return values


def timing_summary(records):
    output = {}
    for key, values in timing_groups(records).items():
        output[key] = {
            "samples": len(values),
            "minimum_ms": min(values),
            "mean_ms": statistics.fmean(values),
            "median_ms": statistics.median(values),
            "p95_ms": percentile(values, 0.95),
            "maximum_ms": max(values),
        }
    return output


def compare_timings(left, right):
    left_summary = timing_summary(left)
    right_summary = timing_summary(right)
    results = []
    for key in sorted(set(left_summary) | set(right_summary)):
        lhs = left_summary.get(key)
        rhs = right_summary.get(key)
        result = {"kind": key[0], "name": key[1], "occurrence": key[2]}
        if lhs is None or rhs is None:
            result["status"] = "missing"
            result["missing_framework"] = "microllm" if lhs is None else "pytorch"
        else:
            result["status"] = "measured"
            result["microllm"] = lhs
            result["pytorch"] = rhs
            micro_median = lhs["median_ms"]
            result["pytorch_over_microllm"] = (
                rhs["median_ms"] / micro_median if micro_median > 0 else None)
        results.append(result)
    return results


def markdown_report(report):
    summary = report["summary"]
    lines = [
        "# microLLM / PyTorch alignment report",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Status: **{summary['status']}**",
        f"- Value checkpoints: {summary['value_records']}",
        f"- Passed checkpoints: {summary['value_passed']}",
        f"- Failed or missing checkpoints: {summary['value_failed']}",
        f"- Tolerance: `atol={report['atol']}`, `rtol={report['rtol']}`",
        "",
        "## Largest numerical differences",
        "",
        "| Kind | Name | # | Status | Max abs | Max rel | Cosine |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    numerical = [item for item in report["values"] if "max_abs" in item]
    numerical.sort(key=lambda item: item["max_abs"], reverse=True)
    for item in numerical[:20]:
        lines.append(
            f"| {item['kind']} | `{item['name']}` | {item['occurrence']} | "
            f"{item['status']} | {item['max_abs']:.6g} | {item['max_rel']:.6g} | "
            f"{item['cosine']:.8f} |"
        )
    for title, key in (("Operator timing", "operator_timings"),
                       ("Layer and model timing", "layer_timings"),
                       ("Backward timing", "backward_timings")):
        lines += ["", f"## {title}", "",
                  "| Kind | Name | # | microLLM median ms | PyTorch median ms | PyTorch/microLLM |",
                  "|---|---|---:|---:|---:|---:|"]
        measured = [item for item in report[key] if item["status"] == "measured"]
        measured.sort(key=lambda item: item["microllm"]["median_ms"], reverse=True)
        for item in measured[:30]:
            ratio = item["pytorch_over_microllm"]
            ratio_text = f"{ratio:.4g}" if ratio is not None else "n/a"
            lines.append(
                f"| {item['kind']} | `{item['name']}` | {item['occurrence']} | "
                f"{item['microllm']['median_ms']:.6g} | "
                f"{item['pytorch']['median_ms']:.6g} | {ratio_text} |"
            )
    lines += ["", "Positive PyTorch/microLLM values greater than 1 mean microLLM was faster for that measured checkpoint.", ""]
    return "\n".join(lines)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    micro_run = json.loads((args.microllm / "microllm_run.json").read_text())
    torch_run = json.loads((args.pytorch / "pytorch_run.json").read_text())
    if micro_run["run_id"] != torch_run["run_id"]:
        raise ValueError("run IDs do not match")

    values = compare_values(
        load_jsonl(args.microllm / "microllm_values.jsonl"),
        load_jsonl(args.pytorch / "pytorch_values.jsonl"),
        args.atol, args.rtol, args.allow_truncated,
    )
    training_values = compare_values(
        load_jsonl(args.microllm / "microllm_training_values.jsonl"),
        load_jsonl(args.pytorch / "pytorch_training_values.jsonl"),
        args.atol, args.rtol, args.allow_truncated,
    )
    operator_timings = compare_timings(
        load_jsonl(args.microllm / "microllm_operator_timing.jsonl"),
        load_jsonl(args.pytorch / "pytorch_operator_timing.jsonl"),
    )
    layer_timings = compare_timings(
        load_jsonl(args.microllm / "microllm_layer_timing.jsonl"),
        load_jsonl(args.pytorch / "pytorch_layer_timing.jsonl"),
    )
    backward_timings = compare_timings(
        load_jsonl(args.microllm / "microllm_backward_timing.jsonl"),
        load_jsonl(args.pytorch / "pytorch_backward_timing.jsonl"),
    )
    values += training_values
    failures = [item for item in values if item["status"] != "pass"]
    report = {
        "schema_version": 1,
        "run_id": micro_run["run_id"],
        "atol": args.atol,
        "rtol": args.rtol,
        "microllm_run": micro_run,
        "pytorch_run": torch_run,
        "summary": {
            "status": "pass" if not failures else "fail",
            "value_records": len(values),
            "value_passed": len(values) - len(failures),
            "value_failed": len(failures),
            "operator_timing_records": len(operator_timings),
            "layer_timing_records": len(layer_timings),
            "backward_timing_records": len(backward_timings),
        },
        "values": values,
        "operator_timings": operator_timings,
        "layer_timings": layer_timings,
        "backward_timings": backward_timings,
    }
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "report.md").write_text(markdown_report(report))
    print(f"comparison_output={args.output}")
    print(f"value_records={len(values)}")
    print(f"value_failed={len(failures)}")
    print(f"status={report['summary']['status']}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
