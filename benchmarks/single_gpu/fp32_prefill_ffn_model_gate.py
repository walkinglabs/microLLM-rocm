#!/usr/bin/env python3
"""Gate batch-selective exact FP32 prefill FFN gate/up projections."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "fp32_prefill_ffn_model_base",
    Path(__file__).with_name("fp32_prefill_attention_selective_gate.py"))
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

COMMON = BASE.COMMON
POLICIES = BASE.POLICIES
BATCHES = BASE.BATCHES
SOLUTION = 296100
SELECTIVE = {
    1: {"qk": -1, "pv": -1, "ffn": SOLUTION},
    2: {"qk": -1, "pv": -1, "ffn": SOLUTION},
    4: {"qk": -1, "pv": -1, "ffn": -1},
    8: {"qk": -1, "pv": -1, "ffn": SOLUTION},
}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--performance-warmup", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args()
    if (not args.manifest.is_file() or not args.binary.is_file() or
            args.context != 2048 or args.runs != 2 or
            args.performance_warmup != 1 or args.timeout_seconds <= 0):
        parser.error("prefill FFN model-gate inputs are outside the contract")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def index_for(policy: str, batch: int) -> int:
    if policy not in POLICIES or batch not in SELECTIVE:
        raise ValueError("unknown prefill FFN policy or batch")
    return SELECTIVE[batch]["ffn"] if policy == "batch-selective" else -1


def command(args: argparse.Namespace, model: dict, policy: str,
            batch: int, warmup: int,
            cache_output: Path | None = None,
            logits_output: Path | None = None) -> list[str]:
    result = BASE.QKV.command(
        args, model, "default", batch, warmup, cache_output, logits_output)
    index = index_for(policy, batch)
    if index >= 0:
        result.extend([
            "--fp32-prefill-ffn-gate-up-solution-index", str(index),
        ])
    return result


def require_route(record: dict, policy: str, batch: int,
                  context: int, warmup: int) -> None:
    index = index_for(policy, batch)
    candidate = index >= 0
    expected = {
        "status": "pass", "batch": batch, "token_count": context,
        "decode_tokens": 1, "kv_cache_dtype": "bf16",
        "fp32_prefill_q_solution_index": -1,
        "fp32_prefill_kv_solution_index": -1,
        "fp32_prefill_attention_qk_solution_index": -1,
        "fp32_prefill_attention_pv_solution_index": -1,
        "fp32_prefill_attention_o_solution_index": -1,
        "fp32_prefill_ffn_gate_up_solution_index": index,
        "fp32_solution_registered_entries": 1 if candidate else 0,
        "fp32_solution_cached_algorithms": 1 if candidate else 0,
    }
    for name, wanted in expected.items():
        if record.get(name) != wanted:
            raise ValueError(
                f"{policy} B{batch} {name} expected {wanted!r}, "
                f"got {record.get(name)!r}")
    if candidate:
        dispatches = 56 * (warmup + 1)
        if (record.get("fp32_solution_registry_hits") != dispatches or
                record.get("fp32_solution_cache_misses") != 1 or
                record.get("fp32_solution_cache_hits") != dispatches - 1 or
                record.get("fp32_solution_dispatches") != dispatches):
            raise ValueError(f"{policy} B{batch} FFN registry counts changed")


def rename_policy(value: str) -> str:
    return "selective-ffn-exact" if value == "batch-selective" else "upstream"


def main() -> int:
    args = options()
    model = COMMON.model_entry(args.manifest, args.model)
    config = json.loads(Path(model["config"]).read_text(encoding="utf-8"))
    vocabulary = int(config["vocab_size"])
    args.output_directory.mkdir(parents=True, exist_ok=True)

    BASE.SELECTIVE = SELECTIVE
    BASE.command = command
    BASE.require_route = require_route
    with tempfile.TemporaryDirectory(prefix="microllm-prefill-ffn-model-") as root:
        precision = BASE.precision_phase(args, model, vocabulary, Path(root))
    performance = BASE.performance_phase(args, model)
    for row in precision + performance:
        row["ffn_gate_up_solution_index"] = index_for(
            row["policy"], row["batch"])
    summary = BASE.summarize(precision, performance)
    summary["record_type"] = "prefill_ffn_gate_up_model_gate"
    summary["selective_indices"] = {
        str(batch): values for batch, values in SELECTIVE.items()}
    svg = (BASE.render(summary)
           .replace("Batch-selective prefill Attention · model gate",
                    "Batch-selective prefill FFN gate/up · model gate")
           .replace("B1 default · B2 PV · B4/B8 local QK/PV winners",
                    "B1/B2/B8 exact gate+up · B4 upstream"))
    for batch in BATCHES:
        old = f'QK -1 · PV -1'
        svg = svg.replace(
            old, f'FFN {SELECTIVE[batch]["ffn"]}', 1)
    for rows in (precision, performance):
        for row in rows:
            row["policy"] = rename_policy(row["policy"])
    for row in summary["cases"]:
        row["policy"] = rename_policy(row["policy"])
    for row in summary["policy_summaries"]:
        row["policy"] = rename_policy(row["policy"])

    (args.output_directory / "precision-raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in precision),
        encoding="utf-8")
    (args.output_directory / "performance-raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in performance),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "ffn-model-gate.svg").write_text(
        svg, encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"cases", "policy_summaries"}},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"fp32_prefill_ffn_model_gate: {error}", file=sys.stderr)
        raise SystemExit(2) from error
