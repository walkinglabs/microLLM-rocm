#!/usr/bin/env python3
"""Gate scoped O=296100 on top of an exact diagnostic Attention core."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "fp32_prefill_o_model_base",
    Path(__file__).with_name("fp32_prefill_attention_model_gate.py"))
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

COMMON = BASE.COMMON
BASE_COMMAND = BASE.command
O_SOLUTION = 296100


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
        parser.error("prefill O model-gate inputs are outside the contract")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def command(args: argparse.Namespace, model: dict, policy: str, batch: int,
            warmup: int, cache_output: Path | None = None,
            logits_output: Path | None = None,
            trace_output: Path | None = None,
            binary_directory: Path | None = None) -> list[str]:
    if policy not in BASE.POLICIES:
        raise ValueError("unknown O model policy")
    result = BASE_COMMAND(
        args, model, "attention-exact", batch, warmup,
        cache_output, logits_output, trace_output, binary_directory)
    if policy == "attention-exact":
        result.extend([
            "--fp32-prefill-attention-o-solution-index", str(O_SOLUTION),
        ])
    return result


def require_route(record: dict, policy: str, batch: int,
                  context: int, warmup: int) -> None:
    candidate = policy == "attention-exact"
    expected = {
        "status": "pass", "batch": batch, "token_count": context,
        "decode_tokens": 1, "kv_cache_dtype": "bf16",
        "fp32_prefill_q_solution_index": BASE.Q_SOLUTION,
        "fp32_prefill_kv_solution_index": BASE.KV_SOLUTION,
        "fp32_prefill_attention_qk_solution_index": BASE.QK_SOLUTION,
        "fp32_prefill_attention_pv_solution_index": BASE.PV_SOLUTION,
        "fp32_prefill_attention_o_solution_index": O_SOLUTION if candidate else -1,
        "fp32_solution_registered_entries": 5 if candidate else 4,
        "fp32_solution_cached_algorithms": 5 if candidate else 4,
    }
    for name, wanted in expected.items():
        if record.get(name) != wanted:
            raise ValueError(
                f"{policy} B{batch} {name} expected {wanted!r}, "
                f"got {record.get(name)!r}")
    dispatches = (168 if candidate else 140) * (warmup + 1)
    entries = 5 if candidate else 4
    if (record.get("fp32_solution_registry_hits") != dispatches or
            record.get("fp32_solution_cache_misses") != entries or
            record.get("fp32_solution_cache_hits") != dispatches - entries or
            record.get("fp32_solution_dispatches") != dispatches):
        raise ValueError(f"{policy} B{batch} O registry counts changed")


def rename_policy(value: str) -> str:
    return "exact-core-o" if value == "attention-exact" else "exact-core"


def main() -> int:
    args = options()
    model = COMMON.model_entry(args.manifest, args.model)
    config = json.loads(Path(model["config"]).read_text(encoding="utf-8"))
    vocabulary = int(config["vocab_size"])
    args.output_directory.mkdir(parents=True, exist_ok=True)

    BASE.command = command
    BASE.require_route = require_route
    with tempfile.TemporaryDirectory(prefix="microllm-prefill-o-model-") as root:
        precision = BASE.precision_phase(args, model, vocabulary, Path(root))
    performance = BASE.performance_phase(args, model)
    summary = BASE.summarize(precision, performance)
    svg = (BASE.render(summary)
           .replace("Scoped exact Attention · complete model gate",
                    "Scoped exact O projection · complete model gate")
           .replace("upstream Q/K/V fixed in both policies",
                    "Q/K/V/QK/PV fixed in both policies"))
    summary["record_type"] = "prefill_o_model_gate"
    summary["o_solution_index"] = O_SOLUTION
    for row in precision:
        row["policy"] = rename_policy(row["policy"])
    for row in performance:
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
    (args.output_directory / "o-model-gate.svg").write_text(svg, encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"cases", "policy_summaries"}},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"fp32_prefill_o_model_gate: {error}", file=sys.stderr)
        raise SystemExit(2) from error
