#!/usr/bin/env python3
"""Gate one batch-selective exact prefill stack against the real upstream route."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "fp32_prefill_exact_stack_base",
    Path(__file__).with_name("fp32_prefill_attention_selective_gate.py"))
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

COMMON = BASE.COMMON
Q_SOLUTION = BASE.Q_SOLUTION
KV_SOLUTION = BASE.KV_SOLUTION
QK_SOLUTION = 304681
PV_SOLUTION = 295716
O_SOLUTION = 296100
POLICIES = BASE.POLICIES
BATCHES = BASE.BATCHES
SELECTIVE = {
    1: {"qk": -1, "pv": -1, "o": -1},
    2: {"qk": QK_SOLUTION, "pv": PV_SOLUTION, "o": O_SOLUTION},
    4: {"qk": QK_SOLUTION, "pv": PV_SOLUTION, "o": O_SOLUTION},
    8: {"qk": QK_SOLUTION, "pv": PV_SOLUTION, "o": -1},
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
        parser.error("exact-stack inputs are outside the measurement contract")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def indices_for(policy: str, batch: int) -> dict[str, int]:
    if policy not in POLICIES or batch not in SELECTIVE:
        raise ValueError("unknown exact-stack policy or batch")
    return (SELECTIVE[batch] if policy == "batch-selective" else
            {"qk": -1, "pv": -1, "o": -1})


def command(args: argparse.Namespace, model: dict, policy: str,
            batch: int, warmup: int,
            cache_output: Path | None = None,
            logits_output: Path | None = None) -> list[str]:
    indices = indices_for(policy, batch)
    result = BASE.QKV.command(
        args, model, "invariant-qkv", batch, warmup,
        cache_output, logits_output)
    for flag, key in (
            ("--fp32-prefill-attention-qk-solution-index", "qk"),
            ("--fp32-prefill-attention-pv-solution-index", "pv"),
            ("--fp32-prefill-attention-o-solution-index", "o")):
        if indices[key] >= 0:
            result.extend([flag, str(indices[key])])
    return result


def require_route(record: dict, policy: str, batch: int,
                  context: int, warmup: int) -> None:
    indices = indices_for(policy, batch)
    extra_entries = sum(value >= 0 for value in indices.values())
    registered = 2 + extra_entries
    hits_per_prefill = 84 + 28 * extra_entries
    expected = {
        "status": "pass", "batch": batch, "token_count": context,
        "decode_tokens": 1, "kv_cache_dtype": "bf16",
        "fp32_prefill_q_solution_index": Q_SOLUTION,
        "fp32_prefill_kv_solution_index": KV_SOLUTION,
        "fp32_prefill_attention_qk_solution_index": indices["qk"],
        "fp32_prefill_attention_pv_solution_index": indices["pv"],
        "fp32_prefill_attention_o_solution_index": indices["o"],
        "fp32_solution_registered_entries": registered,
        "fp32_solution_cached_algorithms": registered,
    }
    for name, wanted in expected.items():
        if record.get(name) != wanted:
            raise ValueError(
                f"{policy} B{batch} {name} expected {wanted!r}, "
                f"got {record.get(name)!r}")
    dispatches = hits_per_prefill * (warmup + 1)
    if (record.get("fp32_solution_registry_hits") != dispatches or
            record.get("fp32_solution_cache_misses") != registered or
            record.get("fp32_solution_cache_hits") != dispatches - registered or
            record.get("fp32_solution_dispatches") != dispatches):
        raise ValueError(f"{policy} B{batch} exact-stack registry counts changed")


def main() -> int:
    args = options()
    model = COMMON.model_entry(args.manifest, args.model)
    config = json.loads(Path(model["config"]).read_text(encoding="utf-8"))
    vocabulary = int(config["vocab_size"])
    args.output_directory.mkdir(parents=True, exist_ok=True)

    BASE.SELECTIVE = SELECTIVE
    BASE.command = command
    BASE.require_route = require_route
    with tempfile.TemporaryDirectory(prefix="microllm-prefill-exact-stack-") as root:
        precision = BASE.precision_phase(args, model, vocabulary, Path(root))
    performance = BASE.performance_phase(args, model)
    for row in precision + performance:
        row["o_solution_index"] = indices_for(
            row["policy"], row["batch"])["o"]
    summary = BASE.summarize(precision, performance)
    summary["record_type"] = "prefill_exact_stack_model_gate"
    summary["selective_indices"] = {
        str(batch): indices for batch, indices in SELECTIVE.items()}
    svg = (BASE.render(summary)
           .replace("Batch-selective prefill Attention · model gate",
                    "Batch-selective exact prefill stack · model gate")
           .replace("B1 default · B2 PV · B4/B8 local QK/PV winners",
                    "B1 upstream · B2/B4 core+O · B8 core-only"))
    for batch in BATCHES:
        old = (f'QK {SELECTIVE[batch]["qk"]} · '
               f'PV {SELECTIVE[batch]["pv"]}')
        svg = svg.replace(old, old + f' · O {SELECTIVE[batch]["o"]}')

    (args.output_directory / "precision-raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in precision),
        encoding="utf-8")
    (args.output_directory / "performance-raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in performance),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "exact-stack-gate.svg").write_text(
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
        print(f"fp32_prefill_exact_stack_gate: {error}", file=sys.stderr)
        raise SystemExit(2) from error
