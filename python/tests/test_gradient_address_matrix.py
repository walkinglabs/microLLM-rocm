#!/usr/bin/env python3
"""Contract test for model gradient Storage address summaries."""

from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "benchmarks/single_gpu/gradient_address_matrix.py"
SPEC = importlib.util.spec_from_file_location("gradient_address_matrix", MODULE_PATH)
MATRIX = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MATRIX)


def row(model: str, precision: str, context: int) -> dict:
    parameters, tensors = MATRIX.EXPECTED[model]
    changed = []
    changed_bytes = 0
    if model == "tiny":
        changed = [
            f"blocks.{block}.attention.{projection}_proj.weight"
            for block in range(2) for projection in ("k", "v")]
        changed_bytes = 8192
    elif model == "deepseek" and context == 512:
        changed = ([f"blocks.{index}.attention.q_proj.weight"
                    for index in range(112)] +
                   [f"blocks.{index}.feed_forward.up_proj.weight"
                    for index in range(84)] +
                   ["token_embedding.weight", "output_head.weight"])
        changed_bytes = 7107772416
    records = []
    for index in range(tensors):
        name = changed[index] if index < len(changed) else f"stable.{index}.weight"
        records.append({
            "name": name, "elements": 1, "bytes": 4,
            "observations": 2, "address_changes": 1 if index < len(changed) else 0,
            "address_stable": index >= len(changed),
            "minimum_storage_use_count": 2,
            "maximum_storage_use_count": 2,
        })
    return {
        "model": model, "precision": precision, "context": context,
        "parameter_count": parameters, "parameter_tensors": tensors,
        "stable_gradient_tensors": tensors - len(changed),
        "changed_gradient_tensors": len(changed),
        "stable_gradient_bytes": parameters * 4 - changed_bytes,
        "changed_gradient_bytes": changed_bytes,
        "all_gradient_addresses_stable": not changed,
        "elapsed_ms": 1.0, "engine_peak_bytes": 2,
        "records": records,
    }


class GradientAddressMatrixTest(unittest.TestCase):
    def test_summary_keeps_qwen_and_rejects_deepseek_t512(self):
        records = []
        for run in range(1, 4):
            for model, precision, context in MATRIX.CASES:
                value = row(model, precision, context)
                value["process_run"] = run
                records.append(value)
        summary = MATRIX.summarize(records, 3)
        self.assertEqual(summary["status"], "pass")
        self.assertTrue(summary["gates"][
            "qwen_t8_t512_all_addresses_stable"])
        self.assertTrue(summary["gates"][
            "deepseek_t512_counterexample_present"])
        deep = next(item for item in summary["comparisons"]
                    if item["model"] == "deepseek" and item["context"] == 512)
        self.assertEqual(deep["changed_gradient_tensors"], 198)
        self.assertIn("stable gradients or recapture", summary["decision"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
