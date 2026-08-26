#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks/results/2026-08-26-official-pytorch-hidden-alignment"


def main() -> int:
    runner = ROOT / "benchmarks/single_gpu/hf_pytorch_hidden_alignment.py"
    renderer = ROOT / "docs/optimization-log/scripts/render_official_pytorch_hidden_alignment.py"
    ast.parse(runner.read_text(encoding="utf-8"))
    ast.parse(renderer.read_text(encoding="utf-8"))
    raw = [json.loads(line) for line in
           (RESULTS / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert len(raw) == 2 and summary["status"] == "pass"
    expected = {
        "qwen2.5-0.5b": (24, 27),
        "deepseek-r1-distill-qwen-1.5b": (28, 31),
    }
    for model in summary["summaries"]:
        layers, stages = expected[model["model"]]
        assert model["stage_count"] == stages
        names = [row["name"] for row in model["stages"]]
        assert names == (["inference.embedding"] +
                         [f"inference.blocks.{index}" for index in range(layers)] +
                         ["inference.final_norm", "inference.logits"])
        assert model["stages"][0]["exact"] is True
        assert model["first_nonzero_stage"] == "inference.blocks.0"
        assert model["maximum_relative_l2"] < 5.0e-5
        assert model["logits_max_abs"] < 1.0e-3
        assert model["logits_rms_abs"] < 1.0e-4
        assert all(math.isfinite(row["relative_l2"]) and row["elements"] > 0
                   for row in model["stages"])
    ET.parse(ROOT / "docs/optimization-log/assets/official-pytorch-hidden-alignment.svg")
    print("official PyTorch hidden alignment contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
