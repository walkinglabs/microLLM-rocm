#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks/results/2026-08-26-official-hf-fixtures"


def main() -> int:
    ast.parse((ROOT / "tools/prepare_hf_fixture.py").read_text(encoding="utf-8"))
    ast.parse((ROOT / "docs/optimization-log/scripts/render_official_hf_fixtures.py")
              .read_text(encoding="utf-8"))
    evidence = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "pass" and len(evidence["models"]) == 2
    expected = {
        "qwen2.5-0.5b": (494032768, 290, "Apache-2.0"),
        "deepseek-r1-distill-qwen-1.5b": (1777088000, 339, "MIT"),
    }
    for model in evidence["models"]:
        parameters, tensors, license_id = expected[model["name"]]
        assert model["parameter_count"] == parameters
        assert model["tensor_count"] == tensors
        assert model["license"] == license_id
        assert model["revision"] and len(model["revision"]) == 40
        assert model["required_files_present"] is True
        assert model["weight_dtypes"] == ["BF16"]
        assert model["weight_bytes"] > 0
    ET.parse(ROOT / "docs/optimization-log/assets/official-hf-fixtures.svg")
    print("official Hugging Face fixture evidence: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
