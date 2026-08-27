#!/usr/bin/env python3
"""Clone one Qwen3 fixture entry into four fixed prompt-content patterns."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


PATTERNS = {
    "constant": [1],
    "alternating": [1, 198, 374, 279, 264],
    "ascending": [1, 2, 3, 4, 5, 6, 7, 8],
    "sensitive": [14582, 198, 262, 1096, 374, 279, 2038, 4285],
}


def build_manifest(document: dict, model_name: str = "qwen3-0.6b") -> dict:
    models = document.get("models") if document.get("schema_version") == 1 else None
    if not isinstance(models, list):
        raise ValueError("input must be a schema-version-1 model manifest")
    selected = [model for model in models if model.get("name") == model_name]
    if len(selected) != 1:
        raise ValueError(f"input must contain exactly one {model_name}")
    source = selected[0]
    if not isinstance(source.get("inference"), dict):
        raise ValueError("source model must contain inference metadata")
    variants = []
    for pattern, tokens in PATTERNS.items():
        variant = copy.deepcopy(source)
        variant["name"] = f"qwen3-{pattern}"
        variant["inference"]["token_ids"] = list(tokens)
        variant["inference"]["prompt_pattern"] = pattern
        variants.append(variant)
    return {"schema_version": 1, "models": variants}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3-0.6b")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input manifest does not exist: {args.input}")
    if args.output.resolve() == args.input.resolve():
        parser.error("output must differ from input")
    document = json.loads(args.input.read_text(encoding="utf-8"))
    result = build_manifest(document, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "pass", "output": str(args.output),
        "models": [model["name"] for model in result["models"]],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
