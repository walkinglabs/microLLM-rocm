#!/usr/bin/env python3
"""Build exact-length Qwen3 prompt fixtures from the pinned local tokenizer."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


PROMPTS = {
    "english": "The quick brown fox jumps over the lazy dog. In one sentence, explain why this pangram is useful.",
    "chinese": "请用三句话向初中生解释为什么天空看起来是蓝色的。",
    "code": "Write a C++20 function that adds two integers safely and include one simple test:",
}
CHAT_MESSAGES = [{
    "role": "user",
    "content": "Explain KV cache in simple words and give one example.",
}]


def build_manifest(document: dict, tokenized: dict[str, list[int]],
                   model_name: str = "qwen3-0.6b") -> dict:
    models = document.get("models") if document.get("schema_version") == 1 else None
    if not isinstance(models, list):
        raise ValueError("input must be a schema-version-1 model manifest")
    selected = [model for model in models if model.get("name") == model_name]
    if len(selected) != 1:
        raise ValueError(f"input must contain exactly one {model_name}")
    if set(tokenized) != {"english", "chinese", "code", "chat"} or any(
            not tokens or any(type(token) is not int or token < 0 for token in tokens)
            for tokens in tokenized.values()):
        raise ValueError("tokenized prompts must contain four nonnegative token lists")
    source = selected[0]
    variants = []
    for family in ("english", "chinese", "code", "chat"):
        variant = copy.deepcopy(source)
        variant["name"] = f"qwen3-natural-{family}"
        inference = variant.setdefault("inference", {})
        inference["token_ids"] = list(tokenized[family])
        inference["prompt_family"] = family
        inference["exact_context"] = len(tokenized[family])
        inference["prompt_text"] = (
            PROMPTS[family] if family in PROMPTS else CHAT_MESSAGES[0]["content"])
        inference["chat_template"] = family == "chat"
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
    selected = [model for model in document.get("models", [])
                if model.get("name") == args.model]
    if len(selected) != 1:
        parser.error(f"input must contain exactly one {args.model}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Path(selected[0]["config"]).parent, local_files_only=True)
    tokenized = {
        family: tokenizer.encode(text, add_special_tokens=False)
        for family, text in PROMPTS.items()
    }
    tokenized["chat"] = tokenizer.apply_chat_template(
        CHAT_MESSAGES, tokenize=True, add_generation_prompt=True,
        enable_thinking=False)
    result = build_manifest(document, tokenized, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "pass", "output": str(args.output),
        "contexts": {model["inference"]["prompt_family"]:
                     model["inference"]["exact_context"]
                     for model in result["models"]},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
