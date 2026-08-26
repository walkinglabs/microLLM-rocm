#!/usr/bin/env python3
"""Prepare or validate pinned Hugging Face fixtures without committing model payloads."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "data/model_fixtures.toml"
ALLOW_PATTERNS = (
    "config.json", "generation_config.json", "model.safetensors",
    "model.safetensors.index.json", "model-*.safetensors", "vocab.json",
    "merges.txt", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "LICENSE", "README.md",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subcommands = result.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare")
    prepare.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--evidence", type=Path)
    prepare.add_argument("--models")
    prepare.add_argument("--download-root", type=Path)
    prepare.add_argument("--model-source", action="append", default=[])
    prepare.add_argument("--tokenizer-source", action="append", default=[])
    validate = subcommands.add_parser("validate")
    validate.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    validate.add_argument("--manifest", type=Path, required=True)
    return result


def load_registry(path: Path) -> dict[str, dict]:
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise RuntimeError("model fixture registry schema is unsupported")
    rows = document.get("model", [])
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows) or not result:
        raise RuntimeError("model fixture ids must be unique and non-empty")
    required = {
        "repo_id", "source", "revision", "license", "license_url",
        "parameter_count", "tensor_count", "tokenizer_family",
        "inference_token_ids", "expected_generated_tokens", "training_tokens",
    }
    for identifier, row in result.items():
        missing = sorted(required - set(row))
        if missing:
            raise RuntimeError(f"{identifier} registry fields missing: {missing}")
        if len(row["revision"]) != 40 or any(
                character not in "0123456789abcdef" for character in row["revision"]):
            raise RuntimeError(f"{identifier} revision must be a pinned 40-character commit")
        stored_count = row["parameter_count"]
        if type(stored_count) is not int or stored_count <= 0:
            raise RuntimeError(
                f"{identifier} stored parameter count must be positive")
        runtime_count = row.get("runtime_parameter_count", row["parameter_count"])
        if type(runtime_count) is not int or runtime_count <= 0:
            raise RuntimeError(
                f"{identifier} runtime parameter count must be positive")
        if runtime_count > stored_count:
            raise RuntimeError(
                f"{identifier} runtime parameter count cannot exceed stored count")
    return result


def assignments(values: list[str], name: str) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise RuntimeError(f"{name} must use MODEL=/path syntax")
        identifier, path = value.split("=", 1)
        if not identifier or not path or identifier in result:
            raise RuntimeError(f"invalid duplicate {name}: {value}")
        result[identifier] = Path(path).resolve()
    return result


def inspect_safetensors(path: Path) -> dict[str, tuple[list[int], str]]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f"safetensors prefix is truncated: {path}")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes == 0 or header_bytes > 100_000_000:
            raise RuntimeError(f"safetensors header length is invalid: {path}")
        payload = stream.read(header_bytes)
        if len(payload) != header_bytes:
            raise RuntimeError(f"safetensors header is truncated: {path}")
    document = json.loads(payload)
    tensors = {}
    for name, row in document.items():
        if name == "__metadata__":
            continue
        shape = row.get("shape")
        dtype = row.get("dtype")
        offsets = row.get("data_offsets")
        if (not isinstance(shape, list) or not isinstance(dtype, str) or
                not isinstance(offsets, list) or len(offsets) != 2 or
                any(not isinstance(value, int) or value < 0 for value in shape) or
                offsets[0] < 0 or offsets[1] < offsets[0]):
            raise RuntimeError(f"invalid tensor metadata for {name} in {path}")
        if name in tensors:
            raise RuntimeError(f"duplicate tensor in {path}: {name}")
        tensors[name] = (shape, dtype)
    if not tensors:
        raise RuntimeError(f"safetensors contains no tensors: {path}")
    return tensors


def inspect_weight_set(model_root: Path) -> tuple[Path, dict, list[Path]]:
    single = model_root / "model.safetensors"
    index = model_root / "model.safetensors.index.json"
    if single.is_file():
        return single.resolve(), inspect_safetensors(single), [single.resolve()]
    if not index.is_file():
        raise RuntimeError(f"no model.safetensors or index under {model_root}")
    document = json.loads(index.read_text(encoding="utf-8"))
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"weight index has no weight_map: {index}")
    shard_names = sorted(set(weight_map.values()))
    if any(not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
           for name in shard_names):
        raise RuntimeError(f"weight index contains an unsafe shard path: {index}")
    tensors = {}
    actual_weight_map = {}
    shards = []
    for shard_name in shard_names:
        shard = (model_root / shard_name).resolve()
        if not shard.is_file():
            raise RuntimeError(f"weight shard is missing: {shard}")
        shards.append(shard)
        for name, metadata in inspect_safetensors(shard).items():
            if name in tensors:
                raise RuntimeError(f"tensor occurs in multiple shards: {name}")
            tensors[name] = metadata
            actual_weight_map[name] = shard_name
    if set(weight_map) != set(tensors):
        raise RuntimeError("weight index membership differs from shard tensors")
    if any(weight_map[name] != actual_weight_map[name] for name in weight_map):
        raise RuntimeError("weight index maps a tensor to the wrong shard")
    return index.resolve(), tensors, shards


def download_model(row: dict, root: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "download mode requires huggingface_hub; install it in an isolated environment") \
            from error
    destination = (root / row["id"]).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=row["repo_id"], revision=row["revision"],
        local_dir=destination, allow_patterns=list(ALLOW_PATTERNS))
    return destination


def prepare_one(row: dict, model_root: Path, tokenizer_root: Path) -> tuple[dict, dict]:
    config = model_root / "config.json"
    vocab = tokenizer_root / "vocab.json"
    merges = tokenizer_root / "merges.txt"
    for path in (config, vocab, merges):
        if not path.is_file():
            raise RuntimeError(f"required fixture file is missing: {path}")
    json.loads(config.read_text(encoding="utf-8"))
    json.loads(vocab.read_text(encoding="utf-8"))
    weight_entry, tensors, shards = inspect_weight_set(model_root)
    stored_parameter_count = sum(
        math.prod(shape) for shape, _dtype in tensors.values())
    if stored_parameter_count != row["parameter_count"]:
        raise RuntimeError(
            f"{row['id']} parameter count {stored_parameter_count} != {row['parameter_count']}")
    runtime_parameter_count = row.get(
        "runtime_parameter_count", stored_parameter_count)
    if len(tensors) != row["tensor_count"]:
        raise RuntimeError(
            f"{row['id']} tensor count {len(tensors)} != {row['tensor_count']}")
    optional = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "generation_config.json", "added_tokens.json"):
        candidates = (tokenizer_root / name, model_root / name)
        selected = next((path for path in candidates if path.is_file()), None)
        if selected is not None:
            optional[name.removesuffix(".json")] = str(selected.resolve())
    manifest = {
        "name": row["id"],
        "repo_id": row["repo_id"],
        "source": row["source"],
        "revision": row["revision"],
        "license": row["license"],
        "license_url": row["license_url"],
        "state": "fixture-ready",
        "parameter_count": runtime_parameter_count,
        "runtime_parameter_count": runtime_parameter_count,
        "stored_parameter_count": stored_parameter_count,
        "loaded_tensors": len(tensors),
        "config": str(config.resolve()),
        "weights": str(weight_entry),
        "vocab": str(vocab.resolve()),
        "merges": str(merges.resolve()),
        "tokenizer_family": row["tokenizer_family"],
        "inference": {
            "token_ids": row["inference_token_ids"],
            "new_tokens": len(row["expected_generated_tokens"]),
            "expected_generated_tokens": row["expected_generated_tokens"],
        },
        "training": {"tokens": row["training_tokens"]},
        **optional,
    }
    evidence = {
        "name": row["id"],
        "repo_id": row["repo_id"],
        "revision": row["revision"],
        "license": row["license"],
        "state": "fixture-ready",
        "parameter_count": stored_parameter_count,
        "runtime_parameter_count": runtime_parameter_count,
        "stored_parameter_count": stored_parameter_count,
        "tensor_count": len(tensors),
        "weight_file_count": len(shards),
        "weight_bytes": sum(path.stat().st_size for path in shards),
        "vocab_bytes": vocab.stat().st_size,
        "merges_bytes": merges.stat().st_size,
        "weight_dtypes": sorted({dtype for _shape, dtype in tensors.values()}),
        "required_files_present": True,
    }
    return manifest, evidence


def validate_manifest(path: Path, registry: dict[str, dict]) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("models"), list):
        raise RuntimeError("fixture manifest schema is invalid")
    seen = set()
    evidence = []
    for model in document["models"]:
        identifier = model.get("name")
        if identifier in seen or identifier not in registry:
            raise RuntimeError(f"fixture manifest model is duplicate or unknown: {identifier}")
        seen.add(identifier)
        row = registry[identifier]
        prepared, record = prepare_one(
            row, Path(model["config"]).resolve().parent,
            Path(model["vocab"]).resolve().parent)
        for key in ("revision", "parameter_count", "runtime_parameter_count",
                    "stored_parameter_count",
                    "loaded_tensors", "config", "weights",
                    "vocab", "merges", "tokenizer_family", "license"):
            if model.get(key) != prepared.get(key):
                raise RuntimeError(f"fixture manifest drift at {identifier}.{key}")
        evidence.append(record)
    return {"schema_version": 1, "status": "pass", "models": evidence}


def main() -> int:
    args = parser().parse_args()
    registry = load_registry(args.registry)
    if args.command == "validate":
        result = validate_manifest(args.manifest, registry)
        print(json.dumps(result, sort_keys=True))
        return 0

    selected = args.models.split(",") if args.models else list(registry)
    if len(selected) != len(set(selected)) or any(name not in registry for name in selected):
        raise RuntimeError("selected fixture model ids are duplicate or unknown")
    model_sources = assignments(args.model_source, "model-source")
    tokenizer_sources = assignments(args.tokenizer_source, "tokenizer-source")
    manifests = []
    evidence = []
    for identifier in selected:
        row = registry[identifier]
        model_root = model_sources.get(identifier)
        tokenizer_root = tokenizer_sources.get(identifier)
        if model_root is None:
            if args.download_root is None:
                raise RuntimeError(
                    f"{identifier} needs --model-source or --download-root")
            model_root = download_model(row, args.download_root)
        if tokenizer_root is None:
            tokenizer_root = model_root
        prepared, record = prepare_one(row, model_root, tokenizer_root)
        manifests.append(prepared)
        evidence.append(record)
    document = {"schema_version": 1, "models": manifests}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence_document = {
        "schema_version": 1, "status": "pass", "models": evidence,
        "boundary": "pinned local fixture metadata; model/tokenizer payloads stay outside Git",
    }
    if args.evidence is not None:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(
            json.dumps(evidence_document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    print(json.dumps(evidence_document, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"prepare_hf_fixture: {error}", file=sys.stderr)
        raise SystemExit(2)
