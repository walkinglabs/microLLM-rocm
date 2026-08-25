#!/usr/bin/env python3
import json
import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "tests/coverage_manifest.json").read_text())


def public_names(path, return_types):
    text = (ROOT / path).read_text()
    pattern = rf"(?:{'|'.join(return_types)})\s+([A-Za-z_]\w*)\s*\("
    return set(re.findall(pattern, text))


def fail(messages):
    for message in messages:
        print(f"coverage audit: {message}", file=sys.stderr)
    raise SystemExit(1)


errors = []
declared_ops = public_names(
    "include/microllm/ops/ops.h",
    ["TensorPair", "TensorTriple", "Tensor", "Bf16FfnDiagnostics",
     "Fp8DispatchStats", "Fp8DynamicQuantStats",
     "Bf16PlanCacheStats", "bool",
     "AttentionLayoutPlanCacheStats",
     "Fp32MatmulSolutionKey", "Fp32MatmulSolutionStats",
     "Bf16GroupedQkvKey", "Bf16GroupedQkvStats",
     "Bf16GroupedGateUpKey", "Bf16GroupedGateUpStats",
     "MatmulImplementation", "AdamWTuningKey",
     "AdamWTuningCacheLoadReport", "AdamWImplementation",
     "AdamWMultiTensorStats",
     "std::size_t", "void"],
)
declared_ops |= public_names(
    "include/microllm/ops/tuning.h",
    ["MatmulAutotuneReport", "AdamWAutotuneReport", "void"],
)
covered_ops = set(MANIFEST["tensor_ops"]) | set(MANIFEST["operator_infrastructure"])
if declared_ops != covered_ops:
    errors.append(
        f"ops.h mismatch missing={sorted(declared_ops-covered_ops)} "
        f"stale={sorted(covered_ops-declared_ops)}"
    )

declared_graph = public_names(
    "include/microllm/autograd/autograd.h",
    ["Value", "ValueTriple", "std::pair<Value,\\s*Value>", "GraphSnapshot", "bool", "void",
     "const\\s+Tensor&", "Tensor&"],
)
covered_graph = set(MANIFEST["graph_ops"]) | set(MANIFEST["graph_infrastructure"])
if declared_graph != covered_graph:
    errors.append(
        f"autograd.h mismatch missing={sorted(declared_graph-covered_graph)} "
        f"stale={sorted(covered_graph-declared_graph)}"
    )

declared_weight_io = public_names(
    "include/microllm/io/safetensors.h",
    ["StateDict", "std::vector<SafetensorsTensorInfo>", "void"],
)
expected_weight_io = {
    "save_safetensors",
    "load_safetensors",
    "load_safetensors_files",
    "load_safetensors_index",
    "inspect_safetensors",
    "visit_safetensors",
}
if declared_weight_io != expected_weight_io:
    errors.append(
        f"safetensors.h mismatch missing={sorted(declared_weight_io-expected_weight_io)} "
        f"stale={sorted(expected_weight_io-declared_weight_io)}"
    )
declared_weight_model = public_names(
    "include/microllm/model/model.h",
    ["io::StateDict", "LoadWeightsReport", "WeightMapping",
     "Bf16GroupedQkvPrewarmReport", "void"],
)
declared_weight_model.discard("to")
expected_weight_model = {
    "state_dict",
    "load_state_dict",
    "load_safetensors",
    "load_safetensors_files",
    "load_safetensors_index",
    "save_safetensors",
    "qwen_style_weight_mapping",
    "set_bf16_ffn_arena_enabled",
    "set_bf16_ffn_norm_fusion_enabled",
    "set_bf16_qkv_arena_enabled",
    "set_bf16_attention_norm_fusion_enabled",
    "set_attention_core_arena_enabled",
    "set_cached_attention_split_sequence",
    "set_cached_attention_materialized_scores",
    "prewarm_bf16_grouped_qkv",
}
if declared_weight_model != expected_weight_model:
    errors.append(
        f"model weight API mismatch missing={sorted(declared_weight_model-expected_weight_model)} "
        f"stale={sorted(expected_weight_model-declared_weight_model)}"
    )
for name, test_file in MANIFEST["weight_api"].items():
    if not (ROOT / test_file).is_file():
        errors.append(f"weight API {name} references missing test file {test_file}")
for name, test_file in MANIFEST["profiling_api"].items():
    if not (ROOT / test_file).is_file():
        errors.append(f"profiling API {name} references missing test file {test_file}")
for name, test_file in MANIFEST["distributed_api"].items():
    if not (ROOT / test_file).is_file():
        errors.append(f"distributed API {name} references missing test file {test_file}")

parity_text = (ROOT / "python/tests/test_operator_parity.py").read_text()
oracle_text = (ROOT / "tests/torch/operator_oracle.cpp").read_text()
for name, gates in MANIFEST["tensor_ops"].items():
    for kind in ("torch", "shape"):
        case = gates.get(kind)
        if not case:
            errors.append(f"{name} has no {kind} gate")
        elif case not in parity_text or case not in oracle_text:
            errors.append(f"{name} {kind} case {case!r} is not present in both oracle sides")
for name, case in MANIFEST["graph_ops"].items():
    if case not in parity_text or case not in oracle_text:
        errors.append(f"graph op {name} PyTorch case {case!r} is missing")
for subsystem, cases in MANIFEST["integration_oracles"].items():
    for case in cases:
        if case not in parity_text or case not in oracle_text:
            errors.append(f"integration oracle {subsystem} case {case!r} is missing")

discovered = set()
for path in ROOT.rglob("*"):
    if not path.is_file():
        continue
    relative = path.relative_to(ROOT)
    if (
        relative.parts[0].startswith("build")
        or ".git" in relative.parts
        or "__pycache__" in relative.parts
    ):
        continue
    is_native_test = re.search(r"_test\.(?:c|cc|cpp|cxx)$", path.name)
    is_python_test = path.name.startswith("test_") and path.suffix == ".py"
    if is_native_test or is_python_test:
        discovered.add(relative.as_posix())
listed = set(MANIFEST["test_files"])
if discovered != listed:
    errors.append(
        f"test file mismatch missing={sorted(discovered-listed)} stale={sorted(listed-discovered)}"
    )

cmake_text = "\n".join(path.read_text() for path in ROOT.rglob("CMakeLists.txt"))
for relative in sorted(discovered):
    if pathlib.Path(relative).name not in cmake_text:
        errors.append(f"test file is not registered by CMake/CTest: {relative}")

if errors:
    fail(errors)
print(
    "coverage audit: pass "
    f"tensor_ops={len(covered_ops)} graph_api={len(covered_graph)} test_files={len(listed)}"
)
