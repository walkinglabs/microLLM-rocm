#!/usr/bin/env python3
"""Prevent the public evidence table from drifting behind measured results."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "docs/development/STATUS.md"


def main() -> int:
    text = STATUS.read_text(encoding="utf-8")
    for token in (
        "RCCL label 53/53",
        "CPU 377/377",
        "ASan/UBSan 375/375",
        "PyTorch-enabled CPU 380/380",
        "single-GPU HIP label 196/196",
        "B2 first drifts at P×V",
        "QK 34/34 and P×V 2/2",
        "complete-logit Max/RMS worsen 1.246×/1.068×",
        "all prefill speeds ≥0.994× and RMS improves 21.6%",
        "O projection is first drift at Max 2.77e-5–3.34e-5",
        "aggregate FFN output first drifts at Max 1.43e-5–2.19e-5",
        "complete-logit Max/RMS improve 24.7%/32.6%",
        "B1 prefill is 0.944×",
        "complete-logit Max worsens 6.9% and RMS improves only 2.5%",
        "Release prefill 0.987×–1.020× passes",
        "explicit detail records gate/up/SwiGLU/down",
        "current T2048/B2/N64 is 0.8158x",
        "experiments through 288",
        "Ranked per-leaf weighted overlap",
        "whole step 0.9594×",
        "Ranked ready-bucket weighting",
        "T128 whole step 1.0661×",
        "Ranked gather-scale fusion",
        "T128 only 1.0140×",
        "ranked reducer local optimization closed",
        "transparent softmax 65.46%–73.56%",
        "eight winners Event 2.381×–8.096×",
        "complete logits Max/RMS 0.05691/0.01370 fail",
        "Event 1.298×–2.617×/wall 1.249×–2.543×",
        "133.78→176.64 tok/s =1.3207×",
        "all T2048 1.1747×–1.3688× pass",
        "1.1777×–1.3687×",
        "DataParallel tests 11/11",
        "total requirement remains unknown",
    ):
        assert token in text
    for stale in (
        "RCCL label 49/49",
        "CPU 374/374",
        "CPU 376/376",
        "ASan/UBSan 374/374",
        "ASan/UBSan 372/372",
        "single-GPU HIP label 192/192",
        "single-GPU HIP label 195/195",
        "PyTorch-enabled CPU 377/377",
        "PyTorch-enabled CPU 379/379",
        "PyTorch-enabled build 323/323",
        "experiments through 287",
        "scale-before-ready weighted overlap ordering",
        "Model-S sync smoke; weighted ready-overlap ordering",
        "environment with >87MB /dev/shm",
    ):
        assert stale not in text

    rows = [line for line in text.splitlines()
            if line.startswith("|") and not line.startswith("|---")]
    names = [line.split("|", 2)[1].strip() for line in rows[1:]]
    assert len(names) == len(set(names))
    assert len(names) >= 120
    for relative in (
        "benchmarks/results/2026-08-25-ranked-weighted-overlap/verification.json",
        "benchmarks/results/2026-08-25-ranked-bucket-weighting/verification.json",
        "benchmarks/results/2026-08-25-ranked-gather-scale/verification.json",
        "docs/optimization-log/assets/ranked-weighted-overlap-discard.svg",
        "docs/optimization-log/assets/ranked-bucket-weighting.svg",
        "docs/optimization-log/assets/ranked-gather-scale-discard.svg",
        "benchmarks/results/2026-08-26-prefill-attention-core-diagnostics/verification.json",
        "benchmarks/results/2026-08-26-prefill-attention-core-diagnostics/diagnostics.svg",
        "benchmarks/results/2026-08-26-prefill-attention-core-matrix/summary.json",
        "benchmarks/results/2026-08-26-prefill-attention-core-matrix/analysis.json",
        "benchmarks/results/2026-08-26-prefill-attention-core-matrix/verification.json",
        "benchmarks/results/2026-08-26-prefill-attention-core-matrix/attention-core.svg",
        "benchmarks/results/2026-08-26-fp32-attention-batch-invariance/summary.json",
        "benchmarks/results/2026-08-26-fp32-attention-batch-invariance/analysis.json",
        "benchmarks/results/2026-08-26-fp32-attention-batch-invariance/verification.json",
        "benchmarks/results/2026-08-26-fp32-attention-batch-invariance/attention-solutions.svg",
        "benchmarks/results/2026-08-26-fp32-prefill-attention-model-gate/summary.json",
        "benchmarks/results/2026-08-26-fp32-prefill-attention-model-gate/analysis.json",
        "benchmarks/results/2026-08-26-fp32-prefill-attention-model-gate/verification.json",
        "benchmarks/results/2026-08-26-fp32-prefill-attention-model-gate/model-gate.svg",
        "benchmarks/results/2026-08-26-fp32-prefill-attention-selective-gate/summary.json",
        "benchmarks/results/2026-08-26-fp32-prefill-attention-selective-gate/analysis.json",
        "benchmarks/results/2026-08-26-fp32-prefill-attention-selective-gate/verification.json",
        "benchmarks/results/2026-08-26-fp32-prefill-attention-selective-gate/selective-gate.svg",
        "benchmarks/results/2026-08-26-post-exact-core-block0-trace/summary.json",
        "benchmarks/results/2026-08-26-post-exact-core-block0-trace/analysis.json",
        "benchmarks/results/2026-08-26-post-exact-core-block0-trace/verification.json",
        "benchmarks/results/2026-08-26-post-exact-core-block0-trace/post-exact-core-trace.svg",
        "benchmarks/results/2026-08-26-post-exact-o-block0-trace/summary.json",
        "benchmarks/results/2026-08-26-post-exact-o-block0-trace/analysis.json",
        "benchmarks/results/2026-08-26-post-exact-o-block0-trace/verification.json",
        "benchmarks/results/2026-08-26-post-exact-o-block0-trace/post-exact-o-trace.svg",
        "benchmarks/results/2026-08-26-fp32-prefill-o-model-gate/summary.json",
        "benchmarks/results/2026-08-26-fp32-prefill-o-model-gate/analysis.json",
        "benchmarks/results/2026-08-26-fp32-prefill-o-model-gate/verification.json",
        "benchmarks/results/2026-08-26-fp32-prefill-o-model-gate/o-model-gate.svg",
        "benchmarks/results/2026-08-26-fp32-prefill-exact-stack-gate/summary.json",
        "benchmarks/results/2026-08-26-fp32-prefill-exact-stack-gate/analysis.json",
        "benchmarks/results/2026-08-26-fp32-prefill-exact-stack-gate/verification.json",
        "benchmarks/results/2026-08-26-fp32-prefill-exact-stack-gate/exact-stack-gate.svg",
    ):
        assert (ROOT / relative).is_file()
    diagnostic_root = ROOT / (
        "benchmarks/results/2026-08-26-prefill-attention-core-diagnostics")
    diagnostic = json.loads(
        (diagnostic_root / "verification.json").read_text(encoding="utf-8"))
    assert diagnostic["experiment"] == 307
    assert diagnostic["default_dispatch_changed"] is False
    assert diagnostic["focused_gates"]["hip_t256_output_exact"] is True
    assert diagnostic["full_gates"] == {
        "cpu_debug": "376/376",
        "asan_ubsan": "374/374",
        "pytorch_enabled_cpu": "379/379",
        "mi300x_gfx942_hip": "195/195",
        "rccl": "53/53",
    }
    ET.parse(diagnostic_root / "diagnostics.svg")
    attention_root = ROOT / (
        "benchmarks/results/2026-08-26-prefill-attention-core-matrix")
    attention = json.loads(
        (attention_root / "summary.json").read_text(encoding="utf-8"))
    assert attention["process_rows"] == 8
    assert attention["binary_files_retained"] == 0
    assert attention["first_causal_nonzero_stage"].endswith(".scores")
    assert attention["first_causal_nonzero_stage_by_batch"]["2"].endswith(
        ".pv_output")
    ET.parse(attention_root / "attention-core.svg")
    solution_root = ROOT / (
        "benchmarks/results/2026-08-26-fp32-attention-batch-invariance")
    solutions = json.loads(
        (solution_root / "summary.json").read_text(encoding="utf-8"))
    solution_operations = {
        row["operation"]: row for row in solutions["operations"]}
    assert solution_operations["qk"]["block_invariant_count"] == 34
    assert solution_operations["pv"]["block_invariant_count"] == 2
    assert solution_operations["qk"]["admitted_index"] == -1
    assert solution_operations["pv"]["admitted_index"] == -1
    ET.parse(solution_root / "attention-solutions.svg")
    model_gate_root = ROOT / (
        "benchmarks/results/2026-08-26-fp32-prefill-attention-model-gate")
    model_gate = json.loads(
        (model_gate_root / "summary.json").read_text(encoding="utf-8"))
    assert model_gate["candidate_core_bitwise_equal"] is True
    assert model_gate["candidate_admitted"] is False
    assert model_gate["performance_gate_passed"] is False
    ET.parse(model_gate_root / "model-gate.svg")
    selective_root = ROOT / (
        "benchmarks/results/2026-08-26-fp32-prefill-attention-selective-gate")
    selective = json.loads(
        (selective_root / "summary.json").read_text(encoding="utf-8"))
    assert selective["performance_gate_passed"] is True
    assert selective["robust_logit_max_improvement"] is False
    assert selective["candidate_admitted"] is False
    ET.parse(selective_root / "selective-gate.svg")
    post_core_root = ROOT / (
        "benchmarks/results/2026-08-26-post-exact-core-block0-trace")
    post_core = json.loads(
        (post_core_root / "summary.json").read_text(encoding="utf-8"))
    assert post_core["first_nonzero_after_cache"].endswith(
        ".attention.output")
    ET.parse(post_core_root / "post-exact-core-trace.svg")
    post_o_root = ROOT / (
        "benchmarks/results/2026-08-26-post-exact-o-block0-trace")
    post_o = json.loads(
        (post_o_root / "summary.json").read_text(encoding="utf-8"))
    assert post_o["first_nonzero_after_cache"].endswith(".ffn_output")
    ET.parse(post_o_root / "post-exact-o-trace.svg")
    o_gate_root = ROOT / (
        "benchmarks/results/2026-08-26-fp32-prefill-o-model-gate")
    o_gate = json.loads(
        (o_gate_root / "summary.json").read_text(encoding="utf-8"))
    assert o_gate["robust_logit_max_improvement"] is True
    assert o_gate["robust_logit_rms_improvement"] is True
    assert o_gate["performance_gate_passed"] is False
    assert o_gate["candidate_admitted"] is False
    ET.parse(o_gate_root / "o-model-gate.svg")
    exact_stack_root = ROOT / (
        "benchmarks/results/2026-08-26-fp32-prefill-exact-stack-gate")
    exact_stack = json.loads(
        (exact_stack_root / "summary.json").read_text(encoding="utf-8"))
    assert exact_stack["robust_logit_max_improvement"] is False
    assert exact_stack["robust_logit_rms_improvement"] is False
    assert exact_stack["performance_gate_passed"] is True
    assert exact_stack["candidate_admitted"] is False
    ET.parse(exact_stack_root / "exact-stack-gate.svg")
    print(f"status contract: pass components={len(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
