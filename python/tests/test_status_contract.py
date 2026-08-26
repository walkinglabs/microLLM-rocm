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
        "RCCL label 55/55",
        "CPU 395/395",
        "ASan/UBSan 392/392",
        "PyTorch-enabled CPU 398/398",
        "single-GPU HIP label 201/201",
        "24/24 launch-correlated adds",
        "residual ≤1.340us",
        "0 device/Stream sync",
        "192/192-GEMM independent Stream pending",
        "3/3 bidirectional PyTorch ROCm native-Stream Event ordering",
        "144MiB exposed, 0 wrapper copy",
        "180MiB, 0 copy, all Max 0",
        "63/63 random Softmax/RMSNorm/SwiGLU rows",
        "MHA/GQA 15/15, 105/105 Attention pointers",
        "RoPE/Embedding/loss 36/36, 108/108 pointers",
        "backward 114/114, 285/285 pointers",
        "18-process Tiny/Model-S matrix",
        "Event 0.792×–0.871×",
        "6-process 20/20 exact",
        "Event 0.469×–0.973×",
        "1.277×–1.411× vs scalar",
        "16M forward 1.142×–1.570× native Torch",
        "1M F+B only 0.615×–0.761×",
        "scalar is 2.07×–2.82× readable native formula",
        "vector/scalar only 0.946×–1.039×",
        "64K/1M Event 1.164×/1.081×",
        "peak -99.42%/-99.96%",
        "manual fused 4.855×–5.271× custom Autograd",
        "3.859×–4.105× native",
        "compiled/eager 0.584×–0.610×",
        "cold 55.8–1160.3ms",
        "C++/Python 1.286×–1.475×",
        "FP32/native 1.136×–1.144×",
        "typed/native 1.048×–1.084×",
        "BF16 exact, FP16 Max2.38e-7",
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
        "explicit filter records gate/up/SwiGLU/down",
        "gate first ordered drift Max 7.63e-6–9.54e-6",
        "common FP32 gate/up solution matrix",
        "only 296100 block-exact",
        "speedups 1.040×/0.951×/0.941×/0.995×",
        "prefill 0.981×–1.005× and Max improves 12.0%",
        "RMS improves only 3.3%",
        "prefill 0.964×–1.000× and Max improves 35.5%",
        "RMS worsens 5.8%",
        "down first drifts at Max 1.05e-5–1.72e-5",
        "K8960/N1536 down solution matrix",
        "only 296100 block-exact",
        "speedups 0.506×/0.758×/0.686×/0.863×",
        "current T2048/B2/N64 is 1.1393x",
        "Kernel 820.74ms",
        "finalize 346.92ms/42.27%",
        "GEMM 272.93ms/33.25%",
        "native 128 changes reduction tree/lane stride",
        "old 128 mapping emulated 256 logical lanes",
        "T2048 Event/wall ≈1.003×",
        "0/4 performance cases pass",
        "stable 65193",
        "Event/wall 1.814×/1.519×",
        "1.00968× below 1.01",
        "1.1393× PyTorch; finalize routes closed",
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
        "RCCL label 53/53",
        "CPU 374/374",
        "CPU 376/376",
        "CPU 377/377",
        "CPU 378/378",
        "CPU 379/379",
        "CPU 380/380",
        "CPU 381/381",
        "CPU 384/384",
        "CPU 385/385",
        "CPU 387/387",
        "CPU 388/388",
        "CPU 389/389",
        "CPU 390/390",
        "CPU 391/391",
        "CPU 392/392",
        "CPU 393/393",
        "CPU 394/394",
        "ASan/UBSan 374/374",
        "ASan/UBSan 375/375",
        "ASan/UBSan 376/376",
        "ASan/UBSan 377/377",
        "ASan/UBSan 378/378",
        "ASan/UBSan 381/381",
        "ASan/UBSan 382/382",
        "ASan/UBSan 384/384",
        "ASan/UBSan 385/385",
        "ASan/UBSan 386/386",
        "ASan/UBSan 387/387",
        "ASan/UBSan 388/388",
        "ASan/UBSan 389/389",
        "ASan/UBSan 390/390",
        "ASan/UBSan 391/391",
        "ASan/UBSan 372/372",
        "single-GPU HIP label 192/192",
        "single-GPU HIP label 195/195",
        "single-GPU HIP label 196/196",
        "single-GPU HIP label 197/197",
        "PyTorch-enabled CPU 377/377",
        "PyTorch-enabled CPU 379/379",
        "PyTorch-enabled CPU 380/380",
        "PyTorch-enabled CPU 381/381",
        "PyTorch-enabled CPU 382/382",
        "PyTorch-enabled CPU 383/383",
        "PyTorch-enabled CPU 384/384",
        "PyTorch-enabled CPU 387/387",
        "PyTorch-enabled CPU 388/388",
        "PyTorch-enabled CPU 390/390",
        "PyTorch-enabled CPU 391/391",
        "PyTorch-enabled CPU 392/392",
        "PyTorch-enabled CPU 393/393",
        "PyTorch-enabled CPU 394/394",
        "PyTorch-enabled CPU 395/395",
        "PyTorch-enabled CPU 396/396",
        "PyTorch-enabled CPU 397/397",
        "PyTorch-enabled build 323/323",
        "experiments through 287",
        "single-GPU HIP label 199/199",
        "single-GPU HIP label 200/200",
        "current T2048/B2/N64 is 0.8158x",
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
        "benchmarks/results/2026-08-26-prefill-ffn-stage-trace/summary.json",
        "benchmarks/results/2026-08-26-prefill-ffn-stage-trace/analysis.json",
        "benchmarks/results/2026-08-26-prefill-ffn-stage-trace/verification.json",
        "benchmarks/results/2026-08-26-prefill-ffn-stage-trace/ffn-stage-trace.svg",
        "benchmarks/results/2026-08-26-fp32-ffn-row-invariance/summary.json",
        "benchmarks/results/2026-08-26-fp32-ffn-row-invariance/analysis.json",
        "benchmarks/results/2026-08-26-fp32-ffn-row-invariance/verification.json",
        "benchmarks/results/2026-08-26-fp32-ffn-row-invariance/ffn-row-invariance.svg",
        "benchmarks/results/2026-08-26-fp32-prefill-ffn-model-gate/summary.json",
        "benchmarks/results/2026-08-26-fp32-prefill-ffn-model-gate/analysis.json",
        "benchmarks/results/2026-08-26-fp32-prefill-ffn-model-gate/verification.json",
        "benchmarks/results/2026-08-26-fp32-prefill-ffn-model-gate/ffn-model-gate.svg",
        "benchmarks/results/2026-08-26-fp32-prefill-ffn-all-exact-gate/summary.json",
        "benchmarks/results/2026-08-26-fp32-prefill-ffn-all-exact-gate/analysis.json",
        "benchmarks/results/2026-08-26-fp32-prefill-ffn-all-exact-gate/verification.json",
        "benchmarks/results/2026-08-26-fp32-prefill-ffn-all-exact-gate/ffn-all-exact-model-gate.svg",
        "benchmarks/results/2026-08-26-post-exact-gate-up-ffn-trace/summary.json",
        "benchmarks/results/2026-08-26-post-exact-gate-up-ffn-trace/analysis.json",
        "benchmarks/results/2026-08-26-post-exact-gate-up-ffn-trace/verification.json",
        "benchmarks/results/2026-08-26-post-exact-gate-up-ffn-trace/post-exact-gate-up-trace.svg",
        "benchmarks/results/2026-08-26-fp32-ffn-down-row-invariance/summary.json",
        "benchmarks/results/2026-08-26-fp32-ffn-down-row-invariance/analysis.json",
        "benchmarks/results/2026-08-26-fp32-ffn-down-row-invariance/verification.json",
        "benchmarks/results/2026-08-26-fp32-ffn-down-row-invariance/ffn-down-row-invariance.svg",
        "benchmarks/results/2026-08-26-clean-deepseek-t2048/summary.json",
        "benchmarks/results/2026-08-26-clean-deepseek-t2048/analysis.json",
        "benchmarks/results/2026-08-26-clean-deepseek-t2048/verification.json",
        "benchmarks/results/2026-08-26-clean-deepseek-t2048-profile/summary.json",
        "benchmarks/results/2026-08-26-clean-deepseek-t2048-profile/analysis.json",
        "benchmarks/results/2026-08-26-clean-deepseek-t2048-profile/verification.json",
        "benchmarks/results/2026-08-26-clean-deepseek-t2048-profile/profile-delta.svg",
        "benchmarks/results/2026-08-26-finalize-architecture-gap-audit/analysis.json",
        "benchmarks/results/2026-08-26-finalize-architecture-gap-audit/finalize-gap.svg",
        "benchmarks/results/2026-08-26-native128-finalize/summary.json",
        "benchmarks/results/2026-08-26-native128-finalize/analysis.json",
        "benchmarks/results/2026-08-26-native128-finalize/verification.json",
        "benchmarks/results/2026-08-26-native128-finalize/native128.svg",
        "benchmarks/results/2026-08-26-bf16-grouped-gate-up-row2/summary.json",
        "benchmarks/results/2026-08-26-bf16-grouped-gate-up-row2/analysis.json",
        "benchmarks/results/2026-08-26-bf16-grouped-gate-up-row2/verification.json",
        "benchmarks/results/2026-08-26-bf16-grouped-gate-up-row2/grouped-row2.svg",
        "benchmarks/results/2026-08-26-grouped-gate-up-decode-model/summary.json",
        "benchmarks/results/2026-08-26-grouped-gate-up-decode-model/analysis.json",
        "benchmarks/results/2026-08-26-grouped-gate-up-decode-model/model-gate.svg",
        "benchmarks/results/2026-08-26-clean-deepseek-local-saturation/analysis.json",
        "benchmarks/results/2026-08-26-clean-deepseek-local-saturation/local-saturation.svg",
        "benchmarks/results/2026-08-26-multishard-streaming/analysis.json",
        "benchmarks/results/2026-08-26-multishard-streaming/verification.json",
        "benchmarks/results/2026-08-26-multishard-streaming/multishard-streaming.svg",
        "benchmarks/results/2026-08-26-indexed-streaming/analysis.json",
        "benchmarks/results/2026-08-26-indexed-streaming/verification.json",
        "benchmarks/results/2026-08-26-indexed-streaming/indexed-streaming.svg",
        "benchmarks/results/2026-08-26-safetensors-mmap/analysis.json",
        "benchmarks/results/2026-08-26-safetensors-mmap/verification.json",
        "benchmarks/results/2026-08-26-safetensors-mmap/mmap-visit.svg",
        "benchmarks/results/2026-08-26-qwen-tool-chat/analysis.json",
        "benchmarks/results/2026-08-26-qwen-tool-chat/verification.json",
        "benchmarks/results/2026-08-26-qwen-tool-chat/tool-chat.svg",
        "benchmarks/results/2026-08-26-python-profile-api/analysis.json",
        "benchmarks/results/2026-08-26-python-profile-api/verification.json",
        "benchmarks/results/2026-08-26-python-profile-api/python-profile.svg",
        "benchmarks/results/2026-08-26-python-perfetto-export/analysis.json",
        "benchmarks/results/2026-08-26-python-perfetto-export/verification.json",
        "benchmarks/results/2026-08-26-python-perfetto-export/perfetto-export.svg",
        "benchmarks/results/2026-08-26-roctx-marker-correlation/analysis.json",
        "benchmarks/results/2026-08-26-roctx-marker-correlation/verification.json",
        "benchmarks/results/2026-08-26-roctx-marker-correlation/roctx-ranges.svg",
        "benchmarks/results/2026-08-26-unified-rocprof-perfetto/analysis.json",
        "benchmarks/results/2026-08-26-unified-rocprof-perfetto/verification.json",
        "benchmarks/results/2026-08-26-unified-rocprof-perfetto/unified-timeline.svg",
        "benchmarks/results/2026-08-26-unified-rocprof-perfetto/unified_hip_api_trace.csv",
        "benchmarks/results/2026-08-26-python-roctx-gpu-perfetto/summary.json",
        "benchmarks/results/2026-08-26-python-roctx-gpu-perfetto/analysis.json",
        "benchmarks/results/2026-08-26-python-roctx-gpu-perfetto/verification.json",
        "benchmarks/results/2026-08-26-python-roctx-gpu-perfetto/calibration-quality.svg",
        "benchmarks/results/2026-08-26-python-hip-event-completion/summary.json",
        "benchmarks/results/2026-08-26-python-hip-event-completion/analysis.json",
        "benchmarks/results/2026-08-26-python-hip-event-completion/verification.json",
        "benchmarks/results/2026-08-26-python-hip-event-completion/event-completion.svg",
        "benchmarks/results/2026-08-26-python-stream-isolation/summary.json",
        "benchmarks/results/2026-08-26-python-stream-isolation/analysis.json",
        "benchmarks/results/2026-08-26-python-stream-isolation/verification.json",
        "benchmarks/results/2026-08-26-python-stream-isolation/stream-isolation.svg",
        "benchmarks/results/2026-08-26-pytorch-native-stream-interop/summary.json",
        "benchmarks/results/2026-08-26-pytorch-native-stream-interop/analysis.json",
        "benchmarks/results/2026-08-26-pytorch-native-stream-interop/verification.json",
        "benchmarks/results/2026-08-26-pytorch-native-stream-interop/native-stream-interop.svg",
        "benchmarks/results/2026-08-26-pytorch-native-stream-interop/rocprof-injection-failure/failure.json",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-tensor/summary.json",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-tensor/analysis.json",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-tensor/verification.json",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-tensor/zero-copy-tensor.svg",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-low-precision/summary.json",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-low-precision/analysis.json",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-low-precision/verification.json",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-low-precision/zero-copy-low-precision.svg",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-operator-matrix/summary.json",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-operator-matrix/analysis.json",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-operator-matrix/verification.json",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-operator-matrix/operator-matrix.svg",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-attention/summary.json",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-attention/analysis.json",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-attention/verification.json",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-attention/attention-matrix.svg",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-sequence-loss/summary.json",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-sequence-loss/analysis.json",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-sequence-loss/verification.json",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-sequence-loss/sequence-loss-matrix.svg",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-backward/summary.json",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-backward/analysis.json",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-backward/verification.json",
        "benchmarks/results/2026-08-26-pytorch-zero-copy-backward/backward-matrix.svg",
    ):
        assert (ROOT / relative).is_file()
    python_timeline_root = ROOT / (
        "benchmarks/results/2026-08-26-python-roctx-gpu-perfetto")
    python_timeline = json.loads(
        (python_timeline_root / "summary.json").read_text(encoding="utf-8"))
    assert python_timeline["status"] == "pass"
    assert python_timeline["run_count"] == 3
    assert python_timeline["iterations_per_run"] == 8
    assert python_timeline["total_correlated_adds"] == 24
    assert python_timeline["total_profile_rows"] == 72
    assert python_timeline["max_scale_error_ppm"] < 16.0
    assert python_timeline["max_abs_residual_ns"] <= 1340.0
    assert python_timeline["max_boundary_width_ns"] <= 9342
    for run in range(1, 4):
        run_root = python_timeline_root / f"run-{run}"
        calibration = json.loads(
            (run_root / "calibration.json").read_text(encoding="utf-8"))
        timeline = json.loads(
            (run_root / "unified.json").read_text(encoding="utf-8"))
        assert (run_root / "python-unified_hip_api_trace.csv").is_file()
        assert calibration["status"] == "pass"
        assert calibration["matched_spans"] == 8
        assert len(timeline["traceEvents"]) == 85
        run_summary = python_timeline["runs"][run - 1]
        assert run_summary["hip_api_events"] == 343
        assert run_summary["correlated_pairs"] == 16
        assert run_summary["correlated_adds"] == 8
    ET.parse(python_timeline_root / "calibration-quality.svg")
    event_root = ROOT / (
        "benchmarks/results/2026-08-26-python-hip-event-completion")
    event_summary = json.loads(
        (event_root / "summary.json").read_text(encoding="utf-8"))
    assert event_summary["status"] == "pass"
    assert event_summary["run_count"] == 3
    assert event_summary["all_pending_at_submit"] is True
    assert event_summary["all_observers_distinct"] is True
    assert event_summary["all_launch_kernel_correlations_exact"] is True
    assert event_summary["marker_kernel_id_equal_count"] == 0
    assert event_summary["total_device_synchronizes"] == 0
    assert event_summary["total_stream_synchronizes"] == 0
    assert event_summary["minimum_host_work_before_observation_ms"] > 2.9
    assert event_summary["maximum_device_elapsed_ms"] < 1.6
    assert event_summary["maximum_output_error"] == 0.0
    for run in range(1, 4):
        assert (event_root / f"run-{run}/event_hip_api_trace.csv").is_file()
        assert (event_root / f"run-{run}/event_marker_api_trace.csv").is_file()
        assert (event_root / f"run-{run}/event_kernel_trace.csv").is_file()
        assert (event_root / f"run-{run}/report.json").is_file()
        profile_rows = [json.loads(line) for line in
                        (event_root / f"run-{run}/profile.jsonl").read_text(
                            encoding="utf-8").splitlines() if line]
        assert len(profile_rows) == 1
        assert profile_rows[0]["kind"] == "hip_event_completion_span"
        assert profile_rows[0]["event_ready_at_submit"] is False
    ET.parse(event_root / "event-completion.svg")
    stream_root = ROOT / (
        "benchmarks/results/2026-08-26-python-stream-isolation")
    stream_summary = json.loads(
        (stream_root / "summary.json").read_text(encoding="utf-8"))
    assert stream_summary["status"] == "pass"
    assert stream_summary["run_count"] == 3
    assert stream_summary["all_targets_pending_at_submit"] is True
    assert stream_summary["all_busy_streams_pending_after_target_wait"] is True
    assert stream_summary["total_busy_kernels"] == 192
    assert stream_summary["total_device_synchronizes"] == 0
    assert stream_summary["total_stream_synchronizes"] == 0
    assert stream_summary["minimum_busy_wait_after_target_ms"] > 6.9
    assert stream_summary["maximum_output_error"] < 3.0e-8
    assert stream_summary["marker_kernel_id_equal_count"] == 0
    for run in range(1, 4):
        run_root = stream_root / f"run-{run}"
        assert (run_root / "stream_hip_api_trace.csv").is_file()
        assert (run_root / "stream_marker_api_trace.csv").is_file()
        assert (run_root / "stream_kernel_trace.csv").is_file()
        report = json.loads(
            (run_root / "report.json").read_text(encoding="utf-8"))
        assert report["busy_pending_after_target_wait"] is True
        assert report["synchronization_scope"] == "hip_event_explicit_stream"
    ET.parse(stream_root / "stream-isolation.svg")
    native_root = ROOT / (
        "benchmarks/results/2026-08-26-pytorch-native-stream-interop")
    native_summary = json.loads(
        (native_root / "summary.json").read_text(encoding="utf-8"))
    assert native_summary["status"] == "pass_with_profiler_boundary"
    assert native_summary["run_count"] == 3
    assert native_summary["all_torch_work_pending_for_microllm"] is True
    assert native_summary["all_microllm_work_pending_for_torch"] is True
    assert native_summary["all_wrappers_non_owning"] is True
    assert native_summary["minimum_torch_to_microllm_wait_ms"] > 8.2
    assert native_summary["minimum_microllm_to_torch_wait_ms"] > 7.9
    assert native_summary["maximum_output_error"] < 3.0e-8
    assert native_summary["rocprof_performance_claim"] is False
    assert native_summary["rocprof_injection"] == \
        "failed_duplicate_llvm_command_line_option"
    failure = json.loads(
        (native_root / "rocprof-injection-failure/failure.json").read_text(
            encoding="utf-8"))
    assert failure["forced_termination"] is True
    for run in range(1, 4):
        report = json.loads(
            (native_root / f"run-{run}/report.json").read_text(encoding="utf-8"))
        assert report["wrapper_owning"] is False
        assert report["torch_pending_for_microllm_event"] is True
        assert report["microllm_pending_for_torch_event"] is True
    ET.parse(native_root / "native-stream-interop.svg")
    zero_copy_root = ROOT / (
        "benchmarks/results/2026-08-26-pytorch-zero-copy-tensor")
    zero_copy = json.loads(
        (zero_copy_root / "summary.json").read_text(encoding="utf-8"))
    assert zero_copy["status"] == "pass_with_profiler_boundary"
    assert zero_copy["run_count"] == 3
    assert zero_copy["all_gates_passed"] is True
    assert zero_copy["total_wrapped_payload_bytes"] == 150994944
    assert zero_copy["total_wrapper_copy_bytes"] == 0
    assert zero_copy["submitted_zero_copy_adds"] == 384
    assert zero_copy["maximum_output_error"] == 0.0
    assert zero_copy["rocprof_performance_claim"] is False
    for run in range(1, 4):
        report = json.loads(
            (zero_copy_root / f"run-{run}/report.json").read_text(
                encoding="utf-8"))
        assert report["pointers_match"] is True
        assert report["wrappers_non_owning"] is True
        assert report["owner_retained_by_wrapper"] is True
        assert report["owner_released_after_close"] is True
        assert report["noncontiguous_rejected"] is True
        assert report["short_storage_rejected"] is True
    ET.parse(zero_copy_root / "zero-copy-tensor.svg")
    low_root = ROOT / (
        "benchmarks/results/2026-08-26-pytorch-zero-copy-low-precision")
    low = json.loads((low_root / "summary.json").read_text(encoding="utf-8"))
    assert low["status"] == "pass_with_profiler_boundary"
    assert low["run_count"] == 3
    assert low["dtype_cases"] == 6
    assert low["all_pointer_gates_passed"] is True
    assert low["all_wrappers_non_owning"] is True
    assert low["pending_event_gates"] == 12
    assert low["maximum_multiply_error"] == 0.0
    assert low["maximum_matmul_error"] == 0.0
    assert low["total_wrapped_payload_bytes"] == 188743680
    assert low["total_wrapper_copy_bytes"] == 0
    assert low["submitted_zero_copy_ops"] == 768
    assert low["rocprof_performance_claim"] is False
    for run in range(1, 4):
        report = json.loads(
            (low_root / f"run-{run}/report.json").read_text(encoding="utf-8"))
        assert {case["dtype"] for case in report["cases"]} == {"fp16", "bf16"}
        assert all(case["pointer_matches"] for case in report["cases"])
        assert all(case["wrappers_non_owning"] for case in report["cases"])
    ET.parse(low_root / "zero-copy-low-precision.svg")
    operator_root = ROOT / (
        "benchmarks/results/2026-08-26-pytorch-zero-copy-operator-matrix")
    operator_matrix = json.loads(
        (operator_root / "summary.json").read_text(encoding="utf-8"))
    assert operator_matrix["status"] == "pass_with_profiler_boundary"
    assert operator_matrix["run_count"] == 3
    assert operator_matrix["record_count"] == 63
    assert operator_matrix["seeds"] == [20260826, 20260827, 20260828]
    assert operator_matrix["all_pointer_gates_passed"] is True
    assert operator_matrix["all_wrappers_non_owning"] is True
    assert operator_matrix["total_wrapper_copy_bytes"] == 0
    assert operator_matrix["maximum_error"] == 0.0625
    assert operator_matrix["maximum_rms_error"] < 0.002
    groups = {
        (row["operation"], row["dtype"]): row
        for row in operator_matrix["groups"]}
    assert groups[("softmax", "fp32")]["rows"] == 12
    assert groups[("rms_norm", "fp32")]["rows"] == 12
    assert groups[("rms_norm_output", "bf16")]["maximum_error"] == 0.0
    assert groups[("swiglu", "bf16")]["maximum_error"] == 0.0625
    assert groups[("swiglu", "bf16")]["maximum_tolerance_fraction"] < 0.9
    ET.parse(operator_root / "operator-matrix.svg")
    attention_root = ROOT / (
        "benchmarks/results/2026-08-26-pytorch-zero-copy-attention")
    attention = json.loads(
        (attention_root / "summary.json").read_text(encoding="utf-8"))
    assert attention["status"] == "pass_with_profiler_boundary"
    assert attention["run_count"] == 3
    assert attention["record_count"] == 15
    assert attention["shape_count"] == 5
    assert attention["mha_rows"] == 3
    assert attention["gqa_rows"] == 12
    assert attention["pending_event_rows"] == 15
    assert attention["pointer_matches"] == 105
    assert attention["non_owning_wrappers"] == 105
    assert attention["maximum_output_error"] < 8.35e-7
    assert attention["maximum_output_rms_error"] < 6.8e-8
    assert attention["maximum_workspace_error"] < 3.0e-8
    assert attention["total_wrapper_copy_bytes"] == 0
    assert attention["rocprof_performance_claim"] is False
    ET.parse(attention_root / "attention-matrix.svg")
    sequence_root = ROOT / (
        "benchmarks/results/2026-08-26-pytorch-zero-copy-sequence-loss")
    sequence = json.loads(
        (sequence_root / "summary.json").read_text(encoding="utf-8"))
    assert sequence["status"] == "pass_with_profiler_boundary"
    assert sequence["run_count"] == 3
    assert sequence["record_count"] == 36
    assert sequence["pointer_matches"] == 108
    assert sequence["non_owning_wrappers"] == 108
    assert sequence["maximum_error"] < 1.0e-6
    assert sequence["maximum_rms_error"] < 1.0e-6
    assert sequence["total_wrapper_copy_bytes"] == 0
    groups = {row["operation"]: row for row in sequence["groups"]}
    assert groups["rope"]["rows"] == 12
    assert groups["embedding"]["maximum_error"] == 0.0
    assert groups["cross_entropy"]["rows"] == 12
    assert groups["cross_entropy"]["maximum_error"] < 1.0e-6
    ET.parse(sequence_root / "sequence-loss-matrix.svg")
    backward_root = ROOT / (
        "benchmarks/results/2026-08-26-pytorch-zero-copy-backward")
    backward = json.loads(
        (backward_root / "summary.json").read_text(encoding="utf-8"))
    assert backward["status"] == "pass_with_profiler_boundary"
    assert backward["run_count"] == 3
    assert backward["record_count"] == 114
    assert backward["gradient_groups"] == 10
    assert backward["pointer_matches"] == 285
    assert backward["non_owning_wrappers"] == 285
    assert backward["maximum_error"] < 8.6e-6
    assert backward["maximum_rms_error"] < 1.5e-6
    assert backward["total_wrapper_copy_bytes"] == 0
    assert backward["rocprof_performance_claim"] is False
    groups = {
        (row["operation"], row["target"]): row
        for row in backward["groups"]}
    assert groups[("embedding_backward", "weight")]["maximum_error"] == 0.0
    assert groups[("rms_norm_backward", "weight")]["maximum_error"] < 8.6e-6
    assert groups[("cross_entropy_backward", "factor")]["maximum_error"] == 0.0
    ET.parse(backward_root / "backward-matrix.svg")
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
    ffn_root = ROOT / (
        "benchmarks/results/2026-08-26-prefill-ffn-stage-trace")
    ffn = json.loads((ffn_root / "summary.json").read_text(encoding="utf-8"))
    assert ffn["process_rows"] == 8
    assert ffn["stage_count"] == 7
    assert ffn["first_nonzero_stage"].endswith(".ffn.gate")
    assert ffn["binary_files_retained"] == 0
    ET.parse(ffn_root / "ffn-stage-trace.svg")
    ffn_solution_root = ROOT / (
        "benchmarks/results/2026-08-26-fp32-ffn-row-invariance")
    ffn_solutions = json.loads(
        (ffn_solution_root / "summary.json").read_text(encoding="utf-8"))
    assert ffn_solutions["block_invariant_indices"] == [296100]
    assert ffn_solutions["performance_admitted_count"] == 0
    assert ffn_solutions["recommended_index"] == -1
    ET.parse(ffn_solution_root / "ffn-row-invariance.svg")
    ffn_model_root = ROOT / (
        "benchmarks/results/2026-08-26-fp32-prefill-ffn-model-gate")
    ffn_model = json.loads(
        (ffn_model_root / "summary.json").read_text(encoding="utf-8"))
    assert ffn_model["robust_logit_max_improvement"] is True
    assert ffn_model["robust_logit_rms_improvement"] is False
    assert ffn_model["performance_gate_passed"] is True
    assert ffn_model["candidate_admitted"] is False
    ET.parse(ffn_model_root / "ffn-model-gate.svg")
    ffn_all_root = ROOT / (
        "benchmarks/results/2026-08-26-fp32-prefill-ffn-all-exact-gate")
    ffn_all = json.loads(
        (ffn_all_root / "summary.json").read_text(encoding="utf-8"))
    assert ffn_all["robust_logit_max_improvement"] is True
    assert ffn_all["robust_logit_rms_improvement"] is False
    assert ffn_all["performance_gate_passed"] is True
    assert ffn_all["candidate_admitted"] is False
    ET.parse(ffn_all_root / "ffn-all-exact-model-gate.svg")
    post_gate_up_root = ROOT / (
        "benchmarks/results/2026-08-26-post-exact-gate-up-ffn-trace")
    post_gate_up = json.loads(
        (post_gate_up_root / "summary.json").read_text(encoding="utf-8"))
    assert post_gate_up["first_nonzero_stage"].endswith(".ffn.down")
    assert post_gate_up["process_rows"] == 8
    assert post_gate_up["binary_files_retained"] == 0
    ET.parse(post_gate_up_root / "post-exact-gate-up-trace.svg")
    down_root = ROOT / (
        "benchmarks/results/2026-08-26-fp32-ffn-down-row-invariance")
    down = json.loads((down_root / "summary.json").read_text(encoding="utf-8"))
    assert down["block_invariant_indices"] == [296100]
    assert down["performance_admitted_count"] == 0
    assert down["recommended_index"] == -1
    ET.parse(down_root / "ffn-down-row-invariance.svg")
    clean_root = ROOT / (
        "benchmarks/results/2026-08-26-clean-deepseek-t2048")
    clean = json.loads((clean_root / "summary.json").read_text(encoding="utf-8"))
    assert clean["rows"][0]["cross_framework_tokens_equal"] is True
    assert clean["rows"][0]["throughput_ratio_microllm_over_pytorch"] > 1.13
    clean_profile_root = ROOT / (
        "benchmarks/results/2026-08-26-clean-deepseek-t2048-profile")
    clean_profile = json.loads(
        (clean_profile_root / "summary.json").read_text(encoding="utf-8"))
    assert clean_profile["kernel_profile"]["categories"][0]["category"] == \
        "cached Attention finalize"
    assert clean_profile["kernel_profile"]["negative_call_delta_names"] == []
    ET.parse(clean_profile_root / "profile-delta.svg")
    finalize_gap_root = ROOT / (
        "benchmarks/results/2026-08-26-finalize-architecture-gap-audit")
    finalize_gap = json.loads(
        (finalize_gap_root / "analysis.json").read_text(encoding="utf-8"))
    assert finalize_gap["idle_threads_during_pv"] == 128
    assert finalize_gap["numerical_order_changes"] is True
    ET.parse(finalize_gap_root / "finalize-gap.svg")
    native_root = ROOT / "benchmarks/results/2026-08-26-native128-finalize"
    native = json.loads((native_root / "summary.json").read_text(encoding="utf-8"))
    assert native["all_accuracy_gates_passed"] is True
    assert native["t2048_performance_pass_count"] == 0
    assert native["candidate_admitted"] is False
    ET.parse(native_root / "native128.svg")
    grouped_root = ROOT / (
        "benchmarks/results/2026-08-26-bf16-grouped-gate-up-row2")
    grouped = json.loads(
        (grouped_root / "summary.json").read_text(encoding="utf-8"))
    grouped_rows = {row["model"]: row for row in grouped["comparisons"]}
    assert grouped_rows["deepseek"]["solution_indices"] == [65193]
    assert len(grouped_rows["qwen"]["solution_indices"]) > 1
    ET.parse(grouped_root / "grouped-row2.svg")
    grouped_model_root = ROOT / (
        "benchmarks/results/2026-08-26-grouped-gate-up-decode-model")
    grouped_model = json.loads(
        (grouped_model_root / "summary.json").read_text(encoding="utf-8"))
    assert grouped_model["tokens_equal"] is True
    assert grouped_model["throughput_speedup"] < 1.01
    assert grouped_model["candidate_admitted"] is False
    assert not (ROOT / "benchmarks/single_gpu/grouped_gate_up_decode_model_gate.py").exists()
    app_source = (ROOT / "apps/hf_infer.cpp").read_text(encoding="utf-8")
    assert "requires HIP BF16 FFN Arena prefill or decode" not in app_source
    assert not (ROOT / "benchmarks/single_gpu/native128_finalize_matrix.py").exists()
    native_sources = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in (
            "include/microllm/ops/ops.h", "src/ops/ops.cpp",
            "src/ops/hip/kernels.h", "src/ops/hip/basic_kernels.hip",
            "benchmarks/micro/benchmark_cached_attention_stages.cpp",
            "tests/ops/ops_test.cpp", "tests/ops/hip_ops_test.cpp",
        ))
    assert "native128" not in native_sources.lower()
    for removed in (
        "benchmarks/single_gpu/fp32_prefill_ffn_model_gate.py",
        "benchmarks/single_gpu/fp32_prefill_ffn_all_exact_gate.py",
        "benchmarks/single_gpu/audit_post_exact_gate_up_ffn.py",
    ):
        assert not (ROOT / removed).exists()
    current_sources = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in (
            "apps/hf_infer.cpp", "include/microllm/ops/context.h",
            "src/model/model.cpp", "src/ops/optimized.cpp",
        ))
    assert "PrefillFfnGateUpProjection" not in current_sources
    assert "fp32-prefill-ffn-gate-up-solution-index" not in current_sources
    print(f"status contract: pass components={len(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
