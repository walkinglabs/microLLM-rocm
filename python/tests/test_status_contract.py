#!/usr/bin/env python3
"""Prevent the public evidence table from drifting behind measured results."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "docs/development/STATUS.md"


def main() -> int:
    text = STATUS.read_text(encoding="utf-8")
    for token in (
        "RCCL label 53/53",
        "CPU 374/374",
        "ASan/UBSan 372/372",
        "PyTorch-enabled CPU 377/377",
        "single-GPU HIP label 192/192",
        "current T2048/B2/N64 is 0.8158x",
        "experiments through 285",
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
        "DataParallel tests 11/11",
        "total requirement remains unknown",
    ):
        assert token in text
    for stale in (
        "RCCL label 49/49",
        "PyTorch-enabled build 323/323",
        "experiments through 284",
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
    ):
        assert (ROOT / relative).is_file()
    print(f"status contract: pass components={len(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
