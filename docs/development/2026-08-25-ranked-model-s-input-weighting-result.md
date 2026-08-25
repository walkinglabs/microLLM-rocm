# 2026-08-25 — ranked Model-S uneven-input result

![Ranked Model-S input weighting](../optimization-log/assets/ranked-model-s-input-weighting.svg)

Model-S `[B1T32,B2T32]` equal-only两rank共同拒绝；token-weighted一步57 Tensor/15,586,176值
rank exact，CPU Max/RMS `0.007760/3.639e-6`，loss差`3.20e-7`。

同步weighted保留。下一节点把scale移入gradient-ready hook，建立weighted overlap顺序合同。
