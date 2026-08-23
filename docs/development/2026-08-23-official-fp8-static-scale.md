# 2026-08-23：official FP8静态scale失败

## 生命周期

全部Linear权重事务式量化成单份E4M3-FNUZ，weight/activation scale持久驻留，FP32 Linear释放。
prepare后tiny HIP forward零H2D/D2H；重复prepare、未加载prepare和reload均拒绝。

## Worker失败与回退

v1 Deep首个worker在M8×K8960×N1536返回status6。exact-shape registry对该shape使用
FP8反量化→BF16 GEMM，并报告native/fallback/calls；v2 36/36执行成功。

## 正式结果

- FP8 resident为FP32的Qwen45.7%、Deep34.9%；
- Qwen T8/T512 FP8/BF16为0.706/0.986×；Deep为0.966/1.044×；
- 四个FP8 precision gate全部失败，max约11–18、RMS约2.1–3.5；
- Qwen T512 top token 9707→23811；
- Deep T8含1 fallback shape/112 calls，T512全部5 shape native。

## 决定

保留执行地基和失败数据，拒绝固定0.025/0.005模型策略。下一步从activation/weight分布和饱和率
设计scale，不根据最终top token反向调参。

详见[Experiment 122](../optimization-log/experiments/122-official-fp8-static-scale.md)。
