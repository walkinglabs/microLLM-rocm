# Qwen3 T128/B2 BF16权重岛

日期：2026-08-26
状态：FFN岛定位完成

审计runner现在允许FFN/Attention BF16独立组合，并记录两个flag。固定共同输入、FP32 Cache与
完整151,936 logits后：FP32和Attention-only选320，FFN-only与全部BF16 weights选25。

结果目录含7个policy/7行raw。FFN-only batch内Max 0.1443也是可见反例；Attention-only为
0.0325。测试固定policy flag、argmax、forced输入和数值边界。

本节点没有修改默认推理策略。下一步只定位FFN层；CPU/HIP框架回归沿用前一节点，新增Python
evidence门单独通过。
