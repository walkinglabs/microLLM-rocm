# 不搬数据，让Attention直接读懂布局

日期：2026-08-24

旧路径每层做四次copy：

    Q、K、V转成Attention布局
    context再转回Linear布局

框架其实已经有训练使用的布局算子。新推理policy把它们接起来：

    Q/K：bias + RoPE + 布局转换一次完成
    V：保持原来的BTHD
    Attention：直接读BTHD V，直接写BTHD context

Qwen的96次copy和DeepSeek的112次copy都变成0，速度快1.115倍和1.094倍，peak还下降4MiB和
7MiB。正式logits逐位相同。

默认仍关闭，只支持已经验证的HIP、长序列、BF16、split-half+bias、无cache写入路径。其他情况
自动使用旧办法。
