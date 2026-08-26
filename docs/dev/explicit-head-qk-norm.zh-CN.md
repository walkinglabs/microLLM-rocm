# 显式head dimension与QK-Norm：hidden width不一定等于Attention width

过去框架默认`head_dim = hidden / heads`，所以Q投影宽度总等于hidden。现代模型可能明确给出
不同head width，并在RoPE前分别归一化Q和K。

可以把残差流看成8车道道路，但Attention临时改成2个头、每头6车道，共12车道：

```text
hidden [B,T,8]
→ Q projection [8,12] → reshape [B,T,2,6] → Q-Norm
→ K/V projection [8,6] → reshape [B,T,1,6] → K-Norm / V
→ RoPE + GQA Attention → context [B,T,12]
→ O projection [12,8] → 回到残差流
```

配置中`attention_head_dimension=0`保持旧的自动推导；正数使用显式宽度。`qk_norm=true`为每层
新增共享的`q_norm.weight [head_dim]`和`k_norm.weight [head_dim]`。它们参与state_dict、strict
load、自动求导、checkpoint与Qwen-style mapping。

错误门包括负head width、奇数RoPE width、投影shape不匹配和缺失Q/K-Norm权重。CPU测试覆盖
前向、全部梯度与cache/full-prefix；HIP测试要求与CPU对齐且执行窗零payload传输；独立PyTorch
全图比较53/53通过。

这还不等于Qwen3官方兼容。官方兼容还需要解析对应`model_type`、`head_dim`、RoPE配置和实际
checkpoint，并做完整logits/token门。
