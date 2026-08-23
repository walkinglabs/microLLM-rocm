# Experiment 139：17倍残差抵消把小误差放大

令`R=attention residual`，`F=FFN output`，`Y=R+F`。完整值逐元素证明FP32/FP8都满足该等式；
输出误差也由两项误差相加重构，仅余浮点舍入。

| 模型/层 | cos(R,F) | cancellation factor | 误差分子因子 | 分母收缩因子 | rel误差放大 |
|---|---:|---:|---:|---:|---:|
| Qwen21 | -0.99355 | **17.02×** | 1.452× | **8.384×** | 12.171× |
| Deep27 | -0.90100 | **4.45×** | 1.547× | **2.297×** | 3.553× |

![Residual cancellation](../assets/fp8-residual-cancellation.svg)

Qwen两项范数1748.7/1697.1，和只有202.4；Deep为6718.9/7174.8，和3122.9。Qwen误差
从FFN output 1.74%到block 21.21%，主要由参考输出范数收缩驱动，不是gate/up突然爆炸。

“存在抵消”已证明，不需再用FP32 block证明。但要回答哪个FP8子路径造成可修复误差，以及关键
block改FP32是否改善logits，仍需mixed counterfactual；抵消本身不是bug，不能删除residual。
