# Step 104 — DeepSeek T2048 cached-Attention score oracle

Status: score oracle implemented; current DeepSeek profile pending

## 为什么不能直接写新Kernel

Experiment 086在旧Release二进制上看到DeepSeek T2048 cached Attention约占decode wall的60%。
之后allocator、BTHD layout、BF16 Q/K和Norm/Arena都发生过变化；旧占比不能直接证明当前瓶颈。

现有`cached_gqa_attention`只返回最终context。一个新online/MFMA Kernel即使最终误差很小，也无法
判断误差来自Q·K dot、softmax还是P·V。因此先补只读诊断API：

```cpp
cached_gqa_attention_scores(query, key_cache, repeats, scale)
    -> scores[B, H, 1, T]
```

它返回softmax之前的逐位置scaled dot，不改变cache、不选择token、不进入模型默认路径。

## 固定正确性矩阵

- layout：query `[B,H,1,D]`，cache `[B,KV,T,D]`；
- GQA：`H = KV × repeats`；
- DeepSeek主shape：`H12 / KV2 / D128`；
- batch：B1、B2；
- context：T31/32/33、T511/512/513、T2048；
- cache dtype：FP32、BF16；query/output保持FP32；
- oracle：C++ CPU、HIP逐项、PyTorch独立实现；
- 检查：完整`B×H×T`输出、Max/RMS、finite、shape、stride、输入不变；
- 失败：bad repeats、非dense prefix、dtype/device/shape错误必须可见。

不能只比较最终context，因为softmax可能掩盖score误差。也不能只测T2048，dispatch边界必须同时
覆盖。

## 当前profile重跑合同

score oracle通过后，才从干净revision重跑当前DeepSeek T2048/B2/N64 steady decode：

- 固定官方权重、tokenizer、prompt与生成token；
- 记录2次warm-up、至少5次measured；
- 分开cache prepare、每tokendecode、端到端；
- 记录完整tokens、KV allocated/active/waste、engine current/peak；
- 使用rocprofv3记录Kernel/HIP API/copy；
- 与当前PyTorch ROCm同shape、同dtype、同resident policy比较；
- 不把旧0.868x或旧60%当成本轮结果。

只有当前trace仍证明cached Attention是最大热点，Step 105才允许提出一个online/MFMA候选。

## 已完成的score门

- CPU手算：`[1,0] / [0,1]`两个query head对3个key位置，完整6个score逐项通过；
- PyTorch：独立`query @ repeat_interleave(key).T * scale`完整对齐；
- HIP：DeepSeek `H12/KV2/D128`，FP32/BF16 cache，B1/B2，T31/32/33、
  511/512/513、2048共16条完整矩阵通过；
- 算子区间H2D/D2H为0，query/cache地址不变，输出连续FP32；
- bad repeats、scale和非dense/错误shape均显式拒绝；
- coverage manifest把新API同时绑定到PyTorch数值case和shape失败case。

完整回归：CPU 371/371、ASan/UBSan 369/369、HIP 191/191、PyTorch OperatorParity通过。
该API仍是diagnostic-only，不进入模型默认forward。下一提交从干净revision重跑当前DeepSeek
T2048 profile。
