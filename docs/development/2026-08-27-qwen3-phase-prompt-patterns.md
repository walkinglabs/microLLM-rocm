# Qwen3 phase策略的四种prompt内容

日期：2026-08-27
状态：三种内容变化直接通过，常量边界保留

同一revision/权重使用constant、alternating、ascending、sensitive四个token seed，在T512/T2048、
B1/B2上跑prefill和N8。64/64 worker完成，32行中29 pass、3 precision mismatch、0 batch失败。

三行分叉全部来自constant seed，并对应已有FP32 oracle；候选分别选2955/2955/16，FP32相同。
alternating、ascending、sensitive合计24/24直接跨框架一致。两个框架的八个B2 cached case都保持
行一致，16/16 KV精确。

最大KV471,597,056字节，峰值microLLM/PyTorch为3.166/4.714GB。每shape只有一个进程，不作
速度排名。

显式策略的内容证据从单一常量扩到三种合成变化；仍不能称为自然语言prompt鲁棒。下一尺度应使用
tokenizer产生的真实短文本/结构化prompt，而不是继续手写token模式。
