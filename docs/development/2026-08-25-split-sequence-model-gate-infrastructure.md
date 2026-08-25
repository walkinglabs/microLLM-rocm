# 2026-08-25：官方模型current/split成对门

## 为什么不能只看算子8倍

一次DeepSeek T2048/N64 generation会经过28层、64次decode。split每次多两个逻辑临时Tensor和一次
combine launch。算子快很多，模型也可能因为分配、布局、其他GEMM或host开销而只快一点。

因此模型门不复用旧current时间。每一对启动两个新进程，一个current、一个split，奇偶轮换先后顺序。

## 固定输出

`compare_cached_attention_split_models.py`为每个进程保存：

- 完整B×vocabulary cached logits二进制；
- 完整generated token suffix；
- decode tokens/s和每token延迟；
- engine current/peak/total、逻辑/backend allocation和cache reuse；
- KV active/capacity/bytes；
- 实际S和minimum；
- measured token与forward数量。

每对再计算完整logit Max/RMS、token是否完全相等、throughput speedup、peak/allocation差值。三对中位数
至少1.05x且任意leave-one-pair-out至少1.01x，才通过性能门。精度门和性能门分开，不能用速度抵消
logit错误。

结果目录包含`raw.jsonl`、`pairs.jsonl`、`summary.json`和`comparison.svg`。伪模型合同测试生成6条
进程记录和3条pair，验证1.5x、完整logit、token、显存/分配差值、顺序轮换和SVG。

正式运行锁定DeepSeek T2048/B2/BF16/S32/N64、warm-up 2、measured steps 5、三对新进程。当前
PyTorch 163.64 tok/s证据继续作为外部参考，但本门先隔离同一microLLM binary的current/split因果。

提交前回归为CPU 374/374、ASan/UBSan 372/372、PyTorch-enabled CPU 377/377；MI300X HIP公共
模型路由门包含在此前已通过的192/192中，本节点未增加HIP-labelled测试数量。
