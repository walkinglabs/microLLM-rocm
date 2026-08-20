# Optimization program

这份文件是专项实验的执行协议，作用类似 autoresearch 的 `program.md`，但目标是
系统性能，不是模型验证集指标。

## 1. 开始前

1. 确认 `main` 工作区干净且 CI 成功；
2. 阅读本目录 README、SCHEMA、PLAN、最新 experiment；
3. 读取最新 `results.tsv`，找到 running best；
4. 确认固定 PyTorch raw baseline 仍适用于当前软件环境；
5. 为新实验分配下一个编号；
6. 从 main 建独立的 `opt/<tag>/<experiment>` 分支；
7. 复制实验模板，先写假设再修改代码。

不要在同一个实验里同时修三个热点。无法判断是哪一项导致结果变化的实验没有研究价值。

## 2. 固定基线

第一阶段参考数据：

```text
benchmarks/results/2026-08-20-mi300x-pytorch-hf-comparison/
```

固定 comparison manifest：

```text
Qwen2.5-0.5B                       2 warm-up + 5 measured
DeepSeek-R1-Distill-Qwen-1.5B     2 warm-up + 5 measured
FP32 compute
same token IDs, checkpoints and optimizer hyperparameters
```

除非开启一个明确的新 track，否则不能修改：

- 模型 revision；
- token IDs；
- dtype；
- warm-up/measured 次数；
- batch/context；
- PyTorch runner；
- 比较公式；
- correctness tolerance。

## 3. 一次实验循环

```text
LOOP:
  1. 从 trace 选择一个热点
  2. 写 hypothesis 和 falsification
  3. commit 实验前合同
  4. 实现一个最小改动
  5. 跑 focused correctness
  6. 若失败，修复实现错误；假设错误则停止
  7. 跑完整 comparison
  8. 追加 results.tsv
  9. 写 experiment 报告
 10. 重新生成 progress.svg
 11. keep: 合入 main
 12. discard/crash: 保留报告，放弃实验分支
 13. 从新的 running best 开始下一次实验
```

不使用破坏性 reset 清理实验。discard 分支可以停止使用；日志通过单独文档提交到 main。

## 4. 先跑正确性

最低顺序：

```bash
cmake --build --preset cpu-debug --parallel
ctest --preset cpu-debug

cmake --build --preset cpu-sanitize --parallel
ctest --preset cpu-sanitize

cmake --build --preset hip-release --parallel
ctest --preset hip-release
```

然后运行受影响算子的 PyTorch oracle、Qwen/DeepSeek exact token 和多步 loss/参数更新。

性能变快但数值门失败，状态必须是 `discard` 或 `invalid`。

## 5. 固定性能运行

生成 microLLM HF matrix：

```bash
python3 benchmarks/single_gpu/hf_model_matrix.py \
  --manifest /path/to/hf-models.local.json \
  --infer-binary build/hip-release/apps/microllm_hf_infer \
  --train-binary build/hip-release/apps/microllm_hf_train_step \
  --device hip --modes infer,train \
  --output /tmp/experiment-microllm.jsonl
```

使用固定 PyTorch raw baseline 计算比值；环境或 PyTorch 改变时必须重跑两边：

```bash
python3 benchmarks/single_gpu/compare_frameworks.py \
  --kind huggingface \
  --microllm /tmp/experiment-microllm.jsonl \
  --pytorch /path/to/fixed-pytorch.jsonl \
  --output /tmp/experiment-comparison.jsonl
```

## 6. Profiler 规则

以下改动必须保存 before/after rocprof：

- Kernel；
- allocator；
- Stream/Event；
- KV Cache；
- fusion；
- HIP Graph；
- hipBLASLt algorithm/plan。

至少报告：

```text
top kernel duration/calls
HIP API duration/calls
H2D/D2H/D2D calls and bytes
allocation/free calls
kernel launches
measured end-to-end tokens/s
```

只展示某个 Kernel 变快，不足以 keep。

## 7. Keep 还是 discard

优先使用 SCHEMA 中的机器门。边缘情况按以下原则：

- 同样性能但代码显著更简单：可以 keep；
- 小幅变快但引入不可解释的复杂模板：倾向 discard；
- 一个 workload 变快、另一个退化：必须定位原因，不能只看平均；
- 只降低 setup、不改善 measured：记录为 setup 优化，不能冒充 steady-state；
- 只降低显存：作为 memory track，不能冒充 throughput 改进；
- 改变 dtype：建立新的 dtype track，不连接 FP32 running-best 线。

## 8. Agent 可以做什么

可以：

- 搜索热点代码；
- 生成一个候选 Kernel；
- 写测试和 benchmark 脚手架；
- 解析 trace；
- 重新生成图表；
- 提出下一假设。

不能自行决定：

- 放宽精度；
- 删除 reference；
- 加入全局同步；
- 更换 workload 后仍沿用旧分数；
- 删除 discard/crash 记录；
- 把单 shape 结果写成普遍性能结论。

## 9. 每轮结束输出

```text
Experiment:
Hypothesis:
Changed:
Correctness:
Before score:
After score:
Per-workload ratios:
Peak memory ratios:
Profiler delta:
Decision:
What was falsified:
Next single variable:
```
