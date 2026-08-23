# Matmul持久cache：优化结果怎样安全带到下一次运行

## 初中生版解释

进程内registry像写在白板上的答案：程序一关，答案就没了。直接把答案保存又有危险——昨天在
gfx942、某个ROCm和某个workspace下最快的算法，今天换环境后可能不支持。

持久cache因此保存“完整题目+答案”，加载时先核对环境：

```text
schema version
M/K/N、dtype、transpose、stride、mode、workspace
GPU architecture
HIP runtime / driver / hipBLASLt version
Readable或hipBLASLt实现
```

环境字段有一个不同，这条记录只计入`stale_entries`，不会注册。

## API

```cpp
microllm::ops::save_matmul_tuning_cache("matmul-cache.jsonl");

const auto report = microllm::ops::load_matmul_tuning_cache(
    "matmul-cache.jsonl", microllm::Device::hip(0));
// report.parsed_entries / loaded_entries / stale_entries
```

默认加载会在全部文件验证成功后一次性替换registry。`replace_existing=false`才做显式merge。

## 文件与失败边界

- 第一行是schema header，后续每行一个JSON entry；
- 输出字段顺序固定，保存相同registry得到相同文本；
- 先写同目录临时文件，再用rename原子替换；
- 缺字段、未知dtype/mode/implementation、重复key、错误schema、超长行或超多entry都会失败；
- 解析、去重和环境筛选全部完成后才拿registry锁，所以损坏文件不会留下半份状态；
- 当前环境没有hipBLASLt时，匹配环境却要求hipBLASLt的entry会失败。

## 测试

CPU测试覆盖确定性round-trip、原子覆盖、旧architecture计为stale、错误schema回滚、重复key回滚。
MI300测试保存真实gfx/HIP/driver/hipBLASLt key，清空后恢复选择；随后把runtime version加一，
确认同一entry变成stale且不会命中。

原始日志与机器摘要在
[`benchmarks/results/2026-08-23-matmul-persistent-cache/`](../../benchmarks/results/2026-08-23-matmul-persistent-cache/)。

## 仍然缺什么

Cache只保存已经验证的决定，不会自己产生决定。下一节点必须实现correctness-before-timing：先与
reference比数值，再热身、重复Event计时、取中位数/P95，最后做端到端回归。没有通过这些门的
候选不能进入持久cache。

最终回归：CPU 253/253、ASan/UBSan 251/251、PyTorch-enabled CPU 227/227、完整CPU/HIP
372/372（2个条件跳过），其中HIP标签115/115。
