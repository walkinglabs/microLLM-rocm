# Qwen3 fixture 的存储参数量与运行时参数量

日期：2026-08-26
状态：契约与真实 fixture 通过

## 看见的问题

Qwen3-0.6B 的配置启用了 tied embedding。Transformers 与 microLLM 在运行时只拥有
596,049,920 个独立参数，但官方 safetensors 同时保存 embedding 和 lm_head，因此完整 header
统计得到 751,632,384 个值。如果本地 manifest 只有一个 `parameter_count`：

- fixture 工具需要它等于磁盘 payload，才能发现文件缺失或重复；
- C++/PyTorch runner 又需要它等于运行时模型，才能检查真实模型没有多建参数。

让同一个数字同时回答两个问题，会让正确的 tied checkpoint 被通用 runner 拒绝。

## 最小改动

注册表继续使用 `parameter_count` 表示物理 safetensors 值数，并允许 tied 模型增加
`runtime_parameter_count`。生成的 manifest 明确写出：

```text
stored_parameter_count   文件中保存多少个值
runtime_parameter_count  去除已验证 alias 后有多少个独立参数
parameter_count          兼容现有 runner，等于 runtime_parameter_count
```

可提交 evidence 的旧 `parameter_count` 继续表示物理文件口径，同时也写出两个显式字段。
这样旧的 fixture 图和新 runner 都不会静默改变语义。

工具拒绝非正整数，也拒绝运行时参数量大于存储值数。它不会仅凭数字相信 alias；Qwen3 的
embedding/lm_head 仍必须经过 strict loader 的命名、shape、dtype 和逐字节验证。

## 证据

- 合成 fixture：默认 6 stored/6 runtime；tied 变体 6 stored/4 runtime；
- 非法 6 stored/7 runtime 在读取 payload 前明确失败；
- 真实 Qwen3：311 stored Tensor、751,632,384 stored values、596,049,920 runtime
  parameters；manifest 重新 validate 通过；
- 通用 Qwen3 T1/B1 prefill+cached preflight 两行均通过，四个 decode token 与
  Transformers 一致；
- CPU fixture 与 official-evidence 两项 focused CTest 为 2/2。

扩展 context/batch/decode-length 真机矩阵是下一独立证据节点；本记录不使用单次 preflight
吞吐宣称稳定加速。
