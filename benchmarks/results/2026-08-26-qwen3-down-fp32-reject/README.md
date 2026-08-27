# Qwen3 down-FP32 candidate rejection

候选让FFN down保持FP32，gate/up与全部Attention为BF16。它通过原五case oracle，并在正式短
decode中以`0.9545×`刚过0.95性能门。

warmup2完整32-row矩阵64/64 worker执行成功，但cross-framework mismatch从当前8增到10，新增
T128/B1与T128/B2长轨迹分叉。对新增T128/B1 step8导出完整logits：FP32与Transformers BF16
选320，down-FP32选25，Max/RMS为0.4048/0.0850。因此候选按答案门拒绝。

同一新增状态的up-FP32控制选320。结合原五case，up-FP32现在6/6 oracle通过，只获得进入完整
shape门的资格。

目录包含：

- `matrix-summary.json` / `matrix-raw.jsonl`：完整32/64矩阵；
- `short-performance-*`：3进程2+5短decode；
- `down-new-oracle-*`：down-FP32新增失败；
- `up-new-oracle-*`：对称控制；
- `summary.json`：最终拒绝决定。
