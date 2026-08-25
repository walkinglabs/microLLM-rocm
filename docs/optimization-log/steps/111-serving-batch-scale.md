# Step 111 — Retained exact path at serving batch scale

Status: planned

Step 110关闭exact-finalize局部搜索。当前B2每层只有24/28个head blocks，远少于MI300X 304个CU；
batch是保持数学完全不变的并行轴。

固定当前no-flag Auto、BF16 KV、T2048、N64，Qwen/DeepSeek分别测B1/B2/B4/B8：

1. 每格至少三个fresh process，并记录实际`auto-enabled`身份；
2. 保存complete tokens、batch row一致性、KV实际/活跃/每请求bytes；
3. 报总tokens/s、相对B1扩展、`scale/batch`效率；
4. 报每token latency、每请求吞吐、peak与tokens/s per peak GiB；
5. 同shape跑固定PyTorch BF16参考；
6. OOM或batch回落必须保留；
7. 只有存在稳定batch区间，才考虑scheduler默认bucket，不从合成batch直接改服务策略。

这一步是容量/服务优化，不声称单请求Kernel更快。
