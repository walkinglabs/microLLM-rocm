# Packed decode metadata：三张小纸条合成一次传输

Positions-aware每一步需要三组Int32：

```text
token IDs   [A]
positions   [A]
cache rows  [A]
```

以前分别创建三个Tensor并H2D。总字节不到1KiB，问题不是带宽，而是很多极小API调用和同步边界。

当token仍在CPU时，新路径先排成一张`[3,A]`表：

```text
row 0: token IDs
row 1: positions
row 2: cache rows
```

整张表只H2D一次，然后三个zero-copy view分别交给Embedding、RoPE、KV store和Attention。调用者若
已经传入device token，仍保留原device-input fallback，不把token拿回CPU。

MI300X continuous-only counter证明：

- R8/S4：H2D 32→16 calls；
- R8/S2：H2D 56→24 calls；
- 两边bytes都保持596；
- D2H、D2D、Cache和checksum不变。

三对交替Release A/B分别提高1.033×和1.065×，6/6逐对candidate更快。它不是大幅Kernel优化，
而是清理host/device边界的小而稳定收益。

实验见 [Experiment 100](../optimization-log/experiments/100-packed-decode-metadata.md)。
