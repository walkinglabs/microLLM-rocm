# Experiment 235 — 把SiLU塞进GEMM，为什么整模还是变慢

Status: `explicit capability only; reject model route`

## 能力边界

本机hipBLASLt 1.3.0提供`HIPBLASLT_EPILOGUE_SWISH_EXT`，但没有双输入SwiGLU epilogue。
因此候选只能做：

```text
grouped gate GEMM + Swish epilogue
grouped up GEMM
→ BF16 multiply_out_
```

它能删除standalone SwiGLU中的sigmoid/SiLU，但仍需一次乘法Kernel。路由是thread-local显式
开关，默认false；plan/kernel key包含epilogue位，不会错复用旧descriptor。

## Operator capability

T1024两shape、每格3进程、64算法、完整BF16输出：

| Shape | Passing | User-arguments speedup | Max/RMS | Reinitialize |
|---|---:|---:|---:|---:|
| Qwen | 64/64 | 1.097× | 1.22e-4 / 3.50e-5 | 0.851× |
| DeepSeek | 64/64 | 1.069× | 2.44e-4 / 4.70e-5 | 0.862× |

说明epilogue有正确pointer-stable算法，也再次说明不能每轮初始化。

## Full-model gate

同一个二进制交替`swish=false/true`，当前grouped FFN/BTHD B1T1024，每格3进程、
2 warm-up + 5 measured：

| Model | Speedup | Complete-logit Max/RMS | Peak/allocation |
|---|---:|---:|---|
| Qwen | 1.00015× | 0.0973 / 0.0211 | unchanged |
| DeepSeek | 0.99114× | 0.0362 / 0.00632 | unchanged |

![Grouped Swish epilogue discard](../assets/bf16-grouped-swish-discard.svg)

候选同时失败正确性和性能门。operator里的小误差经24/28层放大，而epilogue+乘法并没有
比已经很短的SwiGLU kernel更便宜。

## 决定

- 保留`multiply_out_`和默认关闭的epilogue研究开关；
- CLI必须同时提供exact grouped algorithm才允许开启；
- 不改Auto/官方模型默认；
- BF16 FFN的局部activation路线关闭。下一步不再尝试独立SwiGLU或单一epilogue。

发布回归：CPU 344/344、ASan/UBSan 342/342、PyTorch-enabled 318/318、完整CPU/HIP
542/542（3个条件跳过）、HIP标签186/186。覆盖清单仍注册106个测试文件。

证据：

- [`operator capability`](../../../benchmarks/results/2026-08-25-bf16-grouped-swish-operator/)
- [`full-model gate`](../../../benchmarks/results/2026-08-25-bf16-grouped-swish-model-gate/)
