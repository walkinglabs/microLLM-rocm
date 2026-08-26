# PyTorch ROCm native Stream双向顺序

microLLM现在可以非拥有地包装`torch.cuda.Stream.cuda_stream`。wrapper只保存设备和原生指针；销毁
wrapper不能销毁PyTorch Stream。

三次进程分别验证两个方向，每个方向提交64个预分配2048×2048 GEMM：

| 运行 | Torch工作等待于microLLM Event | microLLM工作等待于Torch Event | 最大输出误差 |
|---|---:|---:|---:|
| run-1 | 8.263 ms | 7.966 ms | 2.57e-8 |
| run-2 | 8.380 ms | 7.966 ms | 2.57e-8 |
| run-3 | 8.356 ms | 7.936 ms | 2.57e-8 |

![Native Stream interop](native-stream-interop.svg)

两个方向在Event记录后都处于pending，3/3 wrapper均报告`owning=false`，说明工作确实进入同一个
PyTorch原生Stream并按顺序完成。这个实验只共享Stream，不共享Tensor内存；零复制TensorView仍是
下一边界。

## 必须保留的profiler失败

rocprofv3注入这个PyTorch ROCm进程时，LLVM命令行选项`spirv-expand-step`被注册两次，进程收到
SIGABRT；rocprof自己的信号处理器随后挂住，外层20秒timeout最终强制终止。失败stderr和部分CSV在
`rocprof-injection-failure/`。

因此本节点只有Event顺序、输出和三进程重复证据，没有rocprof Kernel性能结论。不能拿上一节点的
纯microLLM trace替代这个混合进程trace。

普通复现：

```bash
HIP_VISIBLE_DEVICES=0 PYTHONPATH=python \
MICROLLM_LIBRARY="$PWD/build/hip-release/bindings/capi/libmicrollm.so" \
/tmp/microllm-torch-rocm-venv/bin/python \
  benchmarks/single_gpu/pytorch_native_stream_interop.py \
  --profile /tmp/native-stream/profile.jsonl \
  --report /tmp/native-stream/report.json --overwrite
```

版本：PyTorch `2.11.0+rocm7.13.0rc2`、HIP `7.13.99004`、MI300X/gfx942。
