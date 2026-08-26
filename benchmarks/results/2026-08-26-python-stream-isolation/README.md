# Python显式HIP Stream隔离

这组实验回答：等待Stream A上的一个Event，会不会顺手把Stream B也排空？

每次进程先预分配两个2048×2048输出，避免`hipMalloc`改变同步关系。Stream A提交一次matmul并记录
目标Event；Stream B随后向同一个预分配输出提交64次matmul并记录自己的结束Event。Python观察线程
只等待A。验收时A已经完成，但B必须仍然pending，之后才单独等待B Event并检查完整输出。

| 运行 | A提交 | A设备Event | 等完A后B剩余等待 | B Kernel | A/B Stream ID |
|---|---:|---:|---:|---:|---|
| run-1 | 0.114 ms | 0.207 ms | 6.918 ms | 64 | 1 / 2 |
| run-2 | 0.118 ms | 0.209 ms | 7.114 ms | 64 | 1 / 2 |
| run-3 | 0.108 ms | 0.203 ms | 7.274 ms | 64 | 1 / 2 |

![Stream isolation](stream-isolation.svg)

三次目标Event在提交点都未完成，等待目标后独立Stream仍未完成，192/192个busy GEMM都有独立
Stream ID。全进程`hipDeviceSynchronize=0`、`hipStreamSynchronize=0`；清理B也只等待它自己的
Event。四个首尾输出的最大误差为`2.57e-8`。

每个`run-*`保留profile/report、HIP API/marker/kernel/agent CSV和stdout/stderr。复现：

```bash
result=/tmp/microllm-stream-isolation
mkdir -p "$result"
HIP_VISIBLE_DEVICES=0 PYTHONPATH=python \
MICROLLM_LIBRARY="$PWD/build/hip-release/bindings/capi/libmicrollm.so" \
rocprofv3 --hip-trace --marker-trace --kernel-trace --output-format csv \
  --output-file stream --output-directory "$result" -- \
  python3 benchmarks/single_gpu/python_stream_isolation.py \
    --output "$result/profile.jsonl" --report "$result/report.json" --overwrite
```

边界：这证明当前C/Python四个基础算子的显式Stream入口和Event隔离，不等于所有模型、分配器或
外部框架Stream都已绑定。runner使用caller-owned输出，结论不能被推广到每次动态分配输出的路径。
