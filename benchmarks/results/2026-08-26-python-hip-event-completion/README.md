# Python异步HIP Event完成记录

普通Python计时在函数返回时停止，但HIP Kernel可能仍在运行。本节点不加入
`hipDeviceSynchronize`，而是在默认Stream上记录开始/结束Event，函数提交完成后让一个独立Python
线程只等待结束Event。主线程可以继续做host工作。

三次MI300X独立进程全部通过：

| 运行 | 提交时间 | HIP Event设备时间 | 完成被观察 | 提交时Event | 观察前host工作 |
|---|---:|---:|---:|---|---:|
| run-1 | 0.161 ms | 1.519 ms | 4.004 ms | 未完成 | 3.426 ms |
| run-2 | 0.152 ms | 1.517 ms | 3.597 ms | 未完成 | 3.065 ms |
| run-3 | 0.147 ms | 1.510 ms | 3.458 ms | 未完成 | 2.929 ms |

![HIP Event completion](event-completion.svg)

每次正式range内都有2次`hipEventRecord`、1次`hipEventQuery`和1次softmax launch。等待发生在独立
线程，三次共出现0次`hipDeviceSynchronize`和0次`hipStreamSynchronize`。launch API和softmax
Kernel的ID精确一致；marker与Kernel的ID在3/3运行中都不同，再次证明不能直接比较两者ID。

“完成被观察”是上界：GPU完成后，Python观察线程还可能等待GIL或操作系统调度。精确设备区间来自
HIP Event的1.50ms左右结果，不能拿3.3–3.8ms的host观察时间冒充Kernel时间。“观察前host工作”也
只证明主线程没有在提交点阻塞，不声称全部时间都与GPU重叠。

每个`run-*`保留profile/report JSON、HIP API/marker/kernel/agent CSV和stdout/stderr。复现：

```bash
result=/tmp/microllm-event
mkdir -p "$result"
HIP_VISIBLE_DEVICES=0 PYTHONPATH=python \
MICROLLM_LIBRARY="$PWD/build/hip-release/bindings/capi/libmicrollm.so" \
rocprofv3 --hip-trace --marker-trace --kernel-trace --output-format csv \
  --output-file event --output-directory "$result" -- \
  python3 benchmarks/single_gpu/python_event_completion.py \
    --output "$result/profile.jsonl" --report "$result/report.json" --overwrite
```

边界：公开Python Event scope当前记录C API默认Stream。C++运行时已有显式Stream/Event；Python显式
Stream绑定仍是后续接口，不应假装本节点已经覆盖。
