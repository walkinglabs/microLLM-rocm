# Python、ROCTX与GPU统一时间线

这组结果回答一个很具体的问题：Python的`perf_counter_ns`和rocprof的时间戳能不能直接画在同一
条时间轴上？不能先假设，必须测量两只“钟”的对应关系。

每次正式运行执行8个外层Python span。调用ROCTX push和pop前后各读一次Python时钟，rocprof
同时记录range的开始与结束。程序用调用区间中点拟合一条仿射映射，再把没有ROCTX标记的内层
host delay和HIP add span放进rocprof时间轴。每个ROCTX range先包含对应的HIP launch/copy API，
API再用精确`Correlation_Id`指向Kernel；不会把marker和Kernel中偶然相同的数字当作关系。

三次独立进程全部通过：

| 指标 | 最差结果 | 验收门 |
|---|---:|---:|
| 时钟比例偏离 | 15.11 ppm | 10,000 ppm |
| 最大拟合残差 | 1.340 µs | 50 µs |
| 最宽ROCTX调用边界 | 9.342 µs | 100 µs |
| 关联HIP add | 24/24 | 24/24 |
| Python span | 72/72 | 72/72 |

![Calibration quality](calibration-quality.svg)

每个`run-*`目录保留Python JSONL、ROCTX warm-up JSONL、rocprof HIP API/marker/kernel/agent CSV、
校准JSON、统一Perfetto JSON和stdout/stderr。打开任意`run-*/unified.json`即可查看85个Trace Event：
24个Python span、9个ROCTX range、20个GPU Kernel和32个一对一flow事件。

复现一轮：

```bash
result=/tmp/microllm-python-timeline
mkdir -p "$result"
HIP_VISIBLE_DEVICES=0 \
PYTHONPATH=python \
MICROLLM_LIBRARY="$PWD/build/hip-release/bindings/capi/libmicrollm.so" \
rocprofv3 --hip-trace --marker-trace --kernel-trace --output-format csv \
  --output-file python-unified --output-directory "$result" -- \
  python3 benchmarks/single_gpu/python_profile_timeline.py capture \
    --output "$result/profile.jsonl" --iterations 8 --overwrite

PYTHONPATH=python python3 benchmarks/single_gpu/python_profile_timeline.py merge \
  --profile "$result/profile.jsonl" \
  --marker "$result/python-unified_marker_api_trace.csv" \
  --kernel "$result/python-unified_kernel_trace.csv" \
  --hip-api "$result/python-unified_hip_api_trace.csv" \
  --calibration "$result/calibration.json" \
  --output "$result/unified.json"
```

边界：这是host span与GPU时间线的相关性证据，不会把Python wall time改写成Kernel time。异步
Python span若在GPU完成前退出，仍需要HIP Event完成记录；本节点没有声称解决这一点。
