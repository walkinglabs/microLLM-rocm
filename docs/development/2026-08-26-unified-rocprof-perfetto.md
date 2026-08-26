# 2026-08-26 — ROCTX与GPU Kernel统一Perfetto时间线

HIP测试在`microllm.test.finished` ROCTX范围内提交add。修正后的正式数据同时导出244条HIP API：
range包含`hipLaunchKernel`调用，launch与add Kernel共享精确ID。合并器不再假设marker ID等于Kernel
ID，也不要求异步Kernel时间戳落在host range内。结果为2 marker、2 kernel、1个证据链、6个Trace
Event。

Python `perf_counter_ns`与rocprof timestamp尚未校准，所以Python spans不被错误地硬合并。

![Unified timeline](../../benchmarks/results/2026-08-26-unified-rocprof-perfetto/unified-timeline.svg)
