# 2026-08-26 — ROCTX与GPU Kernel统一Perfetto时间线

HIP测试在`microllm.test.finished` ROCTX范围内提交add。rocprof同时导出marker/kernel CSV；两者共享
`Correlation_Id=2`。合并器写marker/kernel `X`事件和`s/f` flow，不用时间包含猜关联。正式结果为
2 marker、2 kernel、1相关ID、6 Trace Events。

Python `perf_counter_ns`与rocprof timestamp尚未校准，所以Python spans不被错误地硬合并。

![Unified timeline](../../benchmarks/results/2026-08-26-unified-rocprof-perfetto/unified-timeline.svg)
