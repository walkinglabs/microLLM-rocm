# 2026-08-26 — C++ TraceTimer可选ROCTX range

`TraceOptions.emit_roctx_ranges`默认false。开启后运行时加载新版`librocprofiler-sdk-roctx`，找不到则回退
`libroctx64`；不把ROCm绝对库路径写入可安装CMake包。TraceTimer在构造时push，`finish()`或析构时保证
pop，普通JSONL记录不变。`roctx_markers_available()`暴露能力。

pilot用rocprofv3 `--marker-trace`捕获finished/destructor两个range，各1条且边界闭合。正式raw证据在基础
设施提交后重新生成。
