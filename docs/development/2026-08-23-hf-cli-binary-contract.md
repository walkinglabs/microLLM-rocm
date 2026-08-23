# hf_infer stale-binary contract

Exp132第一次pilot发现源码/runner接受`device-tensor-amax`，但复用的Release二进制仍只接受旧模式。
独立fresh Ninja build随后暴露真正编译错误：字符串字面量与`const char*`错误相加。旧build目录保留
旧对象，使库和CTest可以继续绿，却没有产生新CLI。

修复使用显式`std::string`拼接。新增`HfCLI.BinaryContract`直接检查目标二进制必须包含当前关键
CLI/JSON合同。它先在旧二进制上按预期失败，指出缺少device模式和device扫描字段；重建后通过。

最终还使用全新Release/HIP/gfx942/hipBLASLt Ninja目录完成34/34 build steps，避免把增量目录
状态当作fresh-build证据。Exp132第一次0-row pilot和fresh-build失败日志均作为实验失败保留。
