# Installable CMake package

## 问题

旧仓库能`cmake --build`，也会把`.a`和头文件复制到prefix，但外部项目不知道库之间的依赖，
不能可靠执行`find_package(microLLM CONFIG REQUIRED)`。手写`-I/-L/-l`会漏掉runtime、HIP、
hipBLASLt或RCCL依赖。

## 实现

- `microLLMConfig.cmake`与版本文件，兼容同一major版本；
- 可迁移的`microLLMTargets.cmake`；
- `microLLM::runtime/core/profiling/ops/autograd/io/model/training/inference`；
- RCCL构建额外导出`microLLM::multi_gpu`；
- `microLLM_WITH_HIP/HIPBLASLT/RCCL`和可用组件metadata；
- CPU包不查找ROCm；HIP包传播HIP/hipBLASLt；多卡包传播RCCL；
- 安装LICENSE与README。

## 不是源码树内的假测试

`PackageConfig.InstalledConsumer`执行：

```text
cmake --install → 新prefix
→ 独立consumer find_package(COMPONENTS inference model)
→ 检查全部targets/metadata
→ 编译 → 静态链接 → 运行
```

三种配置实测通过：CPU-only、HIP+hipBLASLt、HIP+hipBLASLt+RCCL。RCCL验证只证明package
target可配置/链接，不把它写成双卡运行证据；双卡数值门仍由独立RCCL CTest负责。

## 外部用法

```cmake
find_package(microLLM 0.1 CONFIG REQUIRED COMPONENTS inference model)
target_link_libraries(app PRIVATE microLLM::inference)
```

完整命令和target表见根[README](../../README.md#install-and-use-from-another-cmake-project)与
[构建文档](../dev/build.md#installable-cmake-package)。
