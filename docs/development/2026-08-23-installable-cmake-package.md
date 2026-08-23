# Installable CMake package

## 问题

旧仓库能`cmake --build`，也会把`.a`和头文件复制到prefix，但外部项目不知道库之间的依赖，
不能可靠执行`find_package(microLLM CONFIG REQUIRED)`。手写`-I/-L/-l`会漏掉runtime、HIP、
hipBLASLt或RCCL依赖。

## 实现

- `microLLMConfig.cmake`与版本文件；在1.0之前只兼容同一minor版本；
- 可迁移的`microLLMTargets.cmake`；
- `microLLM::runtime/core/profiling/ops/autograd/io/model/training/inference`；
- RCCL构建额外导出`microLLM::multi_gpu`；
- `microLLM_WITH_HIP/HIPBLASLT/RCCL`和可用组件metadata；
- CPU包不查找ROCm；HIP包传播HIP/hipBLASLt；多卡包传播RCCL；
- 安装LICENSE与README。

## 不是源码树内的假测试

`PackageConfig.InstalledConsumer`执行：

```text
cmake --install → 新prefix → 移动整个prefix
→ 独立consumer find_package(COMPONENTS inference model)
→ 检查全部targets/metadata
→ 编译 → 静态链接 → 运行
→ 再请求一个不存在的必需component，确认配置失败
→ 请求不兼容的0.2版本，确认0.1包拒绝
```

三种配置实测通过：CPU-only、HIP+hipBLASLt、HIP+hipBLASLt+RCCL。RCCL验证只证明package
target可配置/链接，不把它写成双卡运行证据；双卡数值门仍由独立RCCL CTest负责。

测试标签会按实际构建增加`hip`或`rccl`，所以`ctest --preset hip-release`与
`ctest --preset rccl-release`不会再漏掉安装包测试。安装target使用GNUInstallDirs中的
头文件目录，不把`include`写死。

## 外部用法

```cmake
find_package(microLLM 0.1 CONFIG REQUIRED COMPONENTS inference model)
target_link_libraries(app PRIVATE microLLM::inference)
```

完整命令和target表见根[README](../../README.md#install-and-use-from-another-cmake-project)与
[构建文档](../dev/build.md#installable-cmake-package)。

## 本次实测记录

环境：CMake 3.31.10、GCC/G++ 13.3.0、gfx942、HIP与RCCL来自当前ROCm开发环境。

| 构建 | 测试 | 结果 |
|---|---|---:|
| CPU Debug | 完整`ctest --preset cpu-debug` | 253/253 |
| CPU Debug | 搬迁prefix后的外部consumer与错误component | 通过 |
| HIP Release | HIP依赖传递、外部静态链接与运行 | 通过 |
| RCCL Release | RCCL依赖传递、`multi_gpu` target与外部链接 | 通过 |
| CPU Release | 自定义`CMAKE_INSTALL_INCLUDEDIR=include/microllm-sdk` | 通过 |

最后一行不是只检查文件是否复制。测试使用安装后的Config重新配置独立consumer，编译并运行，
说明导出target拿到的是GNUInstallDirs定义的真实头文件位置。
