# Installable CMake package

## 问题

旧仓库能`cmake --build`，也会把`.a`和头文件复制到prefix，但外部项目不知道库之间的依赖，
不能可靠执行`find_package(microLLM CONFIG REQUIRED)`。手写`-I/-L/-l`会漏掉runtime、HIP、
hipBLASLt或RCCL依赖。

## 实现

- `microLLMConfig.cmake`与版本文件；在1.0之前只兼容同一minor版本；
- 可迁移的`microLLMTargets.cmake`；
- `microLLM::runtime/core/profiling/ops/autograd/io/model/training/inference`；
- C ABI启用时导出版本化共享库和`microLLM::capi`，可由纯C项目消费；
- RCCL构建额外导出`microLLM::multi_gpu`；
- `microLLM_WITH_HIP/HIPBLASLT/RCCL/CAPI`和可用组件metadata；
- CPU包不查找ROCm；HIP包传播HIP/hipBLASLt；多卡包传播RCCL；
- 安装LICENSE与README。

2026-08-24补齐开发态Config：配置完成后，build目录本身也包含
`microLLMConfig.cmake`、版本文件和`microLLMTargets.cmake`。另一个工程可通过
`-DmicroLLM_DIR=<build目录>`直接使用已经编译的库，不必先安装。这个路径只用于共同开发；
需要搬迁或发布时仍使用安装prefix。

同时修复一个包边界问题：仓库内部的`-Wall`、Sanitizer和coverage编译参数不再作为公开
target属性传播。外部工程只接收C++20、头文件、库依赖和实际启用的后端定义。若消费的本来
就是插桩静态库，Config只传播该对象文件完成最终链接所必需的一个runtime链接参数，并用
`microLLM_WITH_SANITIZERS`或`microLLM_WITH_COVERAGE`明确标记。

## 不是源码树内的假测试

`PackageConfig.InstalledConsumer`执行：

```text
cmake --install → 新prefix → 移动整个prefix
→ 独立consumer find_package(COMPONENTS inference model)
→ 检查全部targets/metadata
→ 编译 → 静态链接 → 运行C++ consumer
→ 若C ABI启用，编译、动态链接并运行纯C consumer
→ 再请求一个不存在的必需component，确认配置失败
→ 请求不兼容的0.2版本，确认0.1包拒绝
```

三种配置实测通过：CPU-only、HIP+hipBLASLt、HIP+hipBLASLt+RCCL。RCCL验证只证明package
target可配置/链接，不把它写成双卡运行证据；双卡数值门仍由独立RCCL CTest负责。

测试标签会按实际构建增加`hip`或`rccl`，所以`ctest --preset hip-release`与
`ctest --preset rccl-release`不会再漏掉安装包测试。安装target使用GNUInstallDirs中的
头文件目录，不把`include`写死。

新增的`PackageConfig.BuildTreeConsumer`执行：

```text
已配置并编译的microLLM build目录
→ 独立consumer使用microLLM_DIR查找Config
→ 检查组件、target和功能metadata
→ 确认仓库内部编译参数和多余链接参数没有泄漏
→ 编译、链接并运行C++ consumer
→ 若C ABI启用，再编译、链接并运行纯C consumer
→ 拒绝不存在的组件和不兼容版本
```

2026-08-24同一源码实测结果：

| 配置 | 结果 |
|---|---:|
| CPU Debug完整回归 | 280/280 |
| ASan/UBSan完整回归 | 278/278 |
| PyTorch-enabled CPU完整回归 | 254/254 |
| CPU+HIP完整配置 | 429/429，3项按环境契约跳过 |
| gfx942 HIP标签 | 143/143 |
| RCCL Config consumer | 安装树与build tree均通过 |

Sanitizer第一次复测准确暴露了一个链接问题：静态库已插桩，但consumer没有runtime链接参数。
最终方案没有恢复全部编译参数传播，只导出必要的最终链接选项；随后两类consumer与278项完整
Sanitizer回归全部通过。

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
| CPU fresh build | 新目录编译83个目标后，搬迁prefix并运行外部consumer | 1/1通过 |

最后一行不是只检查文件是否复制。测试使用安装后的Config重新配置独立consumer，编译并运行，
说明导出target拿到的是GNUInstallDirs定义的真实头文件位置。

最新一次复核使用CMake 3.31.10和GCC/G++ 13.3.0，从空的`build-package-fresh`开始配置，
确认新加入的公开`tuning.h`、`tuning.cpp`与`adamw_tuning.cpp`也随`microLLM::ops`正确编译、
安装和静态链接。外部consumer会实际调用CPU上的matmul/AdamW tuner拒绝路径与显式
bias-gradient实现，避免只有头文件安装成功、静态库却漏掉实现符号。
在旧build目录直接运行`ctest`曾出现缺少新符号的失败；先执行`cmake --build`后消失。
这不是Config漏依赖，而是`ctest`只运行已有产物、不会重新编译。根README已明确写出
`configure -> build -> test`顺序，避免把陈旧二进制误报为package故障。

后续补齐了此前遗漏的C ABI消费面：`libmicrollm`现在具有`0.1.0`文件版本与major SONAME，
并作为`microLLM::capi`进入同一导出集合。搬迁prefix后，独立纯C项目通过该target找到头文件、
链接共享库并实际完成Tensor加法；关闭`MICROLLM_BUILD_CAPI`的fresh build则不会错误暴露
`capi`组件或target。CPU Debug完整回归通过；本次Config改动新增的是同一个
`PackageConfig.InstalledConsumer`内部的纯C编译、链接与运行门。

2026-08-24最后补上诊断metadata：外部项目可以读取`microLLM_VERSION`、
`microLLM_CXX_STANDARD`和`microLLM_HIP_ARCHITECTURES`，快速确认CMake实际找到的是哪一套
CPU/Radeon/Instinct SDK。独立consumer会检查三个字段存在且自洽；重新配置、编译后，
安装树与build tree两条外部消费路径均通过，共2/2。

2026-08-24再次收口外部发布路径：Config新增`VERSION_MAJOR/MINOR/PATCH`，组件不仅检查
声明清单，也检查对应导出target确实存在；缺失必需组件会列出当前包的可用组件。新增
`sdk-cpu` preset与`MICROLLM_BUILD_APPS`开关，使发布SDK时不必顺带编译命令行程序、测试和
benchmark。根README合并重复的Config说明，并将长实验流水默认折叠，首页先展示构建、
安装和外部消费主路径。
