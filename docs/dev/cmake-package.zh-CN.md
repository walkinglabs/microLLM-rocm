# CMake Config 入门：把 microLLM 当成一个真正的 SDK

这篇说明回答一个很具体的问题：

> 别人的 CMake 项目，怎样不用复制 microLLM 源码，就能找到它的头文件和库？

答案是安装 microLLM 的 **CMake Config package**。它可以理解成 SDK 随身携带的
“说明卡”：卡片告诉 CMake 头文件在哪里、应该链接哪些库、需要 C++20 还是 C11，
以及这个 SDK 是否带 HIP、hipBLASLt 或 RCCL。

## 先分清三个目录

假设源码在：

```text
/work/microLLM-rocm
```

构建后会出现三个不同概念：

```text
源码目录 /work/microLLM-rocm
    │  C++/HIP 源文件在这里，不能直接交给 find_package
    ▼
构建目录 /work/microLLM-rocm/build/sdk-cpu
    │  编译结果在这里，适合两个仓库一起开发
    ▼
安装目录 /work/microLLM-rocm/install/microllm
       是可以搬走、压缩或交给另一个项目的 SDK
```

最常见的错误，就是把 `microLLM_DIR` 指向源码目录。源码还没有经过配置和编译，
里面也没有生成的 `microLLMConfig.cmake`，所以 CMake 不可能从中恢复 SDK 信息。

## 第一步：准备编译环境

CPU 构建至少需要：

- Linux；
- CMake 3.25 或更新版本；
- 支持 C++20 的编译器；
- Ninja 或 Make 这类 CMake 构建工具。

本仓库当前 CPU 证据使用 CMake 3.31.10、GCC/G++ 13.3.0。AMD GPU 构建还需要
与目标 GPU 匹配的 ROCm；多卡构建需要 RCCL。完整版本和排错表见
[从源码构建](build.md)。

先检查实际工具，不要只相信已经安装：

```bash
cmake --version
gcc --version
g++ --version
```

## 第二步：构建并安装 SDK

### CPU SDK

在仓库根目录运行：

```bash
cmake --preset sdk-cpu
cmake --build --preset sdk-cpu --parallel
cmake --install build/sdk-cpu --prefix "$PWD/install/microllm"
```

### AMD GPU SDK

下面的 `gfx942` 是 MI300 系列常见架构值，不应照抄到所有显卡。请把它换成实际
机器的架构：

```bash
cmake --preset sdk-hip -DMICROLLM_HIP_ARCHITECTURES=gfx942
cmake --build --preset sdk-hip --parallel
cmake --install build/sdk-hip --prefix "$PWD/install/microllm"
```

### HIP + RCCL 多卡 SDK

```bash
cmake --preset sdk-rccl -DMICROLLM_HIP_ARCHITECTURES=gfx942
cmake --build --preset sdk-rccl --parallel
cmake --install build/sdk-rccl --prefix "$PWD/install/microllm"
```

CPU、HIP 和 RCCL 是三套不同 SDK。不要把它们依次安装到同一个旧目录，否则旧文件
可能让结果看起来像混合了多个后端。每套 SDK 使用独立安装目录最容易检查。

安装成功后，关键文件是：

```text
install/microllm/
├── include/microllm/...
├── lib/libmicrollm_*.a
├── lib/libmicrollm.so                 可选 C ABI
└── lib/cmake/microLLM/
    ├── microLLMConfig.cmake
    ├── microLLMConfigVersion.cmake
    └── microLLMTargets.cmake
```

有些 Linux 系统使用 `lib64`，所以应用不应该自己拼出库文件路径；让
`find_package` 读取 Config 才是稳定方式。

## 第三步：在另一个 C++ 项目里使用

新建一个独立目录，放入下面的 `CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.25)
project(my_microLLM_app LANGUAGES CXX)

find_package(microLLM 0.1 CONFIG REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE microLLM::microLLM)
```

再放入 `main.cpp`：

```cpp
#include <iostream>
#include <microllm/base/device.h>
#include <microllm/model/config.h>

int main() {
    const auto device = microllm::Device::cpu();
    const auto config = microllm::model::ModelConfig::model_s();
    std::cout << (device.is_cpu() ? "cpu" : "hip")
              << " parameters=" << config.parameter_count() << '\n';
}
```

最后告诉 CMake 安装目录在哪里：

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/work/microLLM-rocm/install/microllm
cmake --build build
./build/my_app
```

仓库已经提供可以直接运行的同类项目：
[`examples/package-consumer`](../../examples/package-consumer)。

## `ROOT`、`DIR` 和 `CMAKE_PREFIX_PATH` 有什么区别

它们不是三个随便互换的路径：

| 写法 | 应该指向哪里 | 适合什么时候 |
|---|---|---|
| `microLLM_ROOT=/sdk` | SDK 安装根目录 | 只想指定 microLLM |
| `CMAKE_PREFIX_PATH=/sdk` | 一个或多个 SDK 的安装根目录 | 项目同时寻找多个包 |
| `microLLM_DIR=/sdk/lib/cmake/microLLM` | 直接包含 Config 的目录 | 精确选择一套已安装 SDK |
| `microLLM_DIR=/repo/build/sdk-cpu` | 已配置的构建目录 | 两个仓库同时开发，不准备安装 |

`microLLM_DIR` 不能指向 `/work/microLLM-rocm` 这样的源码目录。

## 只链接自己需要的部分

普通应用先使用完整入口：

```cmake
target_link_libraries(my_app PRIVATE microLLM::microLLM)
```

如果应用只需要 Tensor 基础层，可以缩小依赖：

```cmake
find_package(microLLM 0.1 CONFIG REQUIRED COMPONENTS core)
target_link_libraries(my_app PRIVATE microLLM::core)
```

常用组件有 `runtime`、`core`、`profiling`、`ops`、`autograd`、`io`、`model`、
`training` 和 `inference`。启用 C ABI 后还有 `capi`；启用 RCCL 后还有
`multi_gpu`。请求一个没有编进 SDK 的必需组件会在配置阶段明确失败，不会等到链接
阶段才出现难懂的符号错误。

纯 C 程序这样使用稳定 C ABI：

```cmake
cmake_minimum_required(VERSION 3.25)
project(my_microLLM_c_app LANGUAGES C)

find_package(microLLM 0.1 CONFIG REQUIRED COMPONENTS capi)
add_executable(my_c_app main.c)
target_link_libraries(my_c_app PRIVATE microLLM::capi)
```

## 怎样证明 Config 真的能用

不能只检查三个 `.cmake` 文件是否存在。仓库的包测试会真的建立独立项目，然后完成
配置、编译、链接和运行：

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug --parallel
ctest --preset package-cpu
```

这四道门分别验证：

1. 直接使用已编译的 build tree；
2. 安装 SDK、移动整个安装目录，再从新位置使用；
3. 按 README 公开示例构建外部项目；
4. 拒绝会破坏可搬迁性的安装位置。

测试还覆盖完整 C++ SDK、只链接 `core`、真正的 C-only 项目、缺失组件，以及不兼容
版本。当前 CPU、HIP 和 HIP+RCCL 三种构建的包门均为 4/4 通过。

这只能证明“SDK 能被正确找到和链接”，不能代替算子精度、模型训练或多卡一致性测试。

## 找不到包时怎样排查

让 CMake 只打印 microLLM 的查找过程：

```bash
cmake -S . -B build --fresh \
  -DmicroLLM_ROOT=/absolute/path/to/install/microllm \
  --debug-find-pkg=microLLM
```

最后应当看到目标安装目录里的 `microLLMConfig.cmake`。如果它找到了旧 SDK：

1. 检查 `CMAKE_PREFIX_PATH` 是否还包含旧路径；
2. 用 `microLLM_DIR` 指定准确的 Config 目录；
3. 更换 SDK 后用 `--fresh`，不要让旧 `CMakeCache.txt` 继续记住原路径。

如果 HIP 或 RCCL 版本找不到，问题通常不是 microLLM 头文件，而是消费端 CMake 无法
恢复 SDK 编译时记录的 ROCm 依赖。确认 ROCm 的 CMake package 目录可通过
`CMAKE_PREFIX_PATH` 找到，并且不要把一台机器生成的 GPU SDK 当成完全无依赖的文件包。
