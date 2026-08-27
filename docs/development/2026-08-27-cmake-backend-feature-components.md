# CMake Config 后端能力组件与三后端验收

日期：2026-08-27
状态：已验收

## 问题

现有 SDK 已经能安装 `microLLMConfig.cmake`，外部项目也能请求 `inference`、
`core`、`capi` 等库组件。但是“我要推理库”和“我必须使用 HIP”是两个不同条件：
CPU SDK 同样包含 `inference`，所以只请求这个组件不能防止应用选错 SDK。

可以把 Config 想成 SDK 的机器可读说明卡：

```text
库组件：我要链接哪部分代码
能力组件：这套 SDK 编译时必须具备什么后端能力
```

## 实现

- 保留原有 linkable components 和 `microLLM::*` target，不改变已有下游接口；
- 新增 `hip`、`hipblaslt`、`rccl`、`rocwmma` 四个 feature components；
- 新增 `microLLM_AVAILABLE_FEATURE_COMPONENTS` 供诊断；
- 允许下游直接写：

  ```cmake
  find_package(microLLM 0.1 CONFIG REQUIRED COMPONENTS inference hip)
  ```

- CPU SDK 必须拒绝必需的 `hip`；
- HIP SDK 必须接受必需的 `hip`，并拒绝未启用的 `rccl`；
- RCCL SDK 必须接受必需的 `rccl`；
- 多次配置时去重 `CMAKE_HIP_ARCHITECTURES`，避免把
  `gfx942;gfx942;...` 写进发布的 Config 元数据。

能力组件只负责检查，不是链接 target。需要多卡代码时仍链接
`microLLM::multi_gpu`，需要推理代码时仍链接 `microLLM::inference`。

## 测试中发现并修复的问题

第一轮 HIP consumer 把 `hip` 同时放进 REQUIRED 和 OPTIONAL 列表，CMake 在读取
Config 前就拒绝配置。测试脚手架随后改为先从 optional 列表删除当前 required feature。
这次失败没有被当作 SDK 通过结果。

RCCL 的旧 build tree 还暴露了重复架构元数据。原因是某些 ROCm/CMake 组合在重复
configure 时继续追加自动探测的架构。现在配置阶段只删除重复项，真实的多架构列表仍保留。

## 实测结果

环境：CMake 3.31.10、GCC/G++ 13.3.0、ROCm gfx942；RCCL 构建启用单机多卡库。

| SDK | PackageConfig 结果 | 覆盖的后端能力契约 |
|---|---:|---|
| CPU | 4/4，通过，6.06 秒 | `hip` 必需请求失败 |
| HIP | 4/4，通过，11.29 秒 | `hip` 成功，`rccl` 必需请求失败 |
| HIP + RCCL | 4/4，通过，11.29 秒 | `rccl` 必需请求成功 |

每组 4 道门都包含：build-tree consumer、安装后搬迁的 consumer、README 公开示例、
非法 Config 安装目的地。consumer 会真正配置、编译、链接和运行 C++、窄 `core`
组件、混合 C/C++ 与纯 C 项目，并检查版本、缺失组件、公开 targets、编译/链接选项和
后端依赖；不是只检查 `.cmake` 文件存在。

## 用户入口

- README 给出 CPU/HIP/RCCL 三种 SDK 安装命令、完整 target 表和 feature component 示例；
- `docs/dev/cmake-package.zh-CN.md` 用初学者能理解的方式解释源码、build tree、
  install tree、`ROOT`、`DIR` 和 `CMAKE_PREFIX_PATH`；
- `docs/dev/build.md` 是编译器、CMake、ROCm 依赖和故障排查的完整参考。

这组测试只证明 CMake SDK 的发现、依赖恢复、链接和运行契约。它不代替算子数值、
完整模型训练、推理吞吐或多卡参数一致性测试。
