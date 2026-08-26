# CPU、HIP 与 RCCL 的 CMake SDK 发布路径

## 为什么还要补这一层

仓库已经会安装 `microLLMConfig.cmake`，外部工程也已经能用
`find_package(microLLM CONFIG REQUIRED)` 找到它。原来的不对称之处是：CPU 有一条只构建
SDK 的短路径，HIP 与 RCCL 用户却要先构建测试、应用和 benchmark 都启用的开发配置。

Config 文件像一张由 CMake 读取的“产品说明卡”。SDK preset 则负责把这张卡、公共头文件和
它描述的库一起生产出来。两部分都存在，别人才能用少量命令安装并消费这个框架。

## 本次改变

- 抽出隐藏的 `sdk-base`，统一关闭仓库内部工具和测试；
- 保留 `sdk-cpu`；
- 新增 `sdk-hip`，明确要求 HIP 并启用可发现的 hipBLASLt；
- 新增 `sdk-rccl`，在 HIP SDK 上增加 RCCL 多卡组件；
- Config 新增单值 `microLLM_BACKEND`，外部工程不必自己组合多个功能开关；
- README 与构建文档分别给出三条可以复制的安装命令；
- 文档明确一个 prefix 只放一种 SDK，避免旧文件残留形成混合安装。

## 用户看到的流程

```text
选择 sdk-cpu / sdk-hip / sdk-rccl
→ 编译所选后端的库
→ 安装到一个空 prefix
→ 外部项目 find_package(microLLM CONFIG REQUIRED)
→ 链接 microLLM::microLLM 或更窄的组件 target
```

HIP 和 RCCL 路径要求用户根据机器填写 GPU 架构。例如 MI300X 是 `gfx942`，但文档不把它
写成所有 AMD GPU 的默认值。

## 验收标准

- `CMakePresets.json` 能被 CMake 3.25+ 解析；
- preset 列表同时出现三种 SDK；
- `sdk-cpu` 从配置、编译、安装到外部示例运行全部通过；
- build-tree、搬迁后的 install-tree 和公开示例三条 package gate 全部通过；
- 安装树不记录原始安装 prefix，移动后仍能被找到；
- README 的 target 名称与自动化测试中的名称一致。
- 独立 consumer 检查 `microLLM_BACKEND` 与 HIP/RCCL 功能标记一致。

HIP 与 RCCL 的 Config 消费门此前已经在 `gfx942` 构建中各通过 3/3；新增的精简 preset
只改变哪些仓库附属目标参与构建，不改变核心库、后端依赖或导出 target。发布新的 GPU SDK
时仍应在实际机器上重新执行对应 package gate，不能用 CPU 结果代替 GPU 证据。

## 本次实测结果

本次不是只解析 preset 文件。三种 SDK 都从 `--fresh` 配置开始，完成编译，然后分别运行
build-tree consumer、搬迁安装树 consumer 和公开示例。环境为 CMake 3.31.10、GCC/G++
13.3.0；GPU 构建使用当前机器的 `gfx942` ROCm 工具链。

| SDK preset | 后端摘要 | Config 消费结果 | 公开示例 |
|---|---|---:|---:|
| `sdk-cpu` | CPU | 3/3 通过 | 运行通过 |
| `sdk-hip` | HIP + hipBLASLt | 3/3 通过 | 运行通过 |
| `sdk-rccl` | HIP + hipBLASLt + RCCL | 3/3 通过 | 运行通过 |

每组 consumer 都实际配置、编译、链接和运行，而不是只检查文件存在。搬迁安装树的测试还会
请求不存在的组件和不兼容的 0.2 版本，确认 Config 在配置阶段给出失败。公开示例三次都报告
CPU device 与 Model-S 的 15,586,176 个参数；它验证的是 SDK 发现和链接，不把这条输出误写成
GPU Kernel 或多卡数值测试。
