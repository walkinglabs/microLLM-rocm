# CMake SDK 与 README 收口

## 问题

仓库已经能生成`microLLMConfig.cmake`，但发布一个只供外部工程使用的CPU SDK仍要手写多组
关闭选项，而且无法关闭命令行应用。根README还把大量实验明细放在最前面，第一次访问的
用户很难快速找到构建和`find_package`入口。

## 本次边界

- 不改变现有库名、公共头文件和默认开发构建；
- 新增只面向安装的`MICROLLM_BUILD_APPS`开关和`sdk-cpu` preset；
- 补齐Config的语义版本字段与组件/target一致性检查；
- 继续支持build tree与可搬迁install tree两条消费路径；
- README保留全部实验信息，但默认折叠长流水，合并重复的CMake说明。

## 外部用户路径

```text
sdk-cpu配置
→ 编译C++库与C ABI
→ 安装到独立prefix
→ 外部工程find_package(microLLM CONFIG REQUIRED)
→ 请求组件
→ 链接microLLM::组件名
```

最小命令：

```bash
cmake --preset sdk-cpu
cmake --build --preset sdk-cpu --parallel
cmake --install build/sdk-cpu --prefix "$PWD/install/microllm"
```

## 验收

- `CMakePresets.json`可被CMake解析，且`cmake --list-presets`列出`sdk-cpu`；
- SDK preset从空目录配置和编译；
- 安装prefix被整体移动后，独立C++与C consumer仍能配置、编译、链接和运行；
- build tree consumer不经过安装也能完成相同检查；
- 不存在的必需组件与不兼容版本必须在配置阶段失败；
- 公共示例必须由独立CTest实际编译运行；
- 默认CPU回归保持通过。

实际复核环境为CMake 3.31.10与GCC/G++ 13.3.0，使用干净的提交快照，结果如下：

| 检查 | 结果 |
|---|---:|
| `sdk-cpu`从空目录配置与编译 | 通过，9个C++静态库与1个版本化C ABI共享库 |
| 独立安装prefix与公开C++ consumer | 配置、编译、链接、运行通过 |
| SDK未生成应用目录且未安装训练CLI | 通过 |
| build tree、搬迁install tree、公开示例 | 3/3通过 |
| 同一干净CPU构建的完整CTest | 281/281通过 |

这组281项来自本次关闭examples和benchmarks的独立验证配置，因此不替换根README中更完整
preset的总数；它证明这次CMake/README改动没有破坏该配置实际包含的任何CPU测试。

## 解释给第一次接触CMake的读者

普通源码构建只说明“这个仓库自己能编译”。Config package解决的是另一件事：让第二个仓库
知道头文件在哪里、应该按什么顺序链接库、需要C++哪个版本，以及这个SDK是否依赖HIP或
RCCL。它相当于随SDK一起安装的一张机器可读说明书。外部用户只需要请求能力并链接一个命名
target，不需要猜测编译器参数。
