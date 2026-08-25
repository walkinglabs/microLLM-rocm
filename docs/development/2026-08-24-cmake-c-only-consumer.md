# CMake Config 的纯 C 消费边界

日期：2026-08-24
状态：已验收

## 为什么还要补这一项

README 给 C 用户的示例只写 `project(... LANGUAGES C)`。原来的外部消费测试却在同一个
工程里同时启用了 C 和 C++。它能证明 C 源文件可以调用 C ABI，却不能证明一个完全不启用
C++ 的工程也能加载安装包。两件事看起来相近，CMake 处理它们时并不相同。

可以把边界理解成：

```text
纯 C 工程
→ find_package(microLLM COMPONENTS capi)
→ microLLM::capi 提供头文件和共享库
→ C11 编译器链接
→ 运行一次真实 Tensor 加法
```

这个工程不应知道内部有多少 C++ 静态库，也不应因为读取 Config 文件而悄悄启用 C++。

## 本次改变

- 新增独立的 `tests/package/c_only_consumer`，其 `project()` 只声明 C；
- 配置阶段明确拒绝意外加载 C++ 编译器；
- build-tree Config 与被整体移动后的 install-tree Config 都配置、编译、链接并运行该工程；
- 程序通过公开 C ABI 创建两个 Tensor，检查加法结果为 `[4, 6]`；
- README 与构建文档明确区分“C 文件测试”和“真正的纯 C 工程测试”。

## 本机证据

| 检查 | 结果 |
|---|---:|
| CPU SDK 配置与编译 | 通过 |
| build-tree C-only configure/build/run | 通过 |
| 搬迁 install-tree C-only configure/build/run | 通过 |
| CPU Debug `PackageConfig.*` | 3/3 |
| CPU Debug 完整回归 | 336/336 |
| CPU ASan/UBSan、关闭 C ABI 的包 | 3/3，正确跳过 C-only consumer |
| HIP/gfx942 `PackageConfig.*` | 3/3 |
| RCCL `PackageConfig.*` | 3/3 |
| 测试覆盖清单审计 | 154 个算子接口、40 个图接口、99 个测试文件通过 |

这项证据只说明 CMake SDK 的消费契约成立，不表示 C ABI 已覆盖全部 C++ 模型和训练接口。
