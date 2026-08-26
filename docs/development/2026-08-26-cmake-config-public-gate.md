# CMake Config公开自检入口

日期：2026-08-26
状态：已验收

## 这次补了什么

仓库已有可搬迁的`microLLMConfig.cmake`、版本文件、导出targets和独立consumer测试，
但公开README没有一条专门用于回答“这个Config真的能用吗”的短命令。完整CPU测试能覆盖它，
却会让使用者很难看出哪一部分是在验证SDK安装和下游消费。

这次新增`package-cpu` CTest preset，把已有的四道包门收拢成一个稳定入口：

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug --parallel
ctest --preset package-cpu
```

可以把它理解成搬家验收：先造好SDK，再把整个安装目录换一个位置，最后让完全独立的小项目
只凭Config找到头文件和库。若路径被写死，搬家后就会失败。

## 覆盖边界

- build tree：不安装，直接从已经配置并编译的构建目录消费；
- relocated install tree：安装后整体搬家，再配置、链接并运行；
- C++20 umbrella target：`microLLM::microLLM`；
- narrow component：只请求并链接`microLLM::core`；
- pure C11 consumer：只启用C语言并链接`microLLM::capi`；
- public example：按照README中的`microLLM_ROOT`方式运行；
- negative gates：缺失组件、不兼容版本和prefix外Config目录必须明确失败。

这条门只证明CMake打包和消费契约成立，不证明HIP数值精度、模型训练效果或多卡一致性。

## 实测

在当前`main`上，使用CMake 3.31.10与GCC/G++ 13.3.0重新配置并编译。三套包门均通过：

| SDK | 结果 | 总时间 |
|---|---:|---:|
| CPU | 4/4 | 5.19秒 |
| HIP | 4/4 | 9.88秒 |
| HIP + RCCL | 4/4 | 9.98秒 |

每一行都包含`PackageConfig.InstalledConsumer`、`BuildTreeConsumer`、
`PublicExample`和`RejectsNonRelocatableDestination`，不是只检查生成文件是否存在。
另外重新执行README最短路径：`sdk-cpu` Release构建、安装到临时prefix、独立配置
`examples/package-consumer`、编译和运行全部通过；程序报告`backend=CPU`并输出
Model-S的15,586,176个参数。

README与构建文档现在同时给出这个入口、成功含义和证据边界。新增的
[`cmake-package.zh-CN.md`](../dev/cmake-package.zh-CN.md)用独立小项目解释源码目录、
构建目录、安装目录、`ROOT`/`DIR`/`CMAKE_PREFIX_PATH`和常见错误。
