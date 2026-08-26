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

在干净的远端`main`提交`e8eeb20b`上，使用CMake 3.31.10与GCC/G++ 13.3.0重新配置并
编译CPU Release。`PackageConfig.InstalledConsumer`、`BuildTreeConsumer`、
`PublicExample`和`RejectsNonRelocatableDestination`为4/4通过，总测试时间4.78秒。

README与构建文档现在同时给出这个入口、成功含义和证据边界。
