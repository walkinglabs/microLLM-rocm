# CMake Config可用性收口：窄组件与可搬迁边界

日期：2026-08-26  
状态：已验收

## 为什么继续补

原有Config已经能安装、搬迁，并让外部工程链接完整SDK。不过，“导出了
`microLLM::core`这个名字”还不能证明它可以单独使用；此外，自定义Config安装目录若写成
绝对路径，会绕过安装prefix，使所谓可搬迁SDK名不副实。

可以把这次边界理解成两道门：

```text
外部小工程只请求core → 只链接microLLM::core → 编译并运行Tensor view

绝对Config安装目录 → 配置阶段明确拒绝 → 不产生伪装成可搬迁的SDK
```

## 改动

- Config新增`microLLM_TARGETS`，列出这一套SDK真正导入的全部target；
- 新增独立`core_consumer`，只请求`COMPONENTS core`并只链接`microLLM::core`；
- build tree和被整体移动后的install tree都配置、编译并运行该consumer；
- `MICROLLM_INSTALL_CMAKEDIR`必须是prefix内的非空相对路径；
- 新增错误路径CTest，要求绝对目录在配置阶段因正确原因失败；
- README区分`microLLM_ROOT`、`CMAKE_PREFIX_PATH`和`microLLM_DIR`，并给出定向查找日志命令。
- 公开示例门改用`microLLM_ROOT`，另外两条门继续分别覆盖`CMAKE_PREFIX_PATH`和
  `microLLM_DIR`。

## 验收命令

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug --parallel
ctest --test-dir build/cpu-debug -R '^PackageConfig\.' --output-on-failure
```

## 实测结果

环境为CMake 3.31.10、GCC/G++ 13.3.0。CPU Debug的四个包门结果为4/4：build tree、搬迁
install tree、公开示例和非法目的地拒绝全部通过。两个外部消费门中的core-only工程均完成
配置、编译、静态链接和Tensor view运行检查。

复测还发现旧开发记录把CTest正则写成了两个反斜杠。Shell单引号不会替调用者再解转义，原样
复制会匹配不到任何测试。三处命令已统一为`'^PackageConfig\.'`；这个错误不会影响CTest注册，
但会让人工验收产生错误结论。

最终回归结果：CPU Debug 381/381、ASan/UBSan 378/378、PyTorch-enabled CPU 384/384。
HIP与RCCL构建中的四个包门也分别为4/4；它们实际解析HIP、hipBLASLt和RCCL依赖并链接外部
consumer，但这组包门不替代HIP数值、训练收敛或多卡一致性测试。
