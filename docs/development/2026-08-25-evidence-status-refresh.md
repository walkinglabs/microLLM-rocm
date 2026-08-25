# 2026-08-25 — evidence status refresh

## 为什么做这个节点

优化日志已经到Experiment 280，但公开`STATUS.md`仍停在Experiment 277、RCCL 49/49，并把
weighted overlap写成下一任务。这样的文档会让新贡献者重复已经完成或已经拒绝的工作。

## 修正后的权威状态

- CPU Release：370/370；
- ASan/UBSan：368/368；
- RCCL标签：53/53；
- 单卡HIP历史门：188/188，1个条件skip，未伪装成本轮重跑；
- 优化日志：Experiment 280；
- per-leaf weighted overlap：正确、0.9594x，性能拒绝；
- ready-bucket weighting：1.0661x，显式T128保留；
- gather-scale：1.0140x且慢于running best，性能拒绝；
- world4：总共享内存需求仍unknown，不再写未经证明的`>87MB`阈值。

## 新的防漂移门

`python/tests/test_status_contract.py`检查：

- 当前数字和三条weighted路线必须出现；
- 旧49/49、Experiment 277和猜测的共享内存阈值不能出现；
- 131个组件名称不能重复；
- 三份新verification和三张SVG必须存在。

测试文件覆盖审计从125变为126，并由CMake注册`Documentation.StatusContract`。

## 真实失败与处理

Sanitizer第一次只构建`microllm_tests`后直接跑完整标签，得到364/368。四个失败分别缺安装工具、
新`hf_infer`和fixture，不是sanitizer内存错误。这次失败没有被删除：它说明新加入的package/HF
CLI门需要完整构建配置。

随后执行完整`cmake --build build-final-sanitize`，再从头运行全部368项，得到368/368。CPU
Release也从重新生成后的真实370项完整通过。状态表记录的是第二次完整、依赖齐全的运行，而不是
把失败的四项排除后计算。
