# 可重复的CPU coverage

## 看见的失败

在同一个`build/cpu-coverage`目录重复运行coverage时，`cmake --build --target clean`删除对象，
却保留运行时`.gcda`。新测试binary在`gtest_discover_tests`阶段读取旧checksum，于是输出大量
“existing profile data”冲突警告。最终报告仍能生成，但日志无法证明读到的全部profile都来自
本轮binary。

## 最小改动

`scripts/run_coverage.sh`现在只在固定的`build/cpu-coverage`树内，在CMake clean之后、重新编译
之前删除`.gcda`。它打印删除数量，不删除源码、其他build目录、`.gcno`或用户指定的报告目录。

## 反驳门

如果第二次运行再次出现checksum警告，或者两次summary不同，就说明清理边界不足，不能接受。
修复后连续执行三次；第三次同时启用“profile错误立即失败”：

| 项目 | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| 删除旧profile | 59 | 59 | 59 |
| CTest | 249/249 | 249/249 | 249/249 |
| checksum冲突 | 0 | 0 | 0 |
| Lines | 6,582/7,957（82.7%） | 相同 | 相同 |
| Functions | 706/779（90.6%） | 相同 | 相同 |
| Branches | 6,347/9,961（63.7%） | 相同 | 相同 |

三份JSON summary逐字节一致。完整日志和verification位于
[`benchmarks/results/2026-08-23-repeatable-coverage/`](../../benchmarks/results/2026-08-23-repeatable-coverage/)。

## 没有证明什么

- coverage比例不是正确性结论；
- CPU profile不能代表HIP Kernel覆盖；
- 当前没有设置武断的百分比失败阈值；
- 下一步仍要根据未覆盖分支选择高风险测试，而不是为数字添加无意义用例。
