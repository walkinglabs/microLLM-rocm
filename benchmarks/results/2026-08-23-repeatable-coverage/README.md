# Repeatable CPU coverage evidence

`run-1.log`、`run-2.log`与`run-3.log`是修复后连续三次完整`run_coverage.sh`输出。每次都先发现并删除59个
旧`.gcda`，随后249/249 CTest通过，且没有profile checksum冲突。

三次生成的gcovr `summary.json`逐字节相同。第三次还启用了`GCOV_EXIT_AT_ERROR=1`，未来同类
错误会直接让进程失败。仓库保存一份summary与机器可读verification；HTML与
Cobertura仍留在本地build目录，不进入Git历史。

这个gate只证明CPU coverage流程可重复，并不把82.7% line coverage解释成GPU、数值或模型
正确性。HIP、PyTorch和完整logits仍由各自独立门验证。
