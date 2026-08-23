# Experiment 154 data

逐层leave-one-out搜索使用revision `147864a`的fresh binary。

- 56行：2个FP32 oracle、2个FP8 baseline、52个单层候选；
- Qwen 0–23与DeepSeek 0–27各且仅各运行一次；
- 每行151,936个完整last-token logits，全部finite且top一致；
- `ranked-candidates.tsv`按模型保存完整排名，不只保留赢家；
- Qwen layer9相对baseline Max/RMS为0.713×/0.666×；
- Deep layer9为1.022×/0.994×，且0/28层两项同时不差；
- throughput只记录，不进入排序或keep；
- fresh 50步build与CLI/matrix合同均保存。

这个目录只选下一轮候选，不包含T512或重复进程，因此不能证明Qwen layer9可保留。
