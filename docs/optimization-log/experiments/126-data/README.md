# Experiment 126 data

Qwen-only全局activation scale收尾实验：1.6/3.2乘以原四个weight scale，共1个FP32参考和
8个FP8候选。目录保存命令、三次0/0 GPU2预检、9条raw、summary、空stderr和结论门。

8个候选top token都相同，但0个通过完整151936维logits门。曲线没有字面反弹；工程决策基于
跨模型scale冲突、仍有4.33倍RMS差距以及边际收益缩小，而不是伪称证明所有实数scale都失败。
