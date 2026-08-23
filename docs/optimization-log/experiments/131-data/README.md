# Experiment 131 data

官方Qwen/DeepSeek FFN-only OuterRow矩阵。固定Attention scale0.2，FFN row minimum0.0001，
weight per-Tensor amax。保存36条正式记录、summary、三次GPU预检和3条Qwen pilot。

Qwen每worker 288次、DeepSeek336次outer-row fallback，分别等于层数×3个FFN Linear×4次
forward。native status全部为0；普通unsupported-shape registry保持0，两个口径不可混写。
