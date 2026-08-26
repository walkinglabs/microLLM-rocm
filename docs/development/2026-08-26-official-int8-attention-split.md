# 官方INT8 Attention最终拆分

日期：2026-08-26
状态：两项拒绝，当前PTQ weight-only路线饱和

固定Max≤0.1/RMS≤0.02/token exact门下，QKV-only为0.1355/0.0293，O-only为
0.1076/0.02004；token均一致、速度均提高，但数值都失败。没有事后放宽阈值。当前官方PTQ
weight-only INT8关闭；未来必须是校准/混合bit/QAT新路线。

回归口径保持CPU 421/421、ASan/UBSan 418/418、PyTorch-enabled CPU 422/422、MI300X HIP
211/211和RCCL 55/55；scope扩展复用现有model/CLI门，不增加虚假测试计数。
