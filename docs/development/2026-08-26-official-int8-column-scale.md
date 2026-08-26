# 官方INT8逐输出通道反驳

日期：2026-08-26
状态：算子保留，官方路线拒绝，当前精度线关闭

逐列scale将Qwen完整logits Max/RMS从15.203/3.467改善到5.061/1.286，并恢复第二token；第一
argmax仍24184→785。scale仅1.22MB、准备16.9ms、常驻0.904GB均不能推翻精度失败。保留column
原语和显式开关；官方weight-only INT8停止，未来必须是校准/混合精度/QAT新路线。

回归：CPU 420/420、ASan/UBSan 417/417；PyTorch-enabled CPU注册421项、MI300X HIP注册
211项，新增scalar/column/operator/model/CLI门通过；RCCL保持55/55。
