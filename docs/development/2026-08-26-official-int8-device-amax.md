# 官方INT8 device-amax与精度拒绝

日期：2026-08-26
状态：device准备保留，官方整模拒绝

新增device-resident动态INT8量化和CLI显式开关。Qwen 168个Linear扫描1.431GB为18.5ms、权重
D2H为0，常驻降到0.903GB，短decode提高约9.9%。但完整logits Max/RMS 15.203/3.467，argmax与
生成token均改变。官方route拒绝；下一步只研究逐输出通道scale。

回归：CPU 418/418、ASan/UBSan 415/415；PyTorch-enabled CPU当前注册419项，MI300X HIP
注册210项，新增dynamic/operator/model/CLI门均通过；RCCL保持55/55。
