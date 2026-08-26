# 显式head dimension与QK-Norm核心

日期：2026-08-26
状态：核心可用，官方parser待接

hidden8/head2/head_dim6强制Q/context宽12、O输入12。Q/K-Norm进入参数、mapping、前反向和cache。
PyTorch全图53/53，logits Max2.68e-7、loss精确；HIP与CPU梯度对齐且0 H2D/D2H。没有提前声称
官方Qwen3 checkpoint兼容。

回归：CPU 424/424、ASan/UBSan 421/421；PyTorch-enabled CPU 426项中新增全图alignment与三项
结构/mapping门通过；MI300X HIP新增前反向门后注册212项；RCCL保持55/55。
