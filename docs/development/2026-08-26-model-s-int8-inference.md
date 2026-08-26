# Model-S INT8整模边界

日期：2026-08-26
状态：显式整模准备保留，默认关闭

新增事务式`prepare_int8_inference_weights()`覆盖全部Linear，保留Embedding/Norm FP32。CPU/HIP
单token完整模型对齐且热路径零payload传输。三进程Model-S context1为1.719× FP32，context4因
prefill显式反量化仅0.472×；常驻-59.8%，准备峰值+19.5%。token guard均相同。下一步进入官方
权重前，必须保留这条M>1反例。

回归：CPU 417/417、ASan/UBSan 414/414；PyTorch-enabled CPU注册418项并通过新增模型门，
MI300X HIP注册209项并通过新增整模门；RCCL保持紧邻节点55/55。
