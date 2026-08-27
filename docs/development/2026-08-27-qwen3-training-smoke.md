# Qwen3官方训练step smoke

日期：2026-08-27
状态：FP32执行/局部对齐，BF16仅执行

B1/T32、1 step、lr1e-5真实运行官方Qwen3。FP32 micro/PyTorch loss差2.38e-7，观测final norm
参数更新后差2.57e-10；峰值9.78/12.10GB。单样本吞吐不作排名。

BF16-forward+FP32 master/gradient/AdamW也跑通，但loss差0.00996；196份mirror占880,803,840字节，
micro吞吐只有PyTorch的0.5969和自身FP32的0.7900，峰值反而1.0901倍。BF16不是当前优化路径。

下一步必须增加全参数/梯度签名输出和多步轨迹。只看一个final norm参数不能证明完整backward。
