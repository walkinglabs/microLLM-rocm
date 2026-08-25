# 2026-08-25 — all-reduce后原地average

新增 `scale_in_place_`：连续浮点Tensor、有限factor、Storage地址不变；CPU/HIP/PyTorch均有
数值和非法输入测试。

Communicator默认在all-reduce sum后原地乘 `1/world_size`，并保留显式allocating control用于
同二进制A/B。Model-S 3bucket smoke中：

- average Tensor 6→0；
- backend allocations 126→120；
- temporary bytes 374,068,224→249,378,816；
- step2 communication 7.26→5.15ms，total 22.47→17.88ms（跨commit单点，只作pilot）。

正式矩阵使用三个交替顺序进程，loss与末步参数必须相同后才能保留默认。

