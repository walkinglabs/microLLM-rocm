# 2026-08-26 — FFN的第一处差异在gate/up投影

8个Release进程完整比较了Block-0 FFN前两个batch行。每个B1进程临时导出270,532,608 bytes，每个
B2/B4/B8进程导出541,065,216 bytes；比较后全部删除，仓库没有保留大Tensor。

结果很清楚：FFN norm跨batch和同batch都位级一致。gate是按代码执行顺序看到的第一处差异，B2 Max
为9.54e-6，B4/B8为7.63e-6。up读取同一个exact输入，也独立出现差异。同一batch内的两条相同输入行
也在gate/up GEMM后不同，说明问题不是SwiGLU先引入的。

下一步只研究真实FP32 gate/up descriptor：M随B1/2/4/8为2048/4096/8192/16384，K=1536，N=8960。
候选必须先让跨M行与同M重复行位级一致，再允许计时；不能先选快solution再看精度。
