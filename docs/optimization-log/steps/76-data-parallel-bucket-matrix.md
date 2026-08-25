# Step 76 — Current real bucket-count matrix

Status: complete

固定20 step和final-step参数审计，扫描4B/64B/4KiB/4MiB maximum bucket。每种三个新进程，
报告bucket count、communication/total与loss等价。只有多bucket workload稳定后，才设计
readiness或persistent bucket；synthetic overlap不能代替这个矩阵。

结果：12-bucket通信1.18–1.26ms；one-bucket 0.34–0.39ms。tiny overlap workload拒绝。
