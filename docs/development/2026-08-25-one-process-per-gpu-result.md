# 2026-08-25 — one-process-per-GPU bootstrap结果

![One process per GPU bootstrap](../optimization-log/assets/one-process-per-gpu-bootstrap.svg)

三次fresh launch的6个rank进程完成18个rank-step。728个参数值跨rank完全一致，与CPU global
batch最大差1.19e-7。故障注入让坏rank返回1，并由launcher终止RCCL等待peer（-15）。

bootstrap保留，下一步先把逐parameter collective换成rank-local bucket同步reducer；ready-bucket
overlap尚未跨进程迁移。
