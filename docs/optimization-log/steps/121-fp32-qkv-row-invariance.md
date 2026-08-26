# Step 121 — Large-M FP32 Q/K/V row invariance

Status: planned

Experiment 303把第一处差异定位到Block 0 full-prefill的FP32 Q projection；K/V也在同一组大M GEMM中
出现较小差异。这个节点只看算子。

固定K=1536，M=2048/4096/8192/16384，两类N：

- Q：N=1536；
- K/V：N=256。

每个shape枚举当前gfx942/ROCm/hipBLASLt下32 MiB workspace内的FP32 solutions，取四个M的交集。
对共同候选构造一行FP32输入并重复2048次，再把这个2048-row块重复成2/4/8份；比较每个块的完整
输出、CPU/readable reference、workspace和Event。

候选必须四个M support、reference门通过且第0个2048-row块位级相同，才进入完整模型。solution index
只对当前版本有效，不写默认。若0个候选通过，转向可读保序Kernel或明确容差边界。
