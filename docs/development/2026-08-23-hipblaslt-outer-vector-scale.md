# hipBLASLt outer-vector scale contract

本机ROCm头文件明确提供：

```text
HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F
HIPBLASLT_MATMUL_DESC_A_SCALE_MODE
HIPBLASLT_MATMUL_DESC_B_SCALE_MODE
```

语义是A scale含M个FP32值、B scale含N个值，乘积`(i,j)`自动乘A的第i项和B的第j项。
旧的vector-pointer扩展已废弃，头文件要求改用scale mode属性。

microLLM以row-major `C=A×B`提交成column-major `Cᵀ=Bᵀ×Aᵀ`。因此用户左侧activation的
M个row scales在当前hipBLASLt描述中属于`B_SCALE_POINTER/B_SCALE_MODE`；右侧weight仍属于
`A_SCALE_POINTER`且保持scalar。这个映射必须由算子测试证明，不能凭名称猜测。

下一实现合同：

1. `ScaledTensor`增加显式`Scalar/OuterRow`模式；
2. `quantize_fp8_rows_dynamic`为二维输入生成M个device FP32 scales；
3. native GEMM把用户左侧模式映射到hipBLASLt B outer-vector mode；
4. unsupported shape fallback按row scale做device反量化；
5. 先用不同行幅值的手算矩阵证明row对应关系，再接FFN-only策略；
6. 运行时若当前库/shape拒绝outer-vector，必须显式记录并回退，不能静默当scalar读取。

这是当前安装版本的本地头文件能力证据，不外推到所有ROCm版本。构建信息和正式probe结果会在
实现节点中进入JSON。
