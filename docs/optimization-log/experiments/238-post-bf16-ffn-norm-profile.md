# Experiment 238 — 默认路径变了，重新看时间花在哪里

Status: `profile complete; select Attention input boundary`

## 测量方法

Qwen/DeepSeek各跑load+1和load+6两个rocprof进程，用`(six-one)/5`排除权重加载、
plan初始化和首次准备。runner还要求每个应用记录明确报告
`bf16_ffn_norm_fusion_enabled=true`。

## 前后结果

| Model | Kernel ms before | Kernel ms after | Cast calls before→after | GEMM share |
|---|---:|---:|---:|---:|
| Qwen | 8.315 | 8.208 | 96→72 | 60.9% |
| DeepSeek | 14.862 | 14.659 | 112→84 | 68.2% |

cast减少量正好是24/28，说明每个FFN block删了一次FP32→BF16。新
`rms_norm_bf16_output_kernel`也精确出现24/28次，不是路由计数器单方面声称命中。

![Post FFN Norm profile](../assets/post-bf16-ffn-norm-profile.svg)

## 下一问题

剩余FP32→BF16 cast是48/56次。profile和模型结构共同指向每层的Attention投影输入
与其他BF16边界。下一节只先处理Attention Norm直入QKV Arena，不同时改O projection或
Attention core，这样才能将减少的24/28次与整模收益对上。
