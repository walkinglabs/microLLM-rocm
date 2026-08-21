# Experiment 077 — 服务prefill只需要最后一个位置

Experiment 076显示T2048 B8 prefill远慢于PyTorch。开始写新Attention Kernel前，rocprof先
发现一个更基础的问题：benchmark为全部`B×T`位置计算词表logits，并在计时后把约9.96 GB
数据复制到CPU。真实生成只需要最后位置。

## 假设与反驳条件

最小假设：Transformer层仍处理完整context，但final output head只投影最后hidden position，
能删除不必要工作并降低峰值。

出现任一条件就拒绝：

- full与last完整logits超过`max_abs 1e-4 / RMSE 1e-5`；
- top token变化；
- Qwen或DeepSeek T2048 B8三进程中位数不改善；
- last模式峰值不下降；
- PyTorch仍计算full logits，造成比较语义不一致。

## 公共合同

```text
forward_inference(tokens)             -> [B,T,V]  full reference
forward_inference_last_logits(tokens) -> [B,1,V]  serving prefill

microllm_hf_infer --prefill-logits full|last
matrix --prefill-logits-mode full|last
```

PyTorch的last路径显式传`logits_to_keep=1`。模式写进raw/summary，不能静默混合。曾有一轮
baseline在运行期间修改runner，导致后半段PyTorch切到last；该pilot被判`invalid`，正式数据
重新使用冻结语义跑完。

## 三进程正式结果：T2048 B8

| 模型 | full micro | last micro | 自身加速 | last PyTorch | last micro/PT |
|---|---:|---:|---:|---:|---:|
| Qwen | 43,670 | 129,815 tok/s | 2.973× | 337,752 | 0.384× |
| DeepSeek | 50,447 | 66,444 tok/s | 1.317× | 126,878 | 0.524× |

Qwen full模式的三次结果波动较大，但last三次稳定在129–130k tok/s，改动量级远大于漂移。
DeepSeek收益较小但方向稳定。last仍未达到PyTorch，这个节点只删除无用output工作，不宣称
Attention已经优化完成。

## 宽shape复测

随后用last语义重新跑两模型、T32/512/2048、B1/2/4/8，共48/48进程成功，24/24个shape
的top token与PyTorch一致。B1/B8摘要：

| 模型 | context | micro B1/B8 | PyTorch B1/B8 | B8 micro/PT | B8效率 |
|---|---:|---:|---:|---:|---:|
| Qwen | 32 | 4,281 / 57,087 | 3,128 / 22,438 | 2.544× | 166.7% |
| Qwen | 512 | 93,321 / 197,843 | 46,114 / 254,101 | 0.779× | 26.5% |
| Qwen | 2048 | 102,962 / 129,986 | 165,546 / 335,472 | 0.387× | 15.8% |
| DeepSeek | 32 | 5,461 / 30,555 | 2,595 / 20,080 | 1.522× | 69.9% |
| DeepSeek | 512 | 50,683 / 87,617 | 41,540 / 116,159 | 0.754× | 21.6% |
| DeepSeek | 2048 | 60,776 / 66,419 | 82,247 / 126,614 | 0.525× | 13.7% |

短Qwen的超线性来自B1太小，不能解释成超过理想硬件扩展。更重要的是：删掉full output后，
长B8仍只有PyTorch的0.39×/0.53×，说明Experiment 076发现的Attention扩展问题并未消失，
只是现在不再与错误benchmark语义混在一起。

## 显存与传输

| 模型 | full peak | last peak | 峰值下降 | measured D2H |
|---|---:|---:|---:|---:|
| Qwen | 19.78 GiB | 5.15 GiB | 74.0% | 9.96 GB→4.86 MB |
| DeepSeek | 22.82 GiB | 8.00 GiB | 65.0% | 9.96 GB→4.86 MB |

D2H精确缩小2048×，对应context长度；它发生在forward timer之后，所以不能拿copy时间解释
全部吞吐提升，但它证明输出合同已经从`T`个位置缩到1个位置。

## 完整logits正确性

T2048 B1下，显式full最后一行与last完整词表逐项比较：

| 模型 | values | max abs | RMSE | top token |
|---|---:|---:|---:|---:|
| Qwen | 151,936 | 3.05e-5 | 3.61e-6 | 9707 = 9707 |
| DeepSeek | 151,936 | 1.57e-5 | 1.68e-6 | 30 = 30 |

tiny CPU/HIP测试还覆盖MHA/GQA、B2、shape和零隐式payload transfer。

最终仓库门：CPU 196/196、HIP 84/84、ASan/UBSan 194/194、Torch-enabled 199/199；
Python矩阵合同与优化日志validator也通过。

## Profile解释

profile包含一次warm-up和三次measured：

| 指标 | Qwen before→after | DeepSeek before→after |
|---|---:|---:|
| profiled forward | 414.42→214.91 ms | 324.56→247.66 ms |
| Kernel总时间 | 651.52→501.81 ms | 1292.85→984.43 ms |
| output-head GEMM | 148.68→0.92 ms | 286.21→1.02 ms |
| causal softmax | 131.45→130.72 ms | 132.03→131.81 ms |

output-head时间下降超过99%，softmax基本不变，说明加速来自删除不需要的投影，而不是把
Attention计时藏起来。softmax占比反而升到Qwen 26.1%、DeepSeek 13.4%，成为更干净的下一
热点。

![Serving last-logit prefill](../assets/serving-last-logit-prefill.svg)

## 决定

保留last-logit API与显式双模式，serving矩阵默认last；full继续作为训练式/reference工作负载。
下一节点从last profile出发，优先处理causal softmax和GQA的K/V head展开，不能继续引用已被
修正的Experiment 076 full-logits prefill比值。
