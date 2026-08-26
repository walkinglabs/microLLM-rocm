# Experiment 310：Block 0 Core全Exact，完整Logits反而更差

Status: model route rejected

## 对照

两个policy都固定Q projection=296100、K/V projection=292135，让上游cache保持exact。candidate再加
QK=304681、P×V=295716。B1/2/4/8各两个fresh precision process和两个反向顺序performance process。

candidate precision进程同时完整导出block-0 scores、probabilities、P×V；大文件比较后删除。

## 局部因果成立

- candidate三阶段在B1/2/4/8跨batch和同batch全部位级相同；
- BF16 block-0 K/V cache仍全部位级相同；
- peak与backend allocation每个batch都不变。

## 完整模型否决

| Metric | upstream-exact | attention-exact |
|---|---:|---:|
| 全局logit Max | 0.00125325 | 0.00156164 |
| 全局logit RMS | 0.00029084 | 0.00031048 |
| B1 prefill speed | 1.000× | 0.94954× |

B2 Max从0.00074697恶化到0.00156164；B4/B8有所改善，但不能掩盖B2和全局RMS失败。candidate完整
logits相对upstream本身也改变，B1 Max为0.00122656。O projection和FFN仍会在core之后重新引入差异。

![Model gate](../../../benchmarks/results/2026-08-26-fp32-prefill-attention-model-gate/model-gate.svg)

## 决定

删除默认推广，保留显式scope作为反事实工具。下一实验不再强制四个batch使用同一index，而是使用
每个exact descriptor自己的“接近default且该batch更快”候选：B1保持default，B2只换PV，B4/B8用
各自QK/PV winner。它不承诺bitwise，只检验能否以更小数值变化换取稳健完整logit改善。

证据：[`result directory`](../../../benchmarks/results/2026-08-26-fp32-prefill-attention-model-gate/)
