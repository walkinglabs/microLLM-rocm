# 同一个算法能不能让B1/B2完全一致

M32/M64共有53个hipBLASLt solution。选择共同index `75892`，分别注册给两个exact shape，并由plan
持有所需GPU workspace。未注册shape仍使用默认`algo=nullptr`。

结果非常明确：3/3完整值pair的48个stage全部exact，最终151936维logits也全部exact。默认路径中
gate 0.015625、最终logits 0.153016的差异全部归零。

无trace性能代价：

| batch | 默认tok/s | 75892 tok/s | 比值 |
|---|---:|---:|---:|
| B1 | 5733.66 | 5517.73 | 0.9623× |
| B2 | 10655.17 | 10519.44 | 0.9873× |

所以同算法能恢复跨batch逐值一致，代价约1.3%–3.8%。index只在当前库版本有效，不能写死成全局
默认。仓库保留显式registry和CLI实验开关；普通用户仍走默认最快路径。

![Same BF16 algorithm](../optimization-log/assets/bf16-same-algorithm.svg)

详见[Experiment 110](../optimization-log/experiments/110-bf16-same-algorithm.md)。
