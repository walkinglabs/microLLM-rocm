# Step 130 — Scoped O projection complete model gate

Status: completed by Experiment 314; rejected

对照`exact-core`与`exact-core+O296100`，固定DeepSeek T2048、B1/2/4/8、两个fresh process：

- BF16 cache和完整151,936 logits；
- O output trace因果证据复用Experiment 313；
- full prefill wall/tokens/s、peak、allocation、registry 5/168；
- Max/RMS都至少改善10%，每个batch≥0.95×。

结果：16个precision和16个反向排序performance进程显示，O让全局完整logit Max/RMS从
`0.001562/0.000310`降到`0.001175/0.000209`，改善24.7%/32.6%。但是B1 prefill只有
0.944×，低于0.95门；B8 Max也从0.000823升到0.001134。候选拒绝，O scope仅作诊断。

下一步不能直接把这个相对已拒绝exact-core的改善冒充对真实upstream的改善。Step 131将真实upstream
作为baseline，只在B2/B4使用core+O、B8使用core、B1保持upstream，做最后一次组合反驳。
