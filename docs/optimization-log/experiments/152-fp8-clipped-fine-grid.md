# Experiment 152：只裁5%，worst RMS也翻了2.15倍

精细网格保持Exp151的weight scale、O-only scope、模型和prompt，只测试靠近1.0的fraction：

| fraction | worst RMS / F1 | worst Max / F1 | 全部top一致 |
|---:|---:|---:|---|
| 1.0 | 1.000× | 1.000× | 是 |
| 0.95 | 2.155× | 1.609× | 是 |
| 0.90 | 4.983× | 3.218× | 是 |
| 0.85 | 8.254× | 5.227× | 是 |

![Clipped activation fine grid](../assets/fp8-clipped-fine-grid.svg)

fraction1再次与Exp148四case逐值相同。0.95只裁掉amax顶部5%，但完整向量worst RMS已经翻倍；
top不变不能掩盖这个事实。四个fraction仍0个完整精度门通过。

结合Exp151的0.75/0.5/0.25，预设搜索区间0.25–0.95全部比1.0差。模型activation clipping方向
因此关闭：删除ModelConfig/CLI/通用matrix fraction和专用pilot runner，保留底层C++算子用于
未来不同模型的显式研究，不设模型默认。

下一精度问题不能继续调全局amax fraction，应转向不同粒度、分层校准或其他数值格式。
