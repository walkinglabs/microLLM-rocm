# Experiment 100 — 596字节不变，H2D调用减半

Experiment 099显示R8/S4与R8/S2分别有32和56次H2D，但总量都只有596B。positions-aware每step把
token、position、cache row分三次上传；本节点只合并这三个调用。

## 实现

CPU token输入时先构造`[3,A]` Int32：token/position/row各占一行。一次H2D后用共享Storage view
得到`[A,1]` token和两个`[A]`metadata。若token本来在device，保留原路径。

![Packed decode metadata](../assets/packed-decode-metadata.svg)

## 机制门

| shape | H2D before | H2D after | bytes | D2H | D2D |
|---|---:|---:|---:|---:|---:|
| R8/S4 | 32 | 16 | 596→596 | 9 / 144B | 159 / 113,664B |
| R8/S2 | 56 | 24 | 596→596 | 17 / 136B | 159 / 113,664B |

CPU/HIP A/B/C scheduler test进一步固定6 calls/76B。输出、Cache、positions-aware calls和slot指标不变。

## 严格交替A/B

| shape | baseline median | candidate median | speedup | normalized old→new |
|---|---:|---:|---:|---:|
| R8/S4 | 3730.61 | 3852.46 | 1.033× | 1.6438→1.6841 |
| R8/S2 | 2571.07 | 2737.05 | 1.065× | 1.1469→1.1964 |

6/6逐对candidate更快，reference中位只漂移+0.22%/+1.25%。收益不大，但机制准确、没有坏shape，
因此保留。

## 下一步

现在每positions-aware step仍要一次metadata H2D。如果要完全device-resident，需要GPU维护row position和
admission映射，设计会明显扩大；在做之前应先与batched prefill和官方模型profile比较优先级。

数据见 [`100-data`](100-data/)。
