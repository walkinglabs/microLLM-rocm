# Experiment 153 data

E5 candidate与同revision E4 control各36个独立worker，共72个worker。

- `candidate/`保存E5命令、3次GPU预检、raw、summary、逐case表和verification；
- `control/`保存E4命令、3次GPU预检、raw和summary；
- 两边合计24个FP8目标行，每行比较151,936个完整logits；
- Qwen每worker记录384次dynamic与96次post；DeepSeek为452/112；
- E5八项Max/RMS全部恶化，完整precision为0/4；
- 两项T512速度门通过，resident与peak差值都为零；
- fresh 50步构建、CLI合同与matrix合同均保留。

原始文件固定revision `821e7b8`。归档后的源码删除模型级E5开关，但保留底层混合格式原语；
历史命令只能用于复现实验，不能被当成当前CLI文档。
