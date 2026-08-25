# 2026-08-25 — 长训练门需要逐步 loss 和完整参数比较

两步训练只能发现明显错误，不能发现误差是否随 step 放大。为 Step 70 增加两种可选输出：

- `--loss-trajectory-output`：只保存 measured window 的每步 loss；
- `--gate-up-parameters-output`：把训练后的全部 gate/up FP32 master 参数保存为临时
  safetensors。

这两个输出发生在计时结束后，不进入 tokens/s。普通训练不传参数时，既不写文件也不改变
默认 JSON 体积。

`microllm_compare_safetensors` 会检查两份文件的名字、shape、dtype 和全部元素，报告真实
Max/RMS、最差 Tensor 与有限性。长轨迹 runner 每完成一个进程就保存 JSONL；多 GB 参数
文件在比较后立即从 runner 自己的临时目录删除，仓库只保留汇总证据。

真实 Qwen 3-step 烟测导出 48 个 Tensor、209,190,912 个 FP32 值。完整比较结果为
Max `4.17e-5`、RMS `3.37e-7`，最差 Tensor 名字也被记录；两份临时参数文件随后删除。
