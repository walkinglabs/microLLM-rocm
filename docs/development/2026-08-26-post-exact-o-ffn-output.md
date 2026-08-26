# O修好后，差异到了FFN

给O projection使用跨batch一致的加法顺序后，O输出、残差和FFN前的归一化都完全相同。下一处差异
出现在整个FFN输出。

这还不知道是gate、up、SwiGLU还是down，所以不能直接替换全部FFN。也不能因O trace成功就说O值得
默认：它还要经过完整logits和速度门。

![FFN boundary](../../benchmarks/results/2026-08-26-post-exact-o-block0-trace/post-exact-o-trace.svg)
