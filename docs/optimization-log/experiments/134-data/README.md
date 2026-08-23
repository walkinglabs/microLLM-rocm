# Experiment 134 data

Qwen/DeepSeek T512 retained multi-block dynamic-activation whole-process rocprofv3 profile。

仓库保存命令、三次0/0预检、parsed summary、kernel/API stats和程序输出。大型JSON/Perfetto/逐调用
trace不进入Git历史，可由命令重新生成。`other`混合模型加载、weight准备、prefill与销毁，不被
错误归因成measured forward；只有dynamic三段、Tensile GEMM和fallback列入明确归因集合。
