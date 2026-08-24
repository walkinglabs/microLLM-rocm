# Step 46 — Real backward gradient address stability

Status: complete, diagnostic keep

## Decision

18进程变化集合稳定。Qwen BF16 T8/T512为290/290稳定，DeepSeek T8为339/339稳定；DeepSeek
T512有198项、7,107,772,416字节换地址，tiny两精度均有4项K/V变化。Graph eligibility必须绑定
实际snapshot+context；下一步只对稳定case做optimizer-phase门。
