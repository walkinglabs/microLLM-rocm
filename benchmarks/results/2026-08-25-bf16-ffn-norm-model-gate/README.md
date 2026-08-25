# Direct BF16 FFN Norm full-model gate

The same binary alternates explicit `--bf16-ffn-norm-fusion false/true` on the
retained grouped-FFN/BTHD B1T1024 path. Each policy/model pair uses three fresh
processes, two warm-ups, five measured prefills and complete vocab logits.

Qwen/DeepSeek improve 1.0122x/1.0092x. Complete logits are bit-identical, peak
bytes are unchanged, and measured engine allocations fall by 120/140 (exactly
24/28 layers times five measured forwards). BF16 FFN Arena now enables this
route by default; explicit false remains the rebuttal path.
