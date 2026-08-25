# Direct BF16 Attention Norm full-model gate

The same binary alternates explicit Attention Norm fusion false/true while the
retained FFN Norm fusion remains enabled on both sides. The policy is the
current grouped-FFN/grouped-QKV/BTHD B1T1024 path, with three fresh processes,
two warm-ups, five measured prefills and complete vocab logits.

Qwen/DeepSeek improve 1.01309x/1.01303x. Complete logits are bit-identical,
measured allocations fall by 120/140, and peak bytes fall by 3,670,016/
6,291,456. BF16 QKV Arena now enables this route by default; explicit false
remains the rebuttal path.
