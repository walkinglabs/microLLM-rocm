# SwiGLU Autograd submission attribution

Six fresh MI300X processes rotate three mathematically equivalent FP32 F+B paths at 64K and 1M:

- native PyTorch Autograd: `F.silu(gate) * up`, then sum/backward;
- microLLM Python-registered Autograd: fused forward, sum, scalar-seed fused backward;
- manual fused submission: the same microLLM forward, sum and scalar-seed backward without the
  Autograd engine callback.

All complete losses and both gradients match within `4.77e-7`. At 1M, Event medians are
0.1119/0.1408/0.0290 ms for native/custom/manual. Manual is `4.855×` custom and `3.859×` native;
64K gives `5.271×` and `4.105×`. This localizes the remaining custom F+B gap to the Python
`register_autograd`/engine submission boundary rather than HIP arithmetic.

Manual peak is not a production memory comparison: the attribution loop keeps its explicit output,
loss and gradient tuple alive together. Its role is timing/causality. `raw.jsonl` retains all
processes; `summary.json` contains the complete three-way medians.

