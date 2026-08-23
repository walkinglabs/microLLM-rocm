# Graph test boundary

This directory tests the graph as a system, separately from individual operator
kernels.

| File | Question answered |
|---|---|
| `graph_construction_test.cpp` | Were operation names, parent edges, shared nodes, root shapes, and topological order recorded correctly? |
| `graph_gradient_alignment_test.cpp` | Do branches accumulate, repeated backward stay stable, views restore logical order, layout-fused RoPE/GQA match composed graphs, and bad seeds fail? |
| `hip_graph_alignment_test.cpp` | Does the same Transformer graph produce aligned CPU/HIP loss and every-parameter gradient without hidden host transfers? |

PyTorch graph alignment lives in `python/tests/test_operator_parity.py`, with
`tests/torch/operator_oracle.cpp` supplying microLLM values. It compares both small
graphs and a complete one-layer GQA Transformer.

A graph test does not replace an operator test:

- operator tests isolate one shape and one formula;
- graph construction tests isolate edges and traversal;
- graph gradient tests isolate the chain rule and accumulation;
- model graph tests prove the pieces still agree after composition.
