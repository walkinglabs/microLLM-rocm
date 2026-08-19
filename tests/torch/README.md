# PyTorch oracle test support

`operator_oracle.cpp` runs deterministic microLLM CPU cases and emits names, shapes,
and values as JSON lines. It does not calculate a second reference formula.

`python/tests/test_operator_parity.py` independently rebuilds the same cases using
PyTorch operations and PyTorch autograd. This separation matters: copying the microLLM
backward formula into its C++ test would let the implementation and its supposed oracle
share the same mistake.

The suite covers every public math operator, legal output shapes, invalid shape/dtype
contracts, view gradients, SGD, two AdamW steps and moment state, and a complete tiny
Transformer graph.
