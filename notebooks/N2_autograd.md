# N2 — forward 为什么不能自动训练

## 具体处境与预测

For `loss = sum(a*b + a)`, write the expected gradients before running code:

```text
d(loss)/d(a) = ?
d(loss)/d(b) = ?
```

Use `a=[1,2,3]`, `b=[4,5,6]`. Then draw the graph and mark that `a` appears on two
paths.

## 旧办法

For three operations, handwritten gradients are readable. Add Embedding, RMSNorm,
RoPE, SwiGLU, Softmax, and cross entropy and the handwritten complete-model backward
becomes coupled to every forward change.

## 稳定失败

If the gradient of `a` is assigned instead of accumulated at graph branches, one path
overwrites the other. The expected `a` gradient is `[5,6,7]`; a broken engine returns
only `[4,5,6]` or `[1,1,1]`.

## 任务契约

```text
Goal: eager reverse-mode Value graph above Tensor.
Invariant: Tensor remains data/metadata; graph identity belongs to Value::Node.
Traversal: each node once in reverse topological order.
Accumulation: leaf gradients add across all paths and repeated backward calls.
Oracle: hand graph plus central finite differences.
Boundary: readable CPU FP32 first; HIP must reuse the same semantics.
```

## 运行

```bash
./scripts/configure.sh -DMICROLLM_ENABLE_HIP=OFF
./scripts/build.sh
ctest --test-dir build --output-on-failure -R AutogradTest
```

Focused evidence includes:

- branch accumulation;
- hand-valued matmul backward;
- central finite differences for multiply, cross entropy, RMSNorm, and RoPE;
- repeated-index Embedding scatter;
- causal Softmax future gradients equal zero;
- non-contiguous reshape/transpose backward;
- repeated backward without stale intermediate gradients.

## 审查 Agent 改动

Ask these questions before accepting generated backward code:

1. Does the closure capture forward values or read parameters after an optimizer step?
2. Are gradients accumulated or assigned?
3. Does transpose backward preserve logical order and stride semantics?
4. Does the finite-difference test perturb one scalar at a time?
5. Is an implicit device synchronization hiding in the formula?

## 当前反例

The correctness-first HIP path now executes the complete tiny Transformer forward and
backward graph without host/device transfers between graph entry and graph completion.
The graph engine, reverse-topological traversal, accumulation rules, backward formulas,
and readable HIP kernels are implemented in this repository. The dedicated conformance
test compares every parameter gradient with the CPU reference and checks runtime transfer
counters remain zero.

This does not yet make the complete training step device-native: loss reporting,
gradient-norm reporting, and AdamW still materialize host values. The first HIP kernels
also favor readability over parallel reductions, so correctness is proven before
competitive throughput.

## 下一步

Gradients can update parameters, but an interrupted process still loses AdamW moments,
data position, and random state. N3 treats training state as one versioned object.
