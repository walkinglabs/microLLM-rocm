# PA0 — 规则小世界中的第一次学习

Do not start from Autograd. Predict one update of `y = w*x + b` on paper, then compile
the independent teaching program; it does not link the framework:

```bash
g++ -std=c++20 -O2 pa/PA0/linear_update.cpp -o /tmp/microllm-pa0
/tmp/microllm-pa0
```

## Part A: three regular samples

Use `(1,2)`, `(2,4)`, `(3,6)`, `w=0`, `b=0`, mean squared error, and learning rate
0.1. Complete the table before execution:

| x | y | prediction | residual | contribution to dw | contribution to db |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | | | | |
| 2 | 4 | | | | |
| 3 | 6 | | | | |

Predict updated `w`, `b`, and loss.

## Part B: add an outlier

Add `(3,-10)`. The reference program demonstrates that mean loss falls while at least
one regular sample becomes worse. Explain why these statements are not contradictory.

## Submission

- completed hand table;
- program output and comparison with prediction;
- the locally worse sample;
- one change that would reduce outlier influence;
- Agent work record using the repository task contract.

## Agent boundary

An Agent may add CLI/table formatting and tests. The learner must predict the gradient
sign before running, identify the counterexample, and decide whether the update is
accepted.

The reference implementation's first run intentionally became a repository review
example: aggregate loss fields were not initialized, so a plausible loop printed a
garbage negative loss and failed its acceptance condition. Explicit state
initialization fixed it; code appearance alone would not have.
