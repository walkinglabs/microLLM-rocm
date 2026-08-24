# Step 17 — exact BF16 gate/up startup gate

Status: complete

## Evidence

- six fresh tuner processes and 64 common passing candidates per shape;
- selected indices 76074/76091 improve operator Event 1.059×/1.032×;
- 24 fresh model processes cover cold/steady and default/exact;
- cold ratios are 0.990×/0.996×;
- process-wall ratios are 0.978×/0.981×;
- steady ratios are 0.973×/1.007×;
- complete logits are bit-exact and peak ratios are 1.0.

## Decision

Reject exact gate/up registration and keep the default. Both all-kernel preload and one-shape
solution selection are now closed as cold-start shortcuts.
