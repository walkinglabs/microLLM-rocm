# 2026-08-19 — low loss but beyond-context generation failure

## Observation

The tiny model reached cross entropy `0.00673309` on four-token training windows.
Greedy generation from token zero produced:

```text
expected:  0,1,2,3,0,1,2,3
observed:  0,1,2,3,0,3,0,1
```

The sequence is correct through the trained positional window and fails when asked
to continue at positions not exercised by the context-four training batches.

## Competing explanations

1. The model memorized the cycle only at trained RoPE positions and did not learn a
   position-independent transition rule.
2. Cached decoding diverges from full-prefix decoding at later positions.

Existing evidence weakens explanation 2 because cached/full logits already match
for every position in an MHA and GQA four-token prefix, but it does not test this
trained model beyond position four.

## Rebuttal experiments to run

- compare cached and full-prefix logits on the trained generated sequence at positions
  four through seven;
- train the same seed with context eight while changing no other model component;
- train with randomized starting offsets so each transition appears at more RoPE
  positions.

The executable now reports `trained_prefix_matches=true` and
`beyond_training_context_failure=true`. It fails only if the model no longer learns
the in-window cycle; the longer failure remains visible rather than being deleted.
