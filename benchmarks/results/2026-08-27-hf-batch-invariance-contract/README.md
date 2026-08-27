# Identical-input batch invariance evidence contract

The old PyTorch worker treated different tokens from identical batch rows as a generic process
failure. That discarded the actual rows and made numerical diagnosis impossible. The worker now:

- preserves every generated row;
- checks the full row matrix is deterministic across measured iterations;
- reports `generated_rows_equal` without aborting;
- lets the aggregate classify `batch_invariance_mismatch` before ordinary cross-framework token
  mismatch.

The fixed contract is exercised on official Qwen3-0.6B at T1024/B2/N8. microLLM's two rows are
identical and choose token 2 at the first sensitive step. Transformers BF16 row 0 chooses 474 while
row 1 chooses 2. A separate B1 complete-logit oracle confirms PyTorch FP32, microLLM FP32 and the
phase-selective candidate all choose 2; Transformers BF16 chooses 474.

Both frameworks allocate exactly 236,716,032 KV bytes. This is evidence about identical-row
numerics, not a throughput ranking: each framework has only one process sample in this contract.

Files:

- `matrix-raw.jsonl` / `matrix-summary.json`: two workers and one aggregate row;
- `oracle-raw.jsonl` / `oracle-summary.json`: four complete-logit policies at T1024/B1 step3;
- `summary.json`: compact contract result.
