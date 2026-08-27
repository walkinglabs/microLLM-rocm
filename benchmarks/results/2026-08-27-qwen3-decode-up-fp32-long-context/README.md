# Qwen3 phase-selective policy at T1024/T2048

This extends the fixed repeated-token shape matrix beyond the T512 calibration boundary. It covers
T1024/T2048, B1/B2, prefill and cached N1/N8/N32.

## Result

- 32/32 fresh workers complete;
- 16 aggregate rows: 10 pass, four `precision_mismatch`, two
  `batch_invariance_mismatch`, zero limited;
- all 12 cached rows have exact and cross-framework-equal KV bytes;
- microLLM keeps identical B2 rows in all six cached cases;
- Transformers BF16 keeps identical rows in four of six; T1024/B2 N8/N32 split at output 3;
- the new T1024 first split is token 2 vs 474; a B1 and B2 complete-logit oracle both select 2 for
  PyTorch FP32, microLLM FP32 and the phase candidate;
- the new T2048/B2 first split is token 16 vs 220; both FP32 argmax values and the candidate select
  16, but the microLLM/PyTorch FP32 maximum error is `2.193e-4`, just outside the fixed `2e-4`
  complete-logit gate;
- combining calibration and long states gives 10/10 fixed FP32 argmax matches and 8/10 strict
  common-FP32 complete-logit gates.

The route remains a default-off explicit precision policy. The evidence boundary now reaches T2048
for this one repeated-token prompt; it does not cover diverse natural-language prompts or Radeon.

Peak engine memory at T2048/B2/N32 is 3,171,713,024 bytes for microLLM and 4,719,116,800 bytes for
PyTorch. Maximum KV is 477,102,080 bytes in both frameworks. These are single-process shape samples,
not a repeated performance ranking.

Files:

- `shape-*`: complete 32-worker / 16-row matrix;
- `t1024-b1-step3-oracle-*`: first long-context B1 split;
- `t1024-b2-step3-oracle-*`: row-preserving B2 split;
- `t2048-b2-step4-oracle-*`: T2048 split and strict FP32 boundary;
- `summary.json`: compact final decision.
