# Qwen3 exact tokenizer-generated prompts

Four prompts are tokenized by the pinned local Qwen3 tokenizer and run at their exact lengths. The
orchestrator does not repeat or pad prompt content.

- English continuation: 22 tokens;
- Chinese explanation: 15 tokens;
- C++20 code request: 18 tokens;
- Qwen chat-template request with thinking disabled: 24 tokens.

Each prompt runs B1/B2 prefill and cached N8: 32 fresh workers and 16 aggregate rows.

## Result

- 32/32 workers complete;
- 14 direct pass, two visible precision mismatches, zero batch-invariance or limited rows;
- code and chat pass 8/8 rows directly;
- English B1 differs at step6: candidate 4416, Transformers BF16 785; B2 chooses 4416 in both;
- Chinese B2 differs at step2: candidate 104136, Transformers BF16 3837; B1 chooses 104136 in both;
- four B1/B2 complete-logit audits pass their strict common-FP32 gates and select the candidate token;
- both frameworks keep identical rows in all four B2 decode cases;
- all eight cached rows have exact KV bytes; maximum KV is 7,340,032 bytes;
- maximum microLLM/PyTorch peaks are 1,865,810,304 / 1,314,718,720 bytes.

The evidence supports the default-off phase policy on four exact prompts. It does not evaluate
language quality, sampling, long natural documents, broad prompt suites or other hardware.

Recreate the local manifest:

```bash
/path/to/transformers-python \
  benchmarks/single_gpu/make_qwen3_natural_prompt_manifest.py \
  --input /path/to/qwen3-runtime-manifest.json \
  --output /tmp/qwen3-natural-prompt-manifest.json
```

Then run the command saved in `matrix-command.txt`.

Files:

- `prompts.json`: portable text and token evidence;
- `matrix-*`: combined 32-worker matrix;
- `families/`: four exact-length submatrix summaries;
- `oracles/`: English/Chinese B1/B2 complete-logit audits;
- `summary.json`: final decision.
