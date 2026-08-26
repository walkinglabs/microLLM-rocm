# Official Qwen3-0.6B strict alignment

The pinned local fixture is strict-streamed to MI300X. Before any parameter write, the stored
lm_head payload is compared byte-for-byte with the primary embedding payload using bounded 1MiB
buffers. The model retains one tied runtime parameter.

The same official weights and token 1 are evaluated by microLLM and Transformers in FP32. All
151,936 logits and four greedy tokens are compared. Large model/tokenizer/logit payloads stay
outside Git; the committed summary and contract retain aggregate evidence.
