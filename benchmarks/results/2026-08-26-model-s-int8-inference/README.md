# Model-S INT8 inference boundary

Three fresh processes compare the same random Model-S seed, prompts and generation settings.
Each process performs one warm-up and two measured generation runs of four new tokens.

Context 1 keeps every Linear at M=1 and benefits from fused decode. Context 4 exercises the
explicit-dequantize prefill fallback and is the required counterexample. Token guards match in
both rows. The one-way transactional preparation temporarily holds old and candidate weights,
so preparation peak is reported separately from post-preparation resident bytes.
