# 2026-09-05 — M7 Qwen3 MoE real-checkpoint gate

## Contract

Pin a real Qwen3-MoE checkpoint, add it to `data/model_fixtures.toml`, and run a
fixed-prompt → fixed-token-sequence gate through `hf_infer`, matching the same
standard as the existing Qwen2.5/Qwen3/DeepSeek fixture rows.

## First finding: this machine has network access

M4/M5/M6 all assumed no network access (an assumption never actually re-checked
after the first session). While preparing M7's fixture, `curl` and
`huggingface_hub` both worked immediately. This matters retroactively: M6's
weight-layout decision was made by reading only `transformers`' installed
*source code*, specifically because a real checkpoint wasn't thought to be
obtainable. It was obtainable. See below.

## Second, much bigger finding: M6's weight layout was wrong

Before writing any M7 fixture code, several `tiny-random` Qwen3-MoE test
repositories on the Hub were inspected (searched via `HfApi.list_models`) to
find one usable as a small real checkpoint:

| repo | decoder_sparse_step | hidden_act | field name | rope_theta |
|---|---|---|---|---|
| yujiepan/qwen3-moe-tiny-random | 2 (mixed dense/MoE) | silu | `num_experts` | top-level |
| katuni4ka/tiny-random-qwen3moe | 2 | silu | `num_experts` | top-level |
| hf-tiny-v2/tiny-random-Qwen3MoeForCausalLM | 1 | **gelu** | `num_local_experts` | nested `rope_parameters` |
| optimum-internal-testing/tiny-random-qwen3_moe | 2 | silu | `num_experts` | top-level |
| **amd-quark/tiny-random-qwen3_moe** | **1** | **silu** | **`num_experts`** | **top-level** |
| official Qwen/Qwen3-30B-A3B | 1 | silu | `num_experts` | top-level |

Only `amd-quark/tiny-random-qwen3_moe` and the official model matched what
this repo's parser supports (M4's field-name assumptions held up for the
fields checked). But downloading and inspecting `amd-quark/tiny-random-qwen3_moe`'s
`model.safetensors` — and separately fetching just the official
`Qwen/Qwen3-30B-A3B`'s `model.safetensors.index.json` (a few KB, no need to
download 60GB of weight shards to see the tensor *names*) — showed both real
checkpoints store, per layer:

```text
model.layers.N.mlp.experts.E.gate_proj.weight   [ffn_dim, dim]
model.layers.N.mlp.experts.E.up_proj.weight     [ffn_dim, dim]
model.layers.N.mlp.experts.E.down_proj.weight   [dim, ffn_dim]
```

— per-expert separate tensors, exactly M5's *original* design, not the fused
`gate_up_proj [num_experts, 2*ffn_dim, dim]` M6 had switched to. M6's mistake
was checking only `transformers`' current *in-memory* `Qwen3MoeExperts` module
source, which does use a packed `nn.Parameter`, without checking whether that
matches what is actually serialized on disk. It doesn't, at least not yet:
loading the real checkpoint with `Qwen3MoeForCausalLM.from_pretrained()` and
inspecting the resulting `model.layers[0].mlp.experts.gate_up_proj` confirmed
`transformers` converts the on-disk per-expert tensors into its packed
in-memory shape via a plain `torch.cat([gate_proj, up_proj], dim=0)` at load
time — a conversion layer this repo does not need to replicate, because it
never had a packed in-memory representation problem to solve in the first
place, once the checkpoint format is understood correctly.

**This was surfaced to the user before writing any more code**, since it meant
un-doing the M6 commit's core decision, not just adding to it. Given the choice
between reverting to per-expert internal storage (matching real checkpoints,
zero new loader machinery) or teaching `WeightMapping` a new
many-external-tensors-into-one-internal-slice capability to keep the packed
layout, the user chose the revert.

## What was reverted / replaced

- `MoeFeedForward` is back to `std::vector<Linear>` per projection (M5's
  original design), not `Value gate_up_proj_`/`down_proj_`.
- `ops::moe_split_gate_up`/`_backward` and `autograd::moe_split_gate_up` were
  **deleted** (not deprecated — they solved a problem that, per current
  evidence, does not exist in any real checkpoint) and replaced with
  `ops::moe_stack_experts`/`_backward_one` and `autograd::moe_stack_experts`:
  stacks `num_experts` separate same-shaped tensors into the
  `[num_experts,rows,cols]` shape `moe_expert_ffn` expects. No transpose is
  needed this time — this repo's own `Linear` weight layout (`[input,output]`)
  already matches `moe_expert_ffn`'s per-expert convention, unlike HF's
  nn.Linear-style `[output,input]` per-expert tensors, which still need
  `Transpose2D` at *load* time (unchanged from M5 — this part was never wrong).
- `qwen_style_weight_mapping`'s MoE branch is back to a per-expert loop
  (`Transpose2D` per tensor), matching the real files exactly.
- `MoeFeedForward::forward()`/`forward_tensor()` now call `moe_stack_experts`
  three times (gate/up/down) instead of one `moe_split_gate_up` plus a
  `transpose`.
- The model-level oracle case (`emit_moe_model_gate_case` +
  `python/tests/test_operator_parity.py`'s MoE section) was rewritten to a
  per-expert fixture, reproducing in Python exactly the `cat([gate,up],dim=0)`
  conversion `transformers` does internally to populate the real
  `Qwen3MoeSparseMoeBlock` from per-expert tensors, then compares gradients by
  slicing the packed gradient back apart. Still passes at the same `2e-3`
  tolerance against the real, unmodified module.

## Two more real-checkpoint findings, found by actually running `hf_infer`

Getting the checkpoint to load through the real CLI (`apps/hf_infer.cpp`), not
just through `TransformerModel` directly, surfaced two more gaps:

1. **`router_aux_loss_coef` rejection made every real checkpoint unloadable.**
   M4 rejected this field's mere presence outright. Every config checked in
   the table above — official and third-party alike — serializes it with its
   dataclass default. The rejection is now removed: the field configures a
   training-time loss this repo does not implement anywhere, so accepting it
   silently has no behavior to misrepresent (unlike `decoder_sparse_step`/
   `mlp_only_layers`, which really would change the computation if ignored).
2. **`torch_dtype` vs `dtype`.** This checkpoint's `config.json` serializes
   `"dtype": "float32"` instead of `"torch_dtype"` — a newer Hugging Face
   config schema. `load_huggingface_config` now accepts either key (this field
   is purely informational metadata, never read to select a compute dtype
   anywhere in this codebase).
3. **`qwen3_tied_weight_aliases` assumed every tied checkpoint redundantly
   serializes `lm_head.weight`.** That's true for the pinned Qwen3-0.6B
   fixture but not for this MoE checkpoint (`tie_word_embeddings: true`, no
   separate `lm_head.weight` tensor in the file at all). The existing alias
   *mechanism* in `model.cpp` is correct — an alias it's told about is
   required to be present — the bug was in `hf_infer.cpp` applying the alias
   *unconditionally* whenever `qk_norm && tie_embeddings` held, regardless of
   whether the weight file actually has the redundant tensor. Fixed by
   checking tensor presence (`io::inspect_safetensors`/`inspect_safetensors_index`)
   before applying each declared alias.
4. **`external.model.weight_bytes()` (used only for a benchmark-reporting
   metric, `resident_weight_bytes`) throws for MoE configs**, same as
   `parameter_count()` (M4's deliberate choice — the formula-based count
   doesn't know how to count MoE experts). Replaced with
   `model.parameter_count() * sizeof(float)` — the *model's own* live
   parameter count, which already correctly reflects real loaded weights for
   both dense and MoE models, and is arguably a strict improvement even for
   dense models (it reflects what's actually loaded rather than a formula).

None of these four were MoE-routing bugs; all four were "this code path was
never exercised against an actual downloaded, actually-tied, newer-schema
checkpoint before."

## The fixture

`amd-quark/tiny-random-qwen3_moe`: 2 layers, hidden_size 128, 4 experts,
`num_experts_per_tok` 2, `moe_intermediate_size` 256, tied embeddings, 44
tensors, ~20.3M parameters, FP32, ~81MB on disk. It is a *structurally real*
`Qwen3MoeForCausalLM` (real architecture, real class, real tensor names/shapes)
with *randomly initialized* weights — not a pretrained model. This was the
only readily available small option; the smallest genuinely pretrained
Qwen3-MoE model, `Qwen/Qwen3-30B-A3B`, is far too large to download or run
without a GPU on this machine. The gate below validates numerical/engineering
correctness (does this engine's forward match a real HF module's forward,
bit-for-bit down to the same greedy token choices), not "does the model
produce sensible text" — which a random-weight checkpoint cannot demonstrate
regardless.

**Licensing note:** this repository, and every comparable tiny-random
Qwen3-MoE test repository checked, has no LICENSE file and no license tag on
Hugging Face. Flagged to the user, who decided to add the fixture anyway with
`license = "unspecified (synthetic random-weight test checkpoint; no LICENSE
file published on Hugging Face)"` rather than the registry's usual real
open-source license string — reasoning: the actual weight payload is never
committed to this repository (only this pinned-revision registry entry is),
and there is no pretrained/copyrighted training content here to license in the
first place, only random initialization values.

## The actual gate

```text
$ transformers (Qwen3MoeForCausalLM.from_pretrained, float32, greedy,
  do_sample=False, use_cache=True): generate([9707, 1879], max_new_tokens=4)
  -> [1879, 1879, 1879, 1879]

$ microllm_hf_infer --config config.json --weights model.safetensors \
  --tokens 9707,1879 --new-tokens 4 --device cpu
  -> generated_tokens: [1879, 1879, 1879, 1879]
```

Exact match. `loaded_tensors: 44` and `parameter_count: 20334464` both agree
with the safetensors file's actual contents (independently cross-checked by
hand-summing every tensor's element count). `tools/prepare_hf_fixture.py
prepare`/`validate` both pass against the registry entry added to
`data/model_fixtures.toml`.

Existing `hf_infer` CLI/CTest contracts (`test_hf_cli_binary_contract.py`,
`test_hf_cli_batch_logits.py`) were re-run against the rebuilt binary and still
pass — the `hf_infer.cpp` fixes above are additive/narrowing, not disruptive
to the existing dense-model paths.

## Verification

```text
CPU Debug (MICROLLM_ENABLE_HIP=OFF): 302/302 microllm_tests passed (was 298
after M6's commit; the M6 tests were rewritten in place for the M7 layout, net
+1 from a new router_aux_loss_coef acceptance test, +2 replacing the deleted
moe_split_gate_up tests with moe_stack_experts ones).
CPU ASan/UBSan: 302/302 passed clean.
scripts/audit_test_coverage.py: pass tensor_ops=207 graph_api=49
TorchOps.OperatorParity (conda torch env, transformers 5.8.0), rerun after the
M7 rewrite: all 4 tests pass, including the per-expert MoE model-level gate
against the real, unmodified Qwen3MoeSparseMoeBlock.
Real-checkpoint gate (amd-quark/tiny-random-qwen3_moe, manual run -- no
automated CI wiring exists for data/model_fixtures.toml entries generally,
matching the existing Qwen2.5/Qwen3/DeepSeek rows, none of which are CMake-
wired either): microllm_hf_infer's generated tokens exactly match
transformers' golden greedy generation.
```

## Current boundary

The fixture is a random-weight structural test checkpoint, not a pretrained
one — it proves engine correctness, not model quality, and cannot be used to
sanity-check "does this produce reasonable text." No ROCm hardware exists on
this machine, so the HIP kernels this whole MoE feature ultimately depends on
(M2, still uncompiled) remain unverified; this gate only exercises the CPU
path. The `hf_infer.cpp` fixes are narrowly scoped to what broke while loading
this one checkpoint — other CLI paths (streaming/sharded loading, BF16/FP8/INT8
MoE preparation, cached decode) remain untested against any real MoE
checkpoint and, per M5/M6, explicitly throw rather than silently running for
MoE.
