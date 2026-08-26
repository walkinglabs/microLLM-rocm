# Qwen3-0.6B pinned fixture and parser

The pinned official config and complete safetensors header are validated without committing model
payloads. Config/runtime unique parameters and stored file values are intentionally separate: the
checkpoint stores byte-identical embedding and lm_head tensors even though config declares tying.

Core model math and mapping are ready. Strict streaming must verify that duplicate alias rather
than silently ignore it; official logits remain pending that next node.
