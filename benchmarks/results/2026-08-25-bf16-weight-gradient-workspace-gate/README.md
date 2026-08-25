# Allocating versus preallocated BF16 weight gradient

Experiment 249 compares the public allocating API with an equivalent
caller-preallocated composition after priming the exact-size caching allocator.

| Model | Event preallocated/allocating | Wall preallocated/allocating | Minimum wall | Decision |
|---|---:|---:|---:|---|
| Qwen2.5-0.5B | 1.037× | 0.986× | 0.956× | reject |
| DeepSeek Distill 1.5B | 0.886× | 0.889× | 0.840× | reject |

Every public call performs exactly three cache-reused logical allocations and
zero backend allocations. Preallocation does not clear the 1.01 wall gate on
either model shape and is stably slower on DeepSeek.

No workspace API is created. The standalone operator, allocation attribution
and benchmark runner remain. This closes the local BF16 weight-gradient
allocation track without restoring the rejected model route.

