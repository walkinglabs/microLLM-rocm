# First drift after exact prefill Attention core and O projection

The diagnostic control fixes Q/K/V projection, QK, P×V, and O projection with scoped
solutions. Across B1/B2/B4/B8, context, O output, residual, and FFN norm are bitwise
equal. The first new drift is the aggregate `ffn_output`.

- B2 Max: 0.0000219345;
- B4 Max: 0.0000143051;
- B8 Max: 0.0000181198.

Eight fresh processes compare 17 complete block-0 boundaries. This is numerical trace
evidence, not O projection performance admission.

![post exact O trace](post-exact-o-trace.svg)
