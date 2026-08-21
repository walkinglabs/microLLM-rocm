# Experiment 051 evidence

- `pilot/`: one fresh pair for Qwen and DeepSeek at context 512.
- `formal/`: two models × two frameworks × three fresh processes.
- `profile/`: retained Qwen kernel and HIP API aggregates.
- `comparison.json`: stable-failure performance/memory boundary.
- `profile-summary.json`: aggregate totals, category partition and count identity.

The profile includes loading and three training steps. The Attention count identity is
clean; HIP API time is not used as per-step attribution because initialization is included.
