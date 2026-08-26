# Grouped gate/up cached-decode model gate

DeepSeek T2048/B2/N64 keeps all 64 tokens equal. Candidate throughput is 180.19
versus Arena baseline 178.46 tok/s, or 1.00968x, just below the declared 1.01
gate. Logit Max/RMS are 0.0546/0.0203 and peak changes by 11,456 bytes. Rejected.

![Model gate](model-gate.svg)
