# BF16 V input capability rejection

A temporary candidate retained grouped V in BF16 and changed only the P×V
value layout to BF16 while keeping FP32 probabilities, FP32 accumulation and
FP32 context output. Ordinary interleaved BTHD and zero-stride GQA descriptors
both returned hipBLASLt status 6 before timing. Their retained FP32-V paths
passed. Candidate APIs were removed and no model route was created.
