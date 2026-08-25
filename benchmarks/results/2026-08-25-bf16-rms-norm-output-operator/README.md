# Direct BF16 RMSNorm output operator gate

The baseline writes caller-owned FP32 RMSNorm output and then casts it into a
caller-owned BF16 Tensor. The candidate computes the same FP32 reduction and
rounds only the final store to BF16. Both use one explicit Stream, preallocated
Storage, three fresh processes, three warm-ups and 30 Event measurements.

Complete GPU outputs are bit-identical. At B1T1024, Qwen/DeepSeek Event speedups
are 1.866x/2.070x and wall speedups are 1.399x/1.511x with zero timed payload
transfers. The operator is admitted; model routing is a separate node.
