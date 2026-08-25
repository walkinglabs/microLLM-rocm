# T1024 exact QK full-model gate

The retained B1T1024 BTHD policy compares baseline with one exact QK index per
model over three fresh processes and seven prefills. Qwen dispatches 168 times;
DeepSeek dispatches 196 times. Peak memory and allocation counts are unchanged.

Qwen improves 1.051× but complete logits differ by Max/RMS 0.0733/0.0157,
failing the predeclared 1e-4/1e-5 gate. DeepSeek is exactly aligned but improves
only 1.002×, failing the 1.01 performance gate. No exact T1024 Attention solution
is enabled by default.
