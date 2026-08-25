# Grouped Swish epilogue full-model gate

The same current binary alternates the retained scalar-SwiGLU policy and an
explicit gate-Swish-epilogue plus BF16 multiply policy. Both models use the
pinned grouped FFN/BTHD B1T1024 path, three fresh processes, two warm-ups and
five measured prefills.

Qwen is 1.00015x and DeepSeek is 0.99114x. Complete logits differ by Max/RMS
0.0973/0.0211 and 0.0362/0.00632. Peak bytes and engine allocation calls stay
unchanged. The explicit research route remains default-off and the model track
is closed.
