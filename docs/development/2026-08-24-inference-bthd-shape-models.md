# BTHD在长短文本和batch里都有效吗

日期：2026-08-24

六个case全部变快约1.085–1.142倍，完整logits逐位相同，peak也下降。

B1的Attention copy和总copy都是0。B2还剩一次很小的copy，它负责从每条序列挑最后hidden row：

    Qwen 7,168 bytes
    DeepSeek 12,288 bytes

它的source是unspecified，不属于Attention。因此正式门检查“Attention copy必须为0”，并把其他
copy单独报告，不能混在一起。

cache prefill和value trace还没有使用BTHD，继续走旧fallback。
