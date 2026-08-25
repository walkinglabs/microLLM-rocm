# Step 81 — Gradient views into reduced buckets

Status: planned

Persistent plan已经证明地址复用能让后续backend allocation 120→0并提升Model-S总步时，
但长期持有114个unpacked gradient使live/peak增加124.7/158.0MB。

下一步不再为每个参数申请unpacked Storage：all-reduce完成后，用bucket Storage、参数shape、
连续stride和offset构造Tensor view，直接交给optimizer。目标是unpacked Tensor Storage 114→0、
unpack copy 114→0、plan容量249,378,816→124,689,408B，同时保持30-step loss、参数、地址、
communication/total/current/peak门。默认仍关闭，正式A/B后决定。
