# Step 81 — Gradient views into reduced buckets

Status: complete, kept explicit

Persistent plan已经证明地址复用能让后续backend allocation 120→0并提升Model-S总步时，
但长期持有114个unpacked gradient使live/peak增加124.7/158.0MB。

下一步不再为每个参数申请unpacked Storage：all-reduce完成后，用bucket Storage、参数shape、
连续stride和offset构造Tensor view，直接交给optimizer。目标是unpacked Tensor Storage 114→0、
unpack copy 114→0、plan容量249,378,816→124,689,408B，同时保持30-step loss、参数、地址、
communication/total/current/peak门。默认仍关闭，正式A/B后决定。

实现增加独立`gradient_bucket_views=false`开关，必须同时启用persistent与in-place average。
每个参数Tensor保留原shape和连续stride，但Storage与所属reduced bucket共享，offset等于此前
参数元素前缀和。Model-S smoke达到unpacked Storage/copy 114→0、首步backend alloc 120→6、
后续0、plan容量减半，live回到transient水平；正式三策略A/B仍待运行。

结果：相对persistent-copy communication/total 1.131×/1.067×，相对transient
1.937×/1.367×；114个Storage/copy归零，live与transient相同。peak仍高33,269,000B，
因此不设默认；下一步让Autograd直接向bucket view累加，删除114次pack copy和双表示峰值。
