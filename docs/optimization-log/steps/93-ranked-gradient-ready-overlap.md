# Step 93 — Ranked gradient-ready overlap

Status: planned

Experiment 269已经建立one-process-per-GPU persistent views：3个自然bucket中，前两个在完整
backward结束前ready；unpack和later allocation均为0。下一节点只改变communication enqueue
时机，不改变bucket Storage、view、collective顺序或optimizer。

每个rank在default Stream记录bucket-ready Event，communication Stream等待后pack并按固定range
顺序发起RCCL average；optimizer前等待全部bucket。两个独立进程必须以相同collective顺序进入
RCCL，任一rank错误仍由launcher终止peer。

必须证明完整参数/CPU/loss/故障门不变，overlapped bucket数量精确，steady later allocation仍0，
current/peak不高于views。正式比较同步views与overlap views；若total没有稳定改善则拒绝overlap。
