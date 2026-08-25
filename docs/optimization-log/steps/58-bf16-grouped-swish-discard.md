# Step 58 — grouped gate Swish epilogue

Status: complete, reject model route

## Decision

T1024 pointer-stable operator为1.097×/1.069×，但整模为1.000×/0.991×，logits Max为
0.0973/0.0362。显式研究开关保留但默认关闭，FFN activation局部路线关闭。
