# Step 82 — Direct Autograd accumulation into bucket views

Status: planned

Bucket views已经删除114个unpacked Storage/copy并把live恢复到transient水平，但backward仍先
为每个parameter生成普通gradient，再做114次pack copy；persistent bucket与新gradient重叠，
使peak仍比transient多33,269,000B。

下一步在每次backward前清零persistent bucket，并把parameter gradient预设为对应view，让现有
Autograd累加直接写入最终reducer Storage。目标：pack copy 114→0、communication只剩3次
all-reduce+scale、peak不高于transient、loss/所有参数与普通backward一致。需要先证明当前
accumulation语义不会替换预设Tensor地址，再进入Model-S三策略门；默认保持关闭。
