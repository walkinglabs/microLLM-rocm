# Step 82 — Direct Autograd accumulation into bucket views

Status: implemented, Model-S A/B pending

Bucket views已经删除114个unpacked Storage/copy并把live恢复到transient水平，但backward仍先
为每个parameter生成普通gradient，再做114次pack copy；persistent bucket与新gradient重叠，
使peak仍比transient多33,269,000B。

下一步在每次backward前清零persistent bucket，并把parameter gradient预设为对应view，让现有
Autograd累加直接写入最终reducer Storage。目标：pack copy 114→0、communication只剩3次
all-reduce+scale、peak不高于transient、loss/所有参数与普通backward一致。需要先证明当前
accumulation语义不会替换预设Tensor地址，再进入Model-S三策略门；默认保持关闭。

实现采用leaf-only accumulation target：step 1仍用普通backward建立plan；后续step先按bucket
清零、把114个parameter gradient指向不重叠view，再backward。reducer验证地址/shape/offset后
直接all-reduce，pack/unpack copy均为0。CPU预置值、分叉、重复backward、shared Storage
Embedding和拒绝用例通过；tiny与单卡global-batch参考对齐。Model-S smoke显示communication
约3.47→1.65ms，但forward/backward约10.22→12.95ms，必须用正式轮换矩阵判断total。
