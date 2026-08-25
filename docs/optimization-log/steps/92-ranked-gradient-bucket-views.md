# Step 92 — Ranked gradient-as-bucket views

Status: implemented, formal clean-revision measurement pending

Experiment 268证明persistent copy能消除60次steady backend allocation并恢复完整step速度，但
plan仍为bucket和57个unpacked gradients各保留一份Storage，容量124.69MB/rank，并做57次unpack。

下一节点只改变输出解释：每个parameter gradient成为persistent bucket Storage上的连续view。
pack与3次collective不变，unpack Storage/copy必须57→0，plan容量预期124.69→62.34MB。

必须检查view的storage、offset、shape、地址跨step稳定；完整参数/CPU/loss/故障门不变；正式矩阵
报告相对persistent-copy与逐参数的steady Reducer/total/current/peak。overlap仍不提前加入。

实现新增显式`bucket-views`。pilot得到allocation `[3,0,0]`、unpack `0`、plan 62.34MB；
current回到逐参数水平，peak相对逐参数+62.34MB、相对persistent-copy−62.34MB。单次相对
persistent-copy Reducer/total约1.043×/1.074×。四策略正式矩阵前不迁移overlap。
