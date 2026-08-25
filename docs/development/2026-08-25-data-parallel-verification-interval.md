# 2026-08-25 — 参数一致性检查不是通信本身

`DataParallelConfig.parameter_check_interval` 现在显式控制全参数host审计：

- `1`：默认，每步检查，保持旧行为；
- `N`：只检查step能被N整除的时刻；
- `0`：显式关闭，适合单独的性能测量，不能作为参数一致性证据。

每步metrics新增 `parameter_check_performed` 与 `verification_ms`。跳过时verification严格为0，
执行时仍比较所有rank的全部参数。trace也新增独立阶段。

matrix会轮换三种policy的进程顺序，排除第1步lazy setup，只聚合step 2–20；loss轨迹必须逐项
相同。默认仍是1，性能选择不会静默改变正确性配置。

首版测试还发现旧host审计隐式承担了optimizer完成等待。直接跳过审计会让step在GPU工作未完成
时返回，并破坏后续生命周期。现在optimizer阶段显式同步每个设备，等待计入optimizer_ms；
verification interval只改变审计，不再改变step完成语义。

同一进程内两个trainer不能同时占用同一组设备/communicator，因此interval测试用不重叠scope
顺序构造trainer。这与当前“一个controller独占设备集合”的公开执行模型一致。
