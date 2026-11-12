# T 正则机制核验

本实验回答：固定此前 T 的入口、骨架和候选域，增大线路数惩罚是否足以让真实结构成为训练目标的最优解？它不替代完整算法调参或新样本验证。

只分析 `docs/full_ablation_20261102/lost_exact_cases.csv` 中的六个完整恢复损失输入。原始 960 配置结果、冻结求解源和生产默认不变。候选域取每个 `R1T1E1` 记录的第一次 `joint_tail`，保持原始数据、权重界 `[0,2]`、单线程及 `MIPGap=0.0001`。

令固定骨架包含 k0 个内部集合，q 为候选二元变量的选择个数。对每个可行 q 求解

    phi(q) = min training_MAE, subject to original T constraints and sum(a)=q.

原来带惩罚的联合解等价于最小化 `phi(q) + lambda*q`，固定骨架的 `lambda*k0` 是常数。没有新增正阻抗下界；因此精确 q 的解可能有零权重被选候选，须分别记录二元选择数和正权重连接数。

真实连接仅在后分析中用于参考 LP 和结果核验，不用于约束剖面优化器。真实参考使用固定骨架与真实集合的并，保留固定骨架的计数成本。若真实连接不在 T 候选域或与骨架冲突，直接报告域障碍。

设真实参考损失为 L*、需选择 q* 个候选。真实参考能成为某个非负惩罚的最优解，需要同 q* 最优损失等于 L*，且以下区间非空：

    max(0, max_{q>q*} (L* - phi(q))/(q-q*))
    <= lambda <=
    min_{q<q*} (phi(q)-L*)/(q*-q).

同时报告 incumbent 和有效下界得到的内外区间；精度不足以区分的结果保留为未决。只比较两个已存结构得到的阈值是必要的成对比较，不代表整个候选域的充分条件。

预算：每个基数 MILP 至多 300 秒，每输入至多 3600 秒，固定家族 LP 至多 300 秒。先核验固定家族的独立 HiGHS LP和当前可行解，再运行六输入剖面。失败、超时及未决剖面保留原分母。

验证的参数区间属于六个诊断输入上的事后解释，不是可部署参数或独立测试集成绩。这里只核验 T 子问题，不将其最优拓扑声称为 T+E 全流程输出。

运行入口（拒绝覆盖已有逐输入结果；复现时使用新的 output 路径）：

```powershell
D:/apps/miniconda3/envs/Topo/python.exe -B research_experiments/t_regularization_audit/profile.py --output outputs/t_regularization_audit_replication
D:/apps/miniconda3/envs/Topo/python.exe -B research_experiments/t_regularization_audit/summarize.py outputs/t_regularization_audit_replication
```

默认输出为 `outputs/t_regularization_audit_20261102`。`summarize.py` 只处理已有系数和界，不新增求解或拟合；`figures.py` 绘制默认输出的正则区间。
