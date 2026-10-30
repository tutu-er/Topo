# 概率DR价格与WZZT响应结果表

用户2026-10-30授权落实启发式与小规模测试。本目录隔离实现，不改生产默认。先读METHOD.md和已完成的REPORT.md，恢复时先检查私有protocol、checkpoint、plan.lock和实际进程，禁止重复启动未决记录。

本版锁定既有paper15 b0的全部四档原电压噪声，全部288旧训练和288旧验证保持用途。网络和负荷/DER均为既有合成生成器；不是iFlex实证价格校准，也不是独立现场数据。

实现入口：

- pricing_tables.py：预算菜单、独立/共同整段参与、精确概率和费用、价格查表。
- local_replay.py：多个raw历史解释、真实RNJ native库、共同调权、交叉删除、实际接受和最终选模。
- input_design.py：全45446均值输入设计、121单点、105双点同步、简单轮换、同对节点随机输入设计。
- physical_bridge.py、actual_worker.py：完整历史单位桥接，价格冻结后才生成新实际负荷/响应/AC；实际观测和真实轨迹分离保存。
- study.py：阶段协议、价格冻结、独立完整流程拟合、去重和所有成功/失败/未决记录。
- actual_runner.py：本轮实际运行的推荐入口，补齐同目标单点对照、失败已发生费用、未知费用和未完成全扫描的报告。旧study.actual不作为本轮推荐入口。

建议在Topo环境，PYTHONPATH指向仓库与rnj_wzzt_core，依次运行：

    python -B -m research_experiments.wzzt_dr_selection.dr_competition_bridge.study prepare --root <new-private-root>
    python -B -m research_experiments.wzzt_dr_selection.dr_competition_bridge.study plan --root <same-root>
    python -B -m research_experiments.wzzt_dr_selection.dr_competition_bridge.study actual --root <same-root> --max-records 56

本轮私有根为scratch/wzzt_dr_competition_bridge_v1。原始/聚合观测和真实物理轨迹不上传。协议源哈希变化或已有部分结果时，不用同目录重新启动。

未来真实模拟使用新replicate=10000+10*path+event，四事件profile索引0/2/0/2；相同policy比较共用自然/计量/参与随机流。四噪声×两响应相关性×七策略×一共享实际路径=56记录，且允许精确观测去重；绝不是56个独立自然负荷样本。验证供选模，最终恢复由选后真实有效块评价。

先完成响应表并检查其选择是否与普通设计存在实质差别，再决定是否运行完整实际批次。若表未完成或无值得付费的信息，保留该结论，不把未决损失填0，也不自动提高预算或样本。
