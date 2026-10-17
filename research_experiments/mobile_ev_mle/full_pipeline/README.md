# WZZT—AC—EV完整拓扑测试

先读[实验方案](protocol.md)和[数学口径](math.md)。本轮将使用公开实测建筑进口功率形状驱动合成的6/8末端完整径向网络，候选由实际WZZT/RNJ产生，保留候选漏真，再比较恒充、平方基线与G-SQ/G-AQ。预测只读到达历史。

当前阶段：组件实现与开发成本验证，正式校准/评价尚未开始。原始数据在执行副本data_external/opsd_household_20200415，原始CSV不随代码提交；实际使用的时间窗、hash和精简输入将随证据交付。

源码：C:/Users/23761/.codex/worktrees/36a1/Topo，分支codex/ev-train-validation-design。
执行副本：D:/0-github_workspace/Topo/outputs/ev_full_pipeline_stage_20261018。

科学源码、实际配置和窗口清单冻结后才允许正式评价；后续结果、失败分母、求解界和续接状态均在本目录记录。没有将既有g_precision结果改写为完整拓扑测试。
