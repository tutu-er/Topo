# WZZT—AC—EV完整拓扑测试（已完成）

先读[简明结果报告](report.md)，再按需查[事前实验方案](protocol.md)、[数学口径](math.md)、[复现与证据](REPRODUCE.md)。

本批用OPSD建筑实测进口功率形状驱动6/8末端合成径向网络，实际运行候选生成、历史AC拟合、因果预测、EV MILP、完整AC含噪执行、候选更新及共享线路重拟合。10个主方法，开发24/校准48/评价64个条件，对应6/12/16个日期簇。Q、线路、源、电压和表计为声明的合成模型。

G-SQ全分母点正确48/64，P-SQ42/64；共同含真且可用的41对均无点判别错误。两者校准后唯一正确均24/64，G-SQ更少唯一错误伴随更多未决。当前证据尚未建立稳定的最终可靠性增益；报告保留候选漏真、联合拟合失败和移动域未闭合的求解间隙。

科学计算和物理审计已完成，117项测试通过，23,189次完整AC回放通过。主证据包约19.97MiB，补充包保存全树F1与覆盖审计。源码分支codex/ev-train-validation-design，冻结科学实现dfe24edf；不启动新批次，不更新原长PDF。

执行副本：D:/0-github_workspace/Topo/outputs/ev_full_pipeline_stage_20261018。
当前状态见[checkpoint.json](checkpoint.json)，后续复现见[resume.md](resume.md)。
