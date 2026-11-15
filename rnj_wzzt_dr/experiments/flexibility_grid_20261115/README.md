# 灵活性采购的数据量／误差网格

固定两个核心物理输入，增加Paper15、Soumalas11对照，重新辨识36个历史采样量与量测标准差组合。三个分布约束、旧绝对恢复参考、无采购和同费轮换共享同一实际路径。

当前完整数学模型及原核心结果见 [CORE_MODEL.md](../distribution_panel_20261115/CORE_MODEL.md)。新网格规则见 [PROTOCOL.md](PROTOCOL.md)。采样稀疏化、量测误差、预测误差是三个不同问题，本批前两个构成主网格，预测误差固定5%。

36/36格与4608/4608条独立核验已完成。完整结果及正、负机制见 [RESULTS.md](RESULTS.md)。本批三种新方法总体均未胜过无采购或旧参考；Pengwah正例与Soumalas退化例同时保留。后续分开训练量和验证量的设计见 [NEXT_DESIGN.md](NEXT_DESIGN.md)，尚未执行。

四网络逐节点配置、36格实际调用与恢复变化、Soumalas负例的节点时段图和评分分解见 [RESOURCE_EFFECTS.md](RESOURCE_EFFECTS.md)。这次存量数据核算将冻结响应矩阵误差与AC近似余项分开，后者不是所查明显负例的主要来源；新增绘图和诊断入口均不重跑实验。

入口从仓库根运行：

1. `python -B -m rnj_wzzt_dr.experiments.flexibility_grid_20261115.data --prepare --out outputs/flexibility_grid_data_20261115_v1`
2. `python -B -m rnj_wzzt_dr.experiments.flexibility_grid_20261115.data --run --out outputs/flexibility_grid_data_20261115_v1`
3. `python -B -m rnj_wzzt_dr.experiments.flexibility_grid_20261115.run --index outputs/flexibility_grid_data_20261115_v1/grid_index.json --out outputs/flexibility_grid_smoke_20261115_v1 --smoke`
4. `python -B -m rnj_wzzt_dr.experiments.distribution_panel_20261115.audit --run outputs/flexibility_grid_smoke_20261115_v1 --out outputs/flexibility_grid_smoke_audit_20261115_v1`
5. `python -B -m rnj_wzzt_dr.experiments.flexibility_grid_20261115.run --index outputs/flexibility_grid_data_20261115_v1/grid_index.json --out outputs/flexibility_grid_execution_20261115_v1 --smoke-audit outputs/flexibility_grid_smoke_audit_20261115_v1/report.json`
6. `python -B -m rnj_wzzt_dr.experiments.distribution_panel_20261115.audit --run outputs/flexibility_grid_execution_20261115_v1 --out outputs/flexibility_grid_audit_20261115_v1`
7. `python -B -m rnj_wzzt_dr.experiments.flexibility_grid_20261115.report --audit outputs/flexibility_grid_audit_20261115_v1/report.json --out outputs/flexibility_grid_report_20261115_v1`
8. `python -B -m rnj_wzzt_dr.experiments.flexibility_grid_20261115.plot --report outputs/flexibility_grid_report_20261115_v1/report.json --out outputs/flexibility_grid_figures_20261115_v1`

正式入口必须读取8条接线执行全部通过的独立审计收据，并核对同一网格索引、零报价／执行异常和当前源文件哈希。接线后若修改运行源，需重新接线并另存审计。正式36格完成后，再使用第4步同一审计入口核查正式目录。

已存在输出目录不会覆盖；完成格不重新分配拟合预算，部分格必须先核实际进程与已用预算。运行会立即封存每个方法返回的动作，旧完成批及其源码快照保留。原始时序、拟合参数和报价轨迹留在忽略目录outputs，仅提交代码、协议、必要测试与汇总。

36格来自4个已知物理输入，每格1份新预测、16条随机交付与量测路径。用途是找出改善、无改善和恶化的区域；后续确认应使用独立物理日期，不能将网格中效果最好的格直接称为平均泛化表现。
