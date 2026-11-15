# 四类设备灵活性配对测试

入口仅更换灵活性资源，不重跑辨识器或重新拟合R/X。设备为PV、普通单向EV、电阻热水和独立AC家庭储能。取消旧独立SVG；节点和有功容量由历史登记、原缩放、固定容量排名和10%参与比例确定。详细设备假设见 [MODEL.md](MODEL.md)，执行与验收范围见 [PROTOCOL.md](PROTOCOL.md)。

主参照为已完成的 `outputs/node_time_pq_execution_20261115_v1`。保持其三输入、三概率式、训练得到的P/Q动作带、冻结历史目标与全竞争库、预测、AC及原表计噪声；每格16条重放。原结果和原始数据不覆盖。

设备登记只有聚合kW；光伏gross曲线、逆变器kVA、电池kWh/SOC和设备基线是预先声明的场景参数。净负荷只叠加动作增量，不把分项设备基线再次加入。

运行记录、源快照、轨迹、报价和原始数组留ignored outputs。先检查接线与独立审计，再启动正式测试；失败和超时保留在分母中。报告将区分内部拓扑恢复、完整响应支持恢复、预测/AC/量测收益、调用/预约/电费以及可行性与最优性。

已完成的结果与限制见 [RESULTS.md](RESULTS.md)，容量/位置及厂家参数依据见 [RESOURCE_BASIS.md](RESOURCE_BASIS.md)。本地完整报告位于 `outputs/equipment_flex_report_20261115_v2/REPORT.md`，原始运行与独立审计分别位于 `outputs/equipment_flex_execution_20261115_v1`、`outputs/equipment_flex_audit_20261115_v1`。

```powershell
python -m unittest rnj_wzzt_dr.experiments.equipment_flex_20261115.test_resources rnj_wzzt_dr.experiments.equipment_flex_20261115.test_domain rnj_wzzt_dr.experiments.equipment_flex_20261115.test_run
python -m rnj_wzzt_dr.experiments.equipment_flex_20261115.audit --self-test
python -m rnj_wzzt_dr.experiments.equipment_flex_20261115.report --self-test
python -m rnj_wzzt_dr.experiments.equipment_flex_20261115.run --out outputs/equipment_flex_smoke_20261115_v1 --smoke
python -m rnj_wzzt_dr.experiments.equipment_flex_20261115.audit --run outputs/equipment_flex_smoke_20261115_v1 --out outputs/equipment_flex_smoke_audit_20261115_v2
python -m rnj_wzzt_dr.experiments.equipment_flex_20261115.run --out outputs/equipment_flex_execution_20261115_v1 --preflight outputs/equipment_flex_smoke_audit_20261115_v2/report.json --seconds 1200 --workers 3
python -m rnj_wzzt_dr.experiments.equipment_flex_20261115.audit --run outputs/equipment_flex_execution_20261115_v1 --out outputs/equipment_flex_audit_20261115_v1
python -m rnj_wzzt_dr.experiments.equipment_flex_20261115.report --audit outputs/equipment_flex_audit_20261115_v1/report.json --out outputs/equipment_flex_report_20261115_v2
```

以上为实际运行路径索引；输出目录已存在，复现时使用新目录并相应修改引用，不能覆盖旧结果。第一份smoke审计仅有缓存切片接线错误，已保留而没有改原执行数据。
