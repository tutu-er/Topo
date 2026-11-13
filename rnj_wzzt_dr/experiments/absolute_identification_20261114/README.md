# 绝对拓扑恢复与配网灵活性服务

本目录隔离保存2026-11-14的推导、实现和扩大试验。生产默认未改动，原辨识器及旧结果保持不变。

- `STUDY_LOG.md`：逐轮目标、发现、修正和锁定决定，包括失败批次。
- `ANALYSIS.md`：800日与1408日完整结果、正反例、模型失配及继续迭代的依据。
- `reports/REPEATED_CASES.md`：全部44份重复历史的恢复次数与费用。
- `NATIVE_FAILURE_ANALYSIS.md`、`METER_MODEL_AUDIT.md`：逐事件归因、P/Q外推问题及表计评分对照。
- `BIAS_DIAGNOSTIC.md`、`ITERATION_DESIGN.md`：只用历史的偏差诊断、第四轮P/Q限制与固定评分障碍。
- `ITERATION4_LOCK.md`、`R4_PROTOCOL_REVIEW.md`、`reports/iteration4_pq/`：新增704日迭代的锁定设置、核查和全部结果。
- `INVENTORY.md`：400个冻结历史库、真拓扑覆盖和物理历史分组。
- `THEORY_NEXT.md`：绝对恢复条件、对称候选保护、模型偏差与QCQP形式。
- `data.py`：原完整候选库适配，不重新拟合。
- `solver.py`：对称绝对恢复；`solver_v2.py`：加入乘性均值校正及不计预测偏差改善的版本。
- `commonseeds.py`：相同可行资源方向组成的初值集合，以及原增量QCQP的统一初值适配。
- `driver.py`：先封存采购，再在真实AC／真实线路线性潮流下执行；同费轮换、未决零采购和全部失败均保留。
- `linear_truth.py`、`AC_ABLATION.md`：真实物理网络的线性潮流消融，未用目标候选生成真值。
- `diagnostics.py`：把事后实测证据变化分解为预测距离、冻结参数偏差和表计误差。
- `report.py`、`reports/`：完整分组结果、费用、初始化和求解边界。详细JSON在本机生成，不加入Git。

原始P/Q/V、预测、拟合矩阵和逐日报价在忽略的`outputs/absolute_*`目录；每批含协议、来源哈希、源代码副本、策略封存及完成记录。`outputs/astra_inventory_20261114`保存库存及固定开发／历史保留名单。

主结果使用原验证历史的全部32点加新日全部96点、原完整冻结参数库、原严格MSE选择规则。64点训练数据仍只用于原拟合，不混称为选模的历史评分。代表模型归一化权重和拒绝规则另列。AC与线性结果共享随机路径，不能计为两倍独立样本。

运行例：

```powershell
D:/apps/miniconda3/envs/Topo/python.exe -m rnj_wzzt_dr.experiments.absolute_identification_20261114.driver --out outputs/NEW_RUN --inputs outputs/astra_inventory_20261114/development32.json --days 2 --seconds 12 --workers 4 --common-seeds --replicate-base NEW_SEED --modes old_gamma2,symmetric_guard,multiplicative_absolute,capped_absolute
D:/apps/miniconda3/envs/Topo/python.exe -m rnj_wzzt_dr.experiments.absolute_identification_20261114.report --run outputs/NEW_RUN --destination rnj_wzzt_dr/experiments/absolute_identification_20261114/reports/NEW_REPORT
```

800日、1408日和新增704日均已执行并独立核验，报告见对应reports子目录。固定库缺少真拓扑的输入保持在报告中；拒绝输出、零采购和完整恢复分别计数。新增P/Q限制降低费用也损失恢复收益，留作隔离实验，不替换主方法。已打开的确认数据只作诊断材料；原始私有结果与失败记录均保留。
