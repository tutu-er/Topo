# 15分钟滚动灵活性

本目录复用192点训练、96点验证得到的冻结候选，在第4天执行96轮调度。每轮规划剩余全天，只执行15分钟，继承实际设备状态和每个候选的累计平方误差。保留三种概率门槛，并加入履约服务、采购反馈和全天费用账本。

模型说明见[MODEL.md](MODEL.md)，锁定的输入、预算和验收范围见[PROTOCOL.md](PROTOCOL.md)，完成结果见[RESULTS.md](RESULTS.md)。私人P/Q/V、拟合矩阵和逐轮轨迹只保存在忽略提交的`outputs`中。

## 运行

从仓库根目录执行。以下使用新输出目录；不要覆盖已封存批次。

```powershell
$py = 'D:/apps/miniconda3/envs/Topo/python.exe'
$taskOut = 'outputs/rolling15_example_v1'

& $py -B -m rnj_wzzt_dr.experiments.rolling_flexibility_20261115.run --out $taskOut --prepare-only
& $py -B -m rnj_wzzt_dr.experiments.rolling_flexibility_20261115.run --out $taskOut --resume
```

默认复用`outputs/history288_pq_noise_execution_20261115_v1`，固定Sou h00、Sou h03和Peng h00的P/Q噪声0.005设置。每输入1条配对路径，比较无额外采购及`normal_fixed`、`mgf_fixed`、`mgf_forecast`，共12条轨迹、1152轮。每轨迹规划预算1200秒，单轮最多30秒，按剩余轮数分配；评价侧AC及输出耗时另列。这是两个已知物理背景的开发测试。

短接线可在准备时使用`--input-id`、`--modes normal_fixed`、`--day-seconds`和`--round-seconds`明确缩小范围，具体设置会写入协议。`--resume`读取已封存设置，复用有完成记录的轨迹；未完成轨迹须先检查断点，不自动重置预算。

## 测试

```powershell
& $py -B -m pytest rnj_wzzt_dr/experiments/rolling_flexibility_20261115 -q
```

测试覆盖累计评分、目标变化、2小时等于8轮、设备终态、响应两端可达性、确定性服务尾、费用和非零QCQP。非零解析小例检验求解器能够找到已知动作及改进初值；它与真实设备闭环效果分开报告。

## 独立审计并生成报告

```powershell
& $py -B -m rnj_wzzt_dr.experiments.rolling_flexibility_20261115.audit --run $taskOut --out outputs/rolling15_example_audit_v1
& $py -B -m rnj_wzzt_dr.experiments.rolling_flexibility_20261115.report --audit outputs/rolling15_example_audit_v1/report.json --out outputs/rolling15_example_report_v1
```

审计输出`report.json`和`records.json`，保留全部计划轨迹与轮次、失败、报价状态、设备状态、累计分数、费用、全网电气结果和数值证书检查。聚合入口进一步生成中文报告、CSV、JSON及逐轮权重/调用图；固定beta下的确认与未决在报告层判定。

已完成`outputs/rolling15_development_20261115_v1`的12条轨迹、1152轮，独立审计v3全部通过，汇总位于`outputs/rolling15_development_report_20261115_v3`。本批仍全部零调用：每策略2/3第一名内部正确、2正确确认和1未决，完整支持正确1/3。它支持闭环接线核验，没有新增非零采购收益证据。更早192轮短接线单独保留，不合并为独立样本。

## 代码入口

| 文件 | 职责 |
|---|---|
| `state.py` / `evidence.py` | 冻结参数、累计评分、当前目标和三种采购门槛 |
| `feedback.py` | 逐轮偏差监测、谨慎限幅、暂停及恢复 |
| `domain.py` | 15分钟设备状态、原任务终态、可达履约尾 |
| `solve.py` | 请求调用费与新增容量费优化、原式验收和求解证书 |
| `execution.py` | 当前点AC、表计误差、交付遥测及费用账本 |
| `run.py` / `audit.py` | 计划先封存后执行；独立逐轮复算 |
