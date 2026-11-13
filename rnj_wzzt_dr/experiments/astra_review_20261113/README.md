# 直接恢复概率与冻结模型偏差：Astra复核

本批将灵活性采购的理论目标直接写为完整拓扑恢复概率。沿用原候选、原评分、原用户设备、价格和AC模拟；新增两个均值假设的采购规则，并用独立新日检验。没有重新拟合辨识器。

阅读顺序：

1. [理论](THEORY.md)：每个竞争者引入一个非负变量即可将正态机会约束保持为QCQP；拒绝规则等价于最小SSE与固定参考B比较。
2. [旧32日诊断](DIAGNOSTICS.md)：名义高估主要来自冻结参数偏差；七种历史均值模型在已打开的旧数据上比较，选定验证段常数加乘性修正。
3. [新48日结果](results/REPORT.md)：弱证据32日、容易8日、缺真候选8日，各自报告，不混成总体识别率。七个配对策略保留费用、失败与下降。
4. [求解时间核查](PRICING.md)：固定两份旧预测，比较3、12、60秒预算的可行费用和有效下界。

源码分工：`solver.py`构建完整原参数库的直接恢复QCQP；`experiment.py`锁协议、冻结方案并执行原AC；`diagnostics.py`仅解释旧数据；`report.py`读取新封存结果。所有原始H/P/Q/V、逐日报价与采购轨迹均在忽略的outputs目录。

新规则预锁90%的模型内恢复要求，给拒绝分配0.02、竞争者合计0.08。共同评分尺度保持原值；修正模型每节点拟合两个历史残差均值参数，方差取原尺度与自由度修正后残差方差的较大值。该估计没有自动获得真实AC的概率校准保证。

主拒绝参考为总残差数乘以2倍已知表计粗略噪声量级，1.25倍和4倍作预锁敏感性对照；三者都不是由新日结果选择。费用为原示例激活补偿，完整市场采购费用仍需另外校准。

每报价12秒，含资源LP方向初值与线搜索；全局非凸求解器随后搜索，原模型可行性另外复算。未决与失败使用明确的零动作回退。2个工作进程，随机序号900000+1000*历史条件序号+新日序号；全部方案在响应和执行电压打开前保存哈希。

```powershell
& D:/apps/miniconda3/envs/Topo/python.exe -B -m unittest rnj_wzzt_dr.experiments.astra_review_20261113.test_experiment
& D:/apps/miniconda3/envs/Topo/python.exe -B -m rnj_wzzt_dr.experiments.astra_review_20261113.test_solver --out outputs/astra_solver_checks_20261113
& D:/apps/miniconda3/envs/Topo/python.exe -B rnj_wzzt_dr/experiments/astra_review_20261113/theory_checks.py
& D:/apps/miniconda3/envs/Topo/python.exe -B -m rnj_wzzt_dr.experiments.astra_review_20261113.experiment prepare --out outputs/astra_confirmation_20261113_v1
& D:/apps/miniconda3/envs/Topo/python.exe -B -m rnj_wzzt_dr.experiments.astra_review_20261113.experiment run --out outputs/astra_confirmation_20261113_v1
```

已完成目录由完成标记保护，不重新当成独立试验；锁定后源码变化会阻止继续执行。结果仅外推到所选三个冻结历史条件及其原新日生成法则。
