# 完整 AC 规划失败与可执行回退的区别

独立评价中的 N4 四条历史使 projection_fixed/mobile 各有四条规划失败：index 8、11（H8），12、15（H96）。两方法合计八个失败槽仍计入分母。index 8、11、15 的移动方法复用固定方法的同一失败缓存；index 12 触发墙钟门，失败 profile 不缓存，移动方法又计算了一次同曲线。独立历史仍只有四条。

五个实际计算 profile 共记录 80 个参考—解释槽，其中75项已尝试、70项成功。5次 joint 达到 nfev=250（对应4个独立参考—解释问题，index12重复一次）；另5槽在60秒软墙钟门后未尝试。两类原因分开记录。

|index/H/真值|失败的参考→解释|nfev|status|optimality|固定线路源成功|解释物理可用|原因|
|---|---|---:|---:|---:|---|---|---|
|8/H8/star|clade_01_23→clade_23|250|0|3.02121|True|True|The maximum number of function evaluations is exceeded.|
|11/H8/clade_01_23|clade_01→clade_01_23|250|0|6.19039|True|True|The maximum number of function evaluations is exceeded.|
|12/H96/star|clade_01→clade_01_23|250|0|31.4901|True|True|The maximum number of function evaluations is exceeded.|
|12/H96/star|clade_01_23→clade_01|None|None|—|None|None|joint predictive budget exhausted|
|12/H96/star|clade_01_23→clade_01_23|None|None|—|None|None|joint predictive budget exhausted|
|12/H96/star|clade_01_23→clade_23|None|None|—|None|None|joint predictive budget exhausted|
|12/H96/star|clade_01→clade_01_23|250|0|31.4901|True|True|The maximum number of function evaluations is exceeded.|
|12/H96/star|clade_01_23→clade_01_23|None|None|—|None|None|joint predictive budget exhausted|
|12/H96/star|clade_01_23→clade_23|None|None|—|None|None|joint predictive budget exhausted|
|15/H96/clade_01_23|star→clade_23|250|0|9.01768|True|True|The maximum number of function evaluations is exceeded.|

所有75项已尝试拟合的固定线路源拟合均成功，保存解释满足电压、源界与线路域。五次失败 joint 的 status=0、nfev=250，最优性指标分别约3.02、6.19、31.49（两次）、9.02；程序保留其物理可行解释，但优化 success=False，梯度可靠标记随之为 false。全 K×K 预测评分要求每项成功，因而不能得到完整规划评分。index12还出现5个明确记录为未尝试的槽；它们没有物理状态，不能写成物理失败。

这些方法没有进入 MILP 新提案步骤（steps=0）。固定方法执行 square_parent，移动方法执行 fixed_parent；逐时功率与对应已保存父计划完全一致。八个回退动作均满足操作/SOC核查，随后对实际计量的 32 个候选 joint 拟合均成功。失败的是规划时的某个预测解释计算，物理操作和最终实际计量拟合是另两项检查。

JSON 保存每个失败的 optimizer message/nfev/optimality、重启摘要、物理可用标记、可行证据损失、profile来源与字节SHA，以及八个方法槽的回退和最终拟合状态。未追加拟合、延长迭代或修改策略。
