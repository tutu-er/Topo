# 保留有符号损失差的QCQP补丁

修正的是参考预测，不改变已拟合候选的R/X。原方案把备选候选本身当作准确电压参考，因此可以把设计目标化成“放大两个预测的距离”。如果该候选在另一处也有结构错误，这个方向就可能反过来支持原候选。

补丁使用一个独立选择的历史电压预测模型`Hbar`，允许它与两个待比较输出都不同：

\[
 P=\widehat p+Bu+\xi_P,\qquad V=\bar H P+b+e.
\]

参数、均值和联合高斯误差协方差在报价前冻结。候选`H0/H1`仍用于计算最终MSE；`Hbar`只预测观测，不被宣布为真实拓扑。价格优化仍要求指定损失差跨过原严格MSE门槛。

## 简单的主线

1. 两个候选残差共享同一个电压观测，直接展开平方差，保留正负：`L=w'A w+2a'w+c`。
2. 将随机功率与剩余电压误差统一写为`w=mean+B*u+U*xi`，其中`xi`是标准高斯因素。
3. 固定一个使`J=I+2*lambda*U'AU`正定的正`lambda`，精确计算`T(u)=-log E exp(-lambda*L)`。它仍是固定二次函数，可能不定。
4. 最小化原费用，要求`T(u)>=lambda*delta+log(1/(1-beta))`。本次`beta=.9`。

这里`delta=(历史验证cell数+新日cell数)*1e-12/K−历史SSE差`，来自实际严格MSE选模。完整历史评分保留，新日权重1，参数在实际评价时不重估。该保证条件于所声明的电压预测与误差法则，不是未知拓扑真实性概率。

详细背景、符号、配方和证明见[两页PDF](theory/reference_repair.pdf)及[可编辑LaTeX](theory/main.tex)。积分是解析预计算，不在优化中抽样。`Q`不定时不能再对整项作切线并称其为下界。

## 文件与接口

| 文件 | 输入、输出与作用 |
|---|---|
| `quadratic.py` | 通用高斯二次式、仿射均值和固定协方差因子→QCQP的Q/q/const/h及概率充分界。只需NumPy，可单独运行确定性核验。 |
| `study.py` | 读取原历史候选/资源→参考选择→有界QCQP→冻结→原随机交付与AC→严格MSE选模。复用原设备图和物理函数。 |
| `local_candidates.py` | 从两个历史候选生成全部单争议区替换及合法并集，不读真值、不拟合。它是单列的候选覆盖诊断，不是DR创新或完整搜索。 |
| `POSITIVE_CONTROL.json` | 一个已知参考的两维正控：原方案调到相反方向，补丁调到正确方向。它是公式/求解验证，不是配网恢复率。 |

`build_quadratic_gate(A,a,c,mean,control,cov_factor,delta,beta,lambda_)`要求积分矩阵正定；失败就明确拒绝，不修改协方差来制造可行性。输出对象的`evaluate(u)`与`-log_mgf(u)`应一致。`bound(u)`报告声明模型下的充分概率下界。

通用接口完整支持非零截距。当前两个物理案例使用原始无截距响应模型；稀疏AR案例入口只实现这个零截距分支，发现外部输入非零截距会拒绝，避免静默忽略。

## 本轮实际测试约定

沿用已有Pengwah原噪声对照和Soumalas已有压力例。96训练、288验证全留；旧QP输出参数、.24收缩、原资源能力/费用/到离站/能量、原AC/PQ/V量测均保留。

参考选择只用已有历史：比较原未作输出收缩的路径模型，按前192验证点MSE选一个，末96点作开发核验。两例都选中既有`model_004`；未重新拟合它，也未使用实际新日或真结构选参考。新拟合的无结构ridge曾在该开发核验失败，保留结果，未被包装成更准确的参考。

功率误差尺度取四个已观测日相对日均预测的RMS；节点内P/Q共享高斯因素，保留原时间AR(.7)。剩余电压误差尺度与上一轮相同，避免通过降低误差制造收益。输入P表示未来量测功率，其预测误差与条件剩余电压误差各计一次。实际乘性响应、相关偏差和饱和仍由原设备规则独立核验。

完整AR协方差通过稀疏精度矩阵表示，和完整高斯积分等价，没有改成逐时独立。辅助变量由线性等式唯一确定；其引入只为减少二次表达式规模，不改变QCQP。31项通用推导核验及447项稀疏/梯度/边界核验通过。

本次另统一生成两个端点之间的单区替换及合法并集，使用原`PoolQP`对全部连续权重从零共同拟合，再按原.24输出规则处理并训练QP。它们只用于同观测的额外选模对照；主DR报价仍针对原冻结输出库。若补候选本身已改善无DR结果，应单独计为候选库收益。

未找到概率可行报价时返回“不购买/未解决”状态。零动作并未因此被声称满足90%条件；也不会像上一轮那样付费执行一个未达标备用方向。

`nominal_solver.py`是另外的有界计算修复：直接优化满足所有原约束的名义小时计划，实际执行仍用原控制器。这个内域可减少混合整数图规模；其下界只适用于内域，不能冒充原全部控制器域的费用下界。

## 运行

在仓库Python环境中，用新的私有输出目录：

```powershell
python -B research_experiments/wzzt_dr_selection/dr_pricing/mse_reference_repair/quadratic.py
python -B research_experiments/wzzt_dr_selection/dr_pricing/mse_reference_repair/study.py prepare --out OUT
python -B research_experiments/wzzt_dr_selection/dr_pricing/mse_reference_repair/study.py plan --out OUT
python -B research_experiments/wzzt_dr_selection/dr_pricing/mse_reference_repair/study.py actual --out OUT
```

依赖原NumPy/SciPy/Pandas/NetworkX/Gurobi环境及已有结果目录。准备阶段保存输入、源文件、模型选择和有限拟合记录；报价冻结后才读新物理日。实际轨迹保留在私有目录，同报价的政策标签共用一次实际运行，完整失败/未决和别名分母均保留。此研究入口不修改生产默认。
