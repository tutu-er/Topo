# paper15：指数概率条件的新一日DR规划

本文件夹包含规划、冻结、独立随机核验、原AC运行和最终选模评价。`run.py`为主入口，`ac.py`接入原AC；`vendor/`、`vendor_ac/`保存逐字复用的原实现及来源清单。完整本机数值输入在`inputs/`，该目录和运行轨迹不上传Git。

本轮研究对象是**完整历史RNJ＋WZZT产生的两个冻结候选之间的最终选择**。两候选及所有R/X固定，不再拟合、不新增W搜索；其输出块族沿用原0.16收缩结果。使用全部288槽训练、288槽验证，15节点全部60个资源单元（4类×15节点），下一日96槽、每槽15分钟、每小时报价。没有缩短历史、提高设备能力、改变线路或降低噪声。

## 运行

将整个文件夹（包括本机的`inputs/`）复制到任意位置，再运行：

```text
python -m pip install -r requirements.txt
python -B run.py all --out runs/run1
```

分阶段执行可明确先冻结价格、后开放真实运行数据：

```text
python -B run.py plan --out runs/plan1
python -B run.py sample --plan runs/plan1 --out runs/sample1
python -B run.py ac --plan runs/plan1 --out runs/ac1
```

使用新输出目录保存每次尝试。`plan`保留输入/源码哈希、费用上下界、概率充分界和`quotes.npz`，以`policy_freeze.json`冻结。`sample`核验规划工作分布；`ac`用新的完整随机路径评价实际效果，失败记录保留。已有完成记录直接读取；运行锁用于阻止重复执行同一AC批。

`quotes.npz`中的`optimized_hourly_requests`为60×24归一化小时激励请求，`optimized/zero/rotation`为60×96名义kW交付。每单位奖励由原`problem.unit_reward(u)`按方向对应的成本×|u|/可用功率计算，费用为原预付二次合同；本轮三个方案的请求、交付和奖励单价均为0。

Gurobi仅在需要正费用优化时调用。若零动作满足条件且设备费用非负，最小费用0可直接证明，此时不伪造一次求解器运行。完整设备图包括小时请求、逐槽功率饱和、效率、能量与截止时段；不把它替换成自由逐槽功率。

## 规划条件及其含义

模型0为当前候选，模型1为备选候选。它们对同一功率向量的电压响应为$H_0P,H_1P$。功率预测及模型1参考电压为

\[
P\sim N(\widehat p+Bu,\Sigma_P),\qquad
V=H_1P+\varepsilon,\quad \varepsilon\sim N(0,\Sigma_V),\quad P\perp\varepsilon.
\]

$u$为名义资源调节量，正号表示增加用电。两项协方差固定，不随报价变化。$\Sigma_P$描述背景预测误差；实际乘性响应和饱和另在AC阶段核验，未被这个固定协方差条件覆盖。$\Sigma_V$仅描述给定功率后剩余的电压误差，避免重复加入功率误差。

对全部观测求标准化平方残差总和

\[
S_k=(V-H_kP)^\top\Sigma_V^{-1}(V-H_kP),\quad k=0,1.
\]

保留完整历史时，新日需要满足的条件为

\[
L=S_0-\kappa S_1>\delta,
\quad\kappa=1.03,\qquad
\delta=\delta_{\rm all}-(S_0^h-\kappa S_1^h),\quad\delta_{\rm all}=0.
\]

规划要求$\Pr_1(L>\delta)\ge0.90$；下标1表示模型1参考及声明的高斯误差成立。这是**平方损失改善事件的条件概率**。

固定$\lambda=1/(4\kappa)$，使用

\[
\Pr_1(L\le\delta)\le e^{\lambda\delta}E_1[e^{-\lambda L}].
\]

两次高斯积分给出固定矩阵和门槛。令

\[
d(u)=(H_1-H_0)(\widehat p+Bu),\quad
\Sigma_D=(H_1-H_0)\Sigma_P(H_1-H_0)^\top,
\]

\[
G=\Sigma_V+\frac{\Sigma_D}{2(\kappa+1)},\qquad
W=\frac{G^{-1}}{4(\kappa+1)},
\]

\[
h=\frac{\delta}{4\kappa}+\log\frac1{1-\beta}
-\frac12\log\frac{|G|}{|\Sigma_V|}
-\frac m2\log\frac{\kappa+1}{2\kappa},\qquad\beta=0.90.
\]

$m$是新日电压观测总数，本例为96×15=1440。优化问题为

\[
\min_{u\in\mathcal U}K(u)\quad\text{s.t.}\quad d(u)^\top Wd(u)\ge h.
\]

$K$和$\mathcal U$沿用原设备费用及可行域；不设额外总费用上限。条件成立时给出充分概率下界，保守性来自上面的指数不等式。逐式代码和理论说明在`vendor/exponential_dr.py`。原精确设备图及Gurobi全局界在`vendor/solver.py`。

为减少矩阵大小，`make_gate`作精确换元：$P'=(H_1-H_0)P/sd$，$V'=(V-H_0P)/sd$，于是$H'_0=0,H'_1=I,\Sigma'_V=I$。输入的`difference_spatial_factor`生成的是$P'$的协方差，已有标准差除法，不能再当原始P/Q协方差使用。

## 两种评价分别报告

1. **平方条件**：用实际量测及冻结H/R/X重算$L>\delta$，检查规划的90%激励目标。
2. **原最终选模**：仍以MAE评价相同观测，将完整历史与新日各计一次；保留共同3%带、支撑数优先及原冻结输出。这里新日权重为1，遵从此前去除权重2的要求。实际错漏按选中的最终块族统计。

平方条件与MAE排除事件有不同的损失口径；两者的成功计数分别列出。仅平方条件通过不算最终拓扑恢复。模型1作为规划参考也不等于读取真实拓扑。

工作模型32768条Gaussian路径；实际32条新的配对整段响应/量测路径，用原AC及原量测噪声（电压0.0005、P/Q0.005）。每条物理路径比较优化、无DR、同费轮换；逐字相同的合同只运行一次AC并明确列出别名。自然负荷日为此前已见的同一个日，不把这些随机路径称为32个独立日期。真支持仅在全部选择写入`predictions_freeze.json`后读取。

## 算例与拓扑接口

顶层`case.json`索引相对路径；规划只读取下面前四类输入：

| 文件 | 核心字段 | 单位与形状 |
|---|---|---|
| `inputs/pricing.json` | 资源/节点ID、设备与原MAE选模配置 | 全部显式；本例60资源、15节点 |
| `inputs/pricing.npz` | 原功率上下限、费用、效率、能量、reset/terminal、未来偿还边界 | 资源×96槽；小时请求60×24 |
| `inputs/models.npz` | `H0,H1` | 15×30；功率列顺序为15个P后15个Q，响应单位为$10^{-6}$p.u.$^2$ |
| 同上 | `forecast_power`、`history_training_power`、`history_power` | p.u.；96×30、288×30、288×30；基准200kW/kvar |
| 同上 | `history_drop`、`likelihood_sd` | $10^{-6}$p.u.$^2$；288×15、96×15 |
| 同上 | `difference_spatial_factor`、`power_time_ar` | 标准化候选差的因素96×15×30及AR系数0.8 |
| 同上 | `resource_node_index`、`resource_q_over_p`、P/Q基准 | 零起始节点索引、kvar/kW；60资源 |
| `inputs/candidates.json` | 两模型ID、原始支撑数、冻结输出块族 | 节点标签与规划接口一致，来源仅历史辨识 |
| `inputs/devices.json` | 原分配方案和行为配置 | 原Devices.draw使用；含整段/节点/时间相关因素 |
| `inputs/physical.json/.npz` | 原真实新日P/Q、根电压、网络基准和噪声设置 | **仅评价阶段**，冻结报价后才读取 |

接入另一组历史辨识结果时，导出两个冻结的`H0/H1`、历史观测、候选输出块族及日前预测，按相同单位准备上述接口。功率预测可以替换；剩余电压误差必须与其定义一致。实际拓扑和AC由`ac.open_runtime / evaluate / finalize`三个入口提供。当前AC适配器锁定paper15原网络；更换物理网络应接入其原AC后端，规划不读取物理拓扑文件。

仅更新同一网络的新日时，更新`forecast_power`及相应预测误差输入，保留历史拟合H/R/X和完整历史；在新目录重新规划并冻结，然后由评价方提供该日`physical.npz`。此简化选模适配器针对当前两候选、当前候选支撑数较少的争议；更多候选或另一种最终规则应提供相应评分接口。`vendor/probability.py`是原设备/选模类的导入依赖，本入口未调用旧L1概率积分。

`vendor_ac/rnj_wzzt/data/`保存本例原拓扑构造，`models/ac_powerflow.py`为原AC方程；`vendor_manifest.json`记录原代码和抽取函数的一致性。设备基线已计入自然总负荷，执行时只添加交付增量一次。

## 结果与恢复

本轮数值与结论见`REPORT.md`和`RESULTS.json`。全部原始观测、输入、冻结、失败与断点留在本机恢复目录；Git只包含代码、配置、源清单及无轨迹汇总。复制本机完整文件夹可从头运行，复制Git目录时需另提供`inputs/`数值接口。
