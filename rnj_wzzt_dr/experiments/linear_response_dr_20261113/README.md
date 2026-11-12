# 连续设备与线性DR检查

本轮回应三个问题：原先的失败是否只是搜索困难；储能连续松弛和纯线性响应能否给出更容易求解的方案；距离明显增加为何只带来小幅后验变化。辨识器、真实算例候选、拟合参数、历史与共同误差尺度保持冻结。人工构造例单列，不替换真实记录。

## 两种设备模型

`storage_relaxation`：保留原小时可用功率、效率、能量上下限、未来可行区间与截止要求，取消固定充放电方向。充电c、放电d允许同时非零，满足`c/cmax+d/dmax<=1`，能量按`eta*c-d/eta`更新。费用仍对净交付`c-d`计。它是原“不触发名义截断的小时计划”模型的连续松弛；原控制器还允许槽内截断，因此不把这一松弛当作完整原控制域的外包络。记录同时充放电及原执行器重放结果。

`linear_response`：保留同样的小时功率、可用窗与净交付费用，删除跨时段能量要求。直接规定功率变化随计划线性变化。这是一个简化的DR响应模型，未经原控制器核验的方案不作为储能可执行报价。

两种模型的设备更新都是凸二次规划。完整规划仍含“预测距离平方达到下限”的非凸二次约束，复用上一轮的球外投影ADMM。另一条更简单的路径是在零点用全局线性下界求一个凸QP提议，再通过原控制器重放；只接受重放后仍满足原证据要求的方案。

## 范围与顺序

1. 先检查旧实验预先选出的6个输入、12个条件目标，每个目标分别求两种设备模型的零点线性下界QP，预算6秒（保存实际含建模耗时）。所有目标保留，不在新未来样本上筛选。
2. 对每个竞争者计算忽略能量耦合后的功率盒上界。上界低于所需增量才支持对应原域不可行；线性下界QP不可行和求解超时只报告其本身。
3. 按原执行器重放后的最弱增量选择两个输入，保留其原两个目标方向。每目标每设备模型再给ADMM30秒，记录初值来源、初值生成时间、全部超时与原执行器可行中间点。先前筛选时间单列。
4. 所有报价冻结后，每目标2048条新的条件模型路径，比较无DR、线性QP重放、ADMM重放、同费轮换。完整候选库评分；目标真假只在离线评价时加入。此次不新开AC路径。
5. 复算上轮已存AC观测，将对数赔率变化精确拆为预测距离和残差交叉项，再检查背景预测与实际响应造成的差别。这是旧数据诊断，不是新的独立测试。

名义增强量仍为`log(2)`。费用使用原合成合同单位，公开表归一化。原始观测、矩阵与轨迹保存在忽略的`outputs/linear_response_dr_20261113_v1/`。

## 距离与后验

为避免旧代码中J的多重含义，这里明确记

\[
J_{dist}(P)=\|(H_t-H_j)P\|_F^2/(2\sigma^2).
\]

它表示以公共误差尺度衡量的预测距离。给定目标生成模型、零均值剩余误差，DR增加的期望对数赔率就是其期望增量。实际观测满足精确恒等式

\[
L_j=L_j^h+J_{dist}(P)+\langle Y-H_tP,(H_t-H_j)P\rangle/\sigma^2.
\]

所以同时报告距离、残差项、对数赔率、平均后验、达标率和费用。旧指数充分界使用`d'Wd`，早期MAE距离使用平均绝对预测差；它们的数值单位与本式不同。

## 人工测试

两个响应矩阵为`H_t=I`和`H_j=.98I+.01*11'`，分别对应星形结构和多一条正共享支路的结构。自然功率两节点平衡时预测相同；线性DR使两节点反向变化，从而产生区分证据。参数人为设置，未改动真实拟合。增量分别要求`log(2),2,4,8`，全部保留；费用最优值可独立手算，方便核验ADMM。该例只说明算法和概率数值关系。

## 入口

```powershell
$py='D:/apps/miniconda3/envs/Topo/python.exe'
& $py -X utf8 -B -m unittest rnj_wzzt_dr.experiments.linear_response_dr_20261113.test_models -v
& $py -X utf8 -B -m rnj_wzzt_dr.experiments.linear_response_dr_20261113.study --stage prepare --output outputs/a_new_linear_study
& $py -X utf8 -B -m rnj_wzzt_dr.experiments.linear_response_dr_20261113.study --stage screen --output outputs/a_new_linear_study
& $py -X utf8 -B -m rnj_wzzt_dr.experiments.linear_response_dr_20261113.pricing --stage plan --run outputs/a_new_linear_study
& $py -X utf8 -B -m rnj_wzzt_dr.experiments.linear_response_dr_20261113.pricing --stage confirm --run outputs/a_new_linear_study
& $py -X utf8 -B -m rnj_wzzt_dr.experiments.linear_response_dr_20261113.synthetic --output outputs/a_new_synthetic_pair
```

代码拒绝覆盖已完成阶段。实际结果以完成标记和结果报告为准，不因单个QP成功就宣称完整非凸问题全局最优。
