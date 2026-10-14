# EV 实际拒绝动作的独立校准：safe v2

状态：2026-10-14，方案和外围实现待 root 审查；创建本协议时新正式 AC 尚未启动。原 v1 的 39 个校准历史已完成，其中 982009 的 strong/uniform、v2g_mobile 有三个竞争候选在 max_nfev=200 下未收敛。原协议把任何必要失败记为无限，因此原阈值保持无限、评价保持 0/48，不改写记录。

本修订来自上述旧校准的计算失败诊断。旧 39 个历史只作为方法开发诊断；新阈值须由新的独立 39 个历史计算，不复用旧分数、旧阈值或挑选旧子样本。动作配置、完整 AC 拟合和科学 21 个源码文件保持不变，修改的是外围拒绝分数和其统计解释。

## 1. 主问题与失败处理

仍比较九方法 none、constant_fixed、quality_charge、charge_fixed、charge_random、charge_mobile、v2g_fixed、v2g_mobile、charge_staff。主要比较固定点优化曲线相对恒充、移动选址相对固定点；V2G 相对纯充、人员接入窗口相对普通移动作次要描述。沿用 v1 的四时段、6 kW、60 kWh 电池、固定终端 26.85 kWh、效率 0.95、电池服务要求和外生人员窗口。

部署时，必要历史拟合、物理评价或联合拟合失败的方法保留所有已知候选，并继续标为 compute_failure。其原始状态、部分拟合值、耗时、失败候选和失败次数均保存。已激活但设计失败后的安全回退仍独立计 design_failure；若实际回退的物理/联合拟合完成，可正常评价该回退的真实结果。未激活单列。

**保留集合只使用损失、候选名称、阈值和实际物理/拟合状态。** 仿真真标签和覆盖审查不参与部署保留集合。覆盖审查放在评价侧：漏真或未知覆盖使校准分数无限并阻止正式收益比较；保留已知候选无法修复遗漏的真结构。

## 2. 分数的明确含义

m 表示九个方法之一，a 表示一个已知候选，a_* 表示仿真真候选。J_m(a) 是历史与新增计量完整 AC 联合拟合的白化平方残差和。定义仅用于实际拒绝规则的调整损失：

\[
 J_m^{safe}(a)=
 \begin{cases}
 J_m(a), & \text{必要物理与拟合均完成},\\
 0, & \text{必要步骤失败，部署保留全部已知候选}.
 \end{cases}
\]

失败时的 0 表示没有实施拒绝动作；它不是实际拟合误差、原始最大似然差或一次成功辨识。该方法的原始 J 值不覆盖成 0，原始 trace 原样保留。

在已知 a_* 属于候选集合时，方法拒绝分数为

\[
 S_m=J_m^{safe}(a_*)-\min_a J_m^{safe}(a),
 \qquad S=\max_{m=1,\ldots,9}S_m.
\]

因此失败但覆盖已知的方法 S_m=0；成功方法仍使用原始完整 AC 的真候选损失差。若真候选缺失、覆盖未知或同一历史的真标签不一致，设 S=+infinity。本分数校准的是实际拒绝行为，不校准计算完成率。

所有成功/失败方法使用同一个非负阈值 kappa：

\[
 \widehat{\mathcal C}_m
 =\{a:J_m^{safe}(a)-\min_bJ_m^{safe}(b)\le\kappa\}.
\]

成功时恰为原平方误差差筛选，失败时所有候选的差都是 0，因此恰好保留全部候选。实际代码可继续用显式失败分支保留全部；独立测试检查它与调整损失表示完全一致。即使所有方法都失败、分数都是 0，也只能得到失败和无排错结果，不能据此宣称算法表现好。

## 3. 为什么误拒校准仍成立

在声明的三末端总体、真候选被包含、算法与生成分布冻结的条件下，任一方法拒绝真候选必有

\[
 J_m^{safe}(a_*)-\min_aJ_m^{safe}(a)>\kappa.
\]

所以“九方法中至少一个误拒”的事件包含在 {S>kappa} 中。失败方法保留全部，已知覆盖时它没有误拒事件，也无需用无限分数惩罚这一次没有拒绝的动作。算法完成率与辨识能力仍通过完整分母中的 compute_failure、唯一正确和安全未决分别评价。

令独立历史的 S 分布函数为 F，固定 alpha=0.1、delta=0.05、n=39、k=39。新校准阈值为

\[
 \kappa=\max_{i=1}^{39}S_i.
\]

给定校准集，下一同分布独立历史的超阈值率是 R(kappa)=1-F(kappa)。若 R(kappa)>0.1，则 39 个分数全部落在 {s:F(s)<0.9}；该集合概率质量最多 0.9。独立性给出

\[
 P_{cal}\{R(\kappa)>0.1\}
 \le0.9^{39}=0.016423203268\ldots<0.05.
\]

等号保留，因此证明允许并列分数。此处保证对象是冻结拒绝算法在声明 iid 混合下、随机校准集产生的下一历史误拒率；不是计算成功率、拓扑正确概率、各弱支路子组风险、固定线路参数风险或现场推广保证。三末端生成总体限定 star 与三种 cherry，真候选包含条件必须单独检查。无限分数只是发现漏真/未知覆盖后停止交付筛选收益，不会自动修复候选覆盖。

## 4. 新历史与冻结

新校准种子固定 984001--984039，新评价种子 985001--985048；与旧 982/983、开发 981 的历史分开。每个独立历史只一种公开质量。case 的 star/weak/moderate/strong 混合为 0.15/0.15/0.40/0.30；三个 cherry 位置均匀，uniform/temporal/spatial 质量等权。标签仍使用 SeedSequence([seed,72148293]).spawn(3)，与物理生成 SeedSequence(seed) 分开。

case/family/quality 只传给仿真生成和评价入口；选择器仍仅接收当前历史拟合、公开质量和设备约束。未来真负荷、真实源和未来噪声不进入曲线选择。校准前冻结完整 87 个标签、方法、混合、23 个文件的字节/哈希（21 科学源、新 safe runner、新 safe 协议）、环境、固定操作/拟合预算及风险参数。旧 dataset 的 schema、源集合、标签和 identity 与本版本不同，恢复时必须拒绝。

## 5. 验收、分母与成本

原硬预算不改：WZZT 0.3 秒、每次 LP/MILP 5 秒、完整 AC 两初值、max_nfev=200、tol=1e-10。选择器 15 秒、单历史 90 秒为成本验收门；慢历史保存并继续，不按结果换 seed、不删除失败、不增加 AC 拟合预算。

新校准中，所有物理/拟合失败仍保留其历史和方法 trace；失败保留全候选但覆盖已知时不使阈值自动无限。覆盖缺失/未知使阈值无限，程序停止收益评价。新评价全部九方法的唯一正确/误拒/安全未决/失败按完整 48 历史互斥报告。缺失裕量记录理由；连续裕量只在两方法均有效时配对，附配对数和 case。每历史保存原子 gzip 记录和可恢复检查点。

先以 fake 数据验证：成功 gap 不变；失败得拒绝分数 0 但仍 compute_failure；调整损失与失败保留分支一致；真标签/覆盖标签不改变部署保留集合；漏真/未知覆盖仍无限；新最大阈值/精确尾界；旧版本 dataset 拒绝；成本和原失败分母不丢失。随后 root 审查并 commit，再运行新 AC。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'D:\apps\miniconda3\envs\Topo\python.exe' -m pytest research_experiments/mobile_ev_mle/test_ev_flexibility_formal_safe.py -q -p no:cacheprovider
& 'D:\apps\miniconda3\envs\Topo\python.exe' -m research_experiments.mobile_ev_mle.ev_flexibility_formal_safe --phase calibration
& 'D:\apps\miniconda3\envs\Topo\python.exe' -m research_experiments.mobile_ev_mle.ev_flexibility_formal_safe --phase evaluation
```

工作目录 C:\Users\23761\.codex\worktrees\36a1\Topo；默认输出 evidence/ev_flexibility/v2_safe_formal。--limit 只允许固定前缀续接，不改变 39/48 最终计划。按开发阶段约 6 秒/历史估算约 9 分钟；实际耗时和超限照实报告。本批次完成也不表示滚动自适应、人员路线和扩网研究已完成。
