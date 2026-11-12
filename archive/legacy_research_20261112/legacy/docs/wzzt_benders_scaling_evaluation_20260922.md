# wzzT Benders 原型：速度、迭代和规模测试

日期：2026-09-22。结论来自本机新增的 12 对匹配算例、24 次求解；另有 1 对冒烟检查，不纳入统计。核心和 Benders 原型文件均未改动。

## 结论

当前简单外循环 Benders **没有表现出整体加速优势，不适合直接替换现有 MILP**。它与原 MILP 求解同一个完整域单步扩展问题，但较弱的下界和重复求解主问题带来了明显开销。

- 8 终端、每场景 24 个样本时，两例 Benders 分别比原 MILP 慢约 6.41 和 5.29 倍，双方都完成最优性认证。
- 12 终端时，Benders 已找到与原 MILP 已认证最优值一致的可行解，但到 20 秒下界仍为 0；原 MILP 在 1.40–2.15 秒内认证。
- 18 终端合成例中，20 秒内 Benders 的可行目标也明显落后，不能只归因为认证慢。
- 32 终端时，原 MILP 在 30 秒预算内未返回整数可行解；Benders 返回可行解，但下界为 0，无法认证。这是可行解获取能力的一个观察，不是最优求解速度优势。
- 两个已有 AC 仿真算例中，双方都未在 30 秒内认证，原 MILP 的下界更好。

上述规模指**终端数**。本实验只做从 singleton 旧支撑族出发的一次完整域扩展，不是完整配网恢复或整棵树的全局优化测试。

## 受控实验口径

独立驱动：`scripts/benchmark_wzzt_benders_scaling_20260922.py`。

1. 双方使用相同观测、相同旧支撑、相同 R/X 上界、相同平均 L1 目标，搜索全部合法新增支撑，不使用候选池。
2. 合成数据：两个场景；除样本扩展例外，每场景 24 个时间样本；噪声标准差 0.003；R/X 上界 0.8/0.6。真实新增支撑只用于生成与事后评估，不传给求解器。
3. 4/8/12/18 终端各两个独立随机实例；`--repeats 2` 实际改变 seed，各实例只计时一次，不是相同实例重复两次。32 终端一个实例。
4. AC 数据来自已有 `advantage_stress_20260910/jobs` 中 Soumalas11 和 Pengwah18 的 `inputs.npz`，是仿真观测而非现场测量。仅加载 `train_*_P_terminal/Q_terminal/drop_target`，三个场景、每场景 32 个时间样本，R/X 上界均为 2；不加载真拓扑或候选列表。
5. 每方法单独子进程，先做微型 LP/MILP 预热，串行运行，交替方法顺序。时间包含建模和审计日志，不包含导入、数据生成与预热。
6. 两方要求绝对最优性 gap 为 `1e-10`；Benders 相对容差置 0。原核心保留其严格证书判断。浮点可行性设置并非逐项完全相同，另以独立模型重建检验最终可行目标。
7. 总预算相同，传给后端的是扣除已用时间后的余额。停止为协作式时限，实测存在轻微超出预算，例如 32 终端原 MILP 为 30.72 秒；不能把时限视为硬实时保证。
8. 原核心会隐藏未认证支撑；驱动只在内存中包装 `_run_milp`，保留后端 incumbent，并检查整数性、约束和重建误差。Benders 包装所有 master/LP 调用，补全原 trace 漏记的最终认证 master；不改变求解逻辑。

运行环境：Python 3.11.15、NumPy 2.4.6、SciPy 1.17.1，双方均使用其 HiGHS 后端及默认线程设置；没有线程数扫描、硬件隔离或统计显著性检验。

## 求解时间与认证

下表是停止时长；只有认证成功时才能视为完成最优求解的用时。区间为两个独立随机实例的范围。

| 数据 | 原 MILP 时间 / 认证 | Benders 时间 / 认证 | 解释 |
|---|---:|---:|---|
| 合成 n=4，T=24 | 0.057–0.122 s，2/2 | 0.069–0.134 s，2/2 | 小规模开销主导 |
| 合成 n=8，T=24 | 0.467–0.800 s，2/2 | 2.995–4.234 s，2/2 | Benders 两例各产生 37 条割、38 次 master |
| 合成 n=12，T=24 | 1.398–2.153 s，2/2 | 20.024–20.030 s，0/2 | 找到相同最优目标，未自行认证 |
| 合成 n=18，T=24 | 19.748–20.127 s，1/2 | 20.037–20.051 s，0/2 | Benders 可行目标也较差 |
| 合成 n=32，T=24 | 30.721 s，0/1 | 30.269 s，0/1 | 原 MILP 未返回 incumbent；Benders 有可行解 |
| 合成 n=8，T=96 | 2.302 s，1/1 | 2.992 s，1/1 | Benders 仍较慢，17 条割 |
| AC Soumalas，n=11 | 30.128 s，0/1 | 30.069 s，0/1 | 双方未认证 |
| AC Pengwah，n=18 | 30.097 s，0/1 | 30.011 s，0/1 | 双方未认证 |

4–18 终端主实验共 8 对：原 MILP 认证 7/8，Benders 4/8。全部 12 对：原 MILP 8/12，Benders 5/12。不能把这个小型、有选择的实例集当作普遍成功概率。

样本数从 24 增至 96 时，使用同一生成机制和参数 seed，但观测不是嵌套抽样；割数变化也受具体数据影响。这一个样本扩展例不能证明样本增多必然改善相对速度。

![停止时间与认证状态](../outputs/wzzt_benders_scaling_20260922/figures/runtime_scaling.png)

## 迭代到底做了什么

Benders 的上界是历次合法支撑拟合得到的最好目标；下界来自当前带割主问题的全局界。它可以访问一个更差的支撑，但保留的最好上界不因此变差。

| 实例 | 原 MILP 认证完成 | Benders 首次达到该最优目标 | 对应 LP 序号 | Benders 最终状态 |
|---|---:|---:|---:|---|
| n8，seed 0 | 0.467 s | 0.655 s | 19 | 2.995 s 认证 |
| n8，seed 1 | 0.800 s | 0.379 s | 11 | 4.234 s 认证 |
| n12，seed 0 | 2.153 s | 5.372 s | 43 | 20.024 s 未认证，LB=0 |
| n12，seed 1 | 1.398 s | 2.750 s | 35 | 20.030 s 未认证，LB=0 |

首次达到目标按与原 MILP 已认证目标相差不超过 `1e-10` 判断。n8 第二例说明 Benders 有时较早得到好解，但后续仍花费较长时间完成证明。原 MILP 的日志只暴露最终结果，不能据此比较它的“首次得到最优 incumbent”时间。

n12 两例的 master 分别用时 12.61/12.35 秒，LP 用时 7.12/7.38 秒，其余约 0.29/0.31 秒。master 占总时间约 62%–63%；两例整个过程 LB 都为 0。当前首要问题是割尚未使大量未访问结构的损失下界抬高，主问题反复探索仍被低估的结构。

全量事件包含 `master/subproblem/subproblem_failure/final`。`cut_count` 只计成功 LP 产生的割；例如 n12 第二例为 63 次 LP 尝试、62 条成功割，不能统称为 62 次全部迭代。

下图每个尺度展示第一个实例；以该例第一个有限 UB 归一化，原 MILP 水平线是其**最终**上下界，不是它的全过程。

![完整上下界轨迹](../outputs/wzzt_benders_scaling_20260922/figures/bound_trajectories.png)

## 较大例的解质量和剩余 gap

目标均为平均 L1 拟合误差，数值越小越好。这里的 UB 是可行解目标，不自动代表最优值。

| 实例 | 原 MILP LB / UB | Benders LB / UB |
|---|---:|---:|
| 合成 n18，seed 0 | 0.00199924 / 0.00228464 | 0 / 0.0508952 |
| 合成 n18，seed 1 | 0.002253664436 / 0.002253664437，已认证 | 0 / 0.0493133 |
| 合成 n32 | 后端未返回有效界或 incumbent | 0 / 0.0768381 |
| AC n11 | 0.000429360 / 0.000670731 | 0 / 0.000712275 |
| AC n18 | 0.000427899 / 0.000933715 | 0 / 0.000933715 |

合成 n18 两例，Benders 可行目标约为原 MILP 的 22.28/21.88 倍。n32 的最优值在本实验中未知；Benders 有解而原 MILP 尚无 incumbent，不能据此宣称前者给出了高质量或最优解。

AC n18 双方最终 UB 一致，但两者都没有认证，因此只能说找到相同目标值的可行解。原 MILP 的最终下界更强；Benders 的 gap 仍为 100%（按 `(UB-LB)/UB`）。

## 与直接调用求解器的区别

| 方面 | 原单体 MILP | 当前 Benders 原型 |
|---|---|---|
| 数学目标和整数支撑域 | 原模型完整域 | 相同完整域 |
| 求解器调用 | 一次 HiGHS MILP | 多次 HiGHS MILP master + HiGHS LP |
| 连续变量处理 | 与整数变量一起进入整体模型 | 固定本轮结构后拟合新旧 R/X、截距、残差 |
| 证明方式 | 整体分支定界与求解器割 | master 全局 LB 与可行拟合 UB 闭合 |
| 搜索状态 | 一次调用内管理树、节点和求解状态 | 每轮重新提交 master；没有搜索树或 LP 基复用 |
| 潜在收益 | 利用完整模型松弛、预处理和启发式 | master 不再携带逐观测残差与拟合约束 |
| 当前瓶颈 | 较大例整数 incumbent 和有效界也难获得 | 割下界弱，重复主问题与 LP 耗时累积 |

Benders 不是脱离求解器的另一种方法，而是改变模型分解和求解组织。只有两者都认证后，才能按同一模型最优值比较；Benders 本身不会改善模型的统计目标或消除组合难度。

设旧支撑数为 K、场景数 S、标量观测数 N，p=n(n+1)/2。当前 master 含 n+3K 个二元变量及 p 个 y 和一个 theta；LP 含 `2(K+1)+2p+Sn+N` 个变量、`2N+6p` 条显式不等式。此实验 K=n，合法新增支撑数为 `2^n-n-1`，移出观测约束不会消除这一组合搜索空间。

## 下一步改进应针对什么

1. **先改善下界和割。** 检查退化对偶带来的割强度，研究有证明的 Pareto 割、分数点分离以及全局有效界；不能把普通固定支撑 LP 的对偶直接拿来当通用割。
2. **复用求解状态。** 持久 master、LP 基复用或在同一搜索树中加入 Benders 割，针对当前重复建树/重求解开销。是否提速仍须重新匹配评测。
3. **提供较好的初始可行支撑和 UB。** 启发式用于 warm start 不需要限制完整主问题域，因此可以保留原问题的最优性目标；它主要改善寻解，不能代替下界证书。
4. **只使用有依据的边界收紧。** 统一提升模型中的有效 R/X 上界影响松弛和割强度；任意缩小上界可能改变原问题，不能当成保持等价的加速。

所有场景共享 R/X，不能简单拆成若干相互独立的场景 LP 然后相加，否则改变当前原问题。以上是针对实测瓶颈的下一步方向，本轮尚未实现或证明其加速效果。

## 审计与复现

- 所有 worker 和总报告均保存源文件 SHA256。28 个核心 Python 文件及原型前后均未变化；本轮只新增 benchmark、绘图脚本、文档和新输出目录。
- 主实验 8 对数据经独立重生成逐字节 hash 比对，输入一致；所有原 MILP 已返回 incumbent 均满足合法整数支撑与 R/X 边界。
- 全部新实验中，原 MILP 已返回 incumbent 的最大全模型可行性误差约 `3.34e-11`，重建目标与后端目标最大差约 `9.67e-13`。超时的 incumbent 仍可用于报告 UB，不能误称为认证结果。
- 所有 24 个求解调用均正常返回结构化记录；未认证和 LP 时限结束均保留。新增驱动另通过 n=4 冒烟对照；没有把之前的原型测试说成本轮重新运行。
- 汇总：`outputs/wzzt_benders_scaling_20260922/figures/summary.json`。
- 原始报告分别位于 `main/larger/archive/samples/benchmark_report.json`，各 job 另有完整 JSON 与即时保存的 events JSONL。

```powershell
# 使用新的输出目录；驱动拒绝覆盖已有 job 结果。
& 'D:\apps\miniconda3\envs\Topo\python.exe' -B scripts/benchmark_wzzt_benders_scaling_20260922.py --sizes 4 8 12 18 --samples-per-scenario 24 --repeats 2 --time-limit 20 --output-dir outputs/wzzt_benders_scaling_rerun/main
& 'D:\apps\miniconda3\envs\Topo\python.exe' -B scripts/benchmark_wzzt_benders_scaling_20260922.py --sizes 32 --samples-per-scenario 24 --repeats 1 --time-limit 30 --output-dir outputs/wzzt_benders_scaling_rerun/larger
& 'D:\apps\miniconda3\envs\Topo\python.exe' -B scripts/benchmark_wzzt_benders_scaling_20260922.py --sizes --archived-job rnj_wzzt_core/outputs/advantage_stress_20260910/jobs/ac_soumalas11_n0_r1_reference --archived-job rnj_wzzt_core/outputs/advantage_stress_20260910/jobs/ac_pengwah18_n0_r1_reference --repeats 1 --time-limit 30 --output-dir outputs/wzzt_benders_scaling_rerun/archive
& 'D:\apps\miniconda3\envs\Topo\python.exe' -B scripts/benchmark_wzzt_benders_scaling_20260922.py --sizes 8 --samples-per-scenario 96 --repeats 1 --time-limit 30 --output-dir outputs/wzzt_benders_scaling_rerun/samples
# 重绘本轮已完成的报告，不重新求解。
& 'D:\apps\miniconda3\envs\Topo\python.exe' -B scripts/plot_wzzt_benders_scaling_20260922.py
```
