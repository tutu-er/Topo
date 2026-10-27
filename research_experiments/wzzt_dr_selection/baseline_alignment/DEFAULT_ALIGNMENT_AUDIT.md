# paper15 默认辨识基线对齐审查

审查日期：2026-10-27。审查者仅阅读代码和已有记录；本文件不报告新求解结果。N0实际运行与N1观测适配由主任务分别保存证据。

结论：end_to_end v1调用了生产WZZT核心，但绕过了当前默认RNJ初始化，并改变了数据数量、P/Q工况、采样间隔、求解后端和时限。它是纯WZZT短预算环境测试，不能作为正常paper15默认辨识基线。当前需先完成N0正常入口检查和N1固定iFlex无奖励观测检查，停用价格效果比较。

## 当前默认与v1的实际差异

当前工作树：`C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo`。下述行号以该工作树当前代码为准。

| 项目 | 当前默认paper15 | end_to_end v1 |
|---|---|---|
| 计算入口 | run.py→cli.main→pipeline.run | 研究adapter直接调用fit_laminar_l1_sensitivity |
| 辨识顺序 | ordered R/X QP→RX75→RNJ→循环分块bootstrap→选边界块→WZZT | 只有WZZT |
| 训练 | 3个工况×96点=288行 | Feb3/4/5历史72行+Feb9行动24行=96行 |
| 路径验证 | 独立replicate的3个工况×96点=288行 | Feb10单日24行 |
| 工况 | default、cloudy_variable_pv、evening_peak_high_load | 固定45户住宅对照组负荷，不含原混合DER工况 |
| 时间间隔 | 15分钟，先生成完整96点物理日 | 1小时实测平均功率，直接在24点上模拟 |
| Q | 不同用户类别和DER无功形成的净Q | 住宅PF曲线乘实际P；不是实测Q |
| RNJ参数 | tolerance=.16、100次bootstrap、块长4、筛选频率≥.75、最多2个边界块 | 未调用 |
| WZZT初始族 | 全部15个singleton＋筛选出的RNJ边界块 | 仅15个singleton |
| 支撑搜索 | 原终端上自由扩展，与初始族相容 | 相同自由扩展，但初始族不同 |
| 求解器 | HiGHS | Gurobi |
| 单次LP/MILP时间上限 | 1800秒 | 30秒 |
| 默认参数界 | 初始R/X均2；×2扩界，最多4次 | 相同 |
| 支撑数上限 | 默认2n−1=29 | 相同 |
| 选模 | 每场景4块的1-SE规则；3验证场景共12块 | 同规则，但1个验证场景只有4块 |
| 目标 | 原始P/Q与同期测得的根平方电压降，无截距、不去均值 | 相同 |
| reference物理尺度 | 阻抗倍率1；rootσ=.003；根电压均值1.02 | 相同参数值，物理过程采样网格不同 |
| 计量噪声标准差 | P/Q相对.005；端/根V相对.0002 | 相同参数值，随机实现不同 |

默认值的代码证据：

- `rnj_wzzt_core/rnj_wzzt/pipeline.py:359-383`：唯一计算默认来源。`time_limit=1800.0`、`milp_solver="highs"`、`selection_only=False`、`run_baseline=False`。
- `pipeline.py:198-256`：分别生成训练与验证replicate并对齐；默认replicate为0和1。
- `pipeline.py:130-195`：RX75、RNJ、bootstrap和边界筛选。bootstrap输出是选择频率，不是线路正确概率。
- `pipeline.py:318-340`：全部singleton和选中RNJ支撑组成initial_supports；只有显式run_baseline=True才增加纯MILP对照。
- `pipeline.py:431-438`：bootstrap随机种子为20260903+training_replicate。
- `rnj_wzzt_core/rnj_wzzt/scenario/settings.py:42-77`：reference的三工况、噪声、根过程及倍率。
- `rnj_wzzt_core/rnj_wzzt/scenario/profiles.py:229-263`：不同需求类型及PV/风电/机组对净P/Q的贡献。
- `rnj_wzzt_core/rnj_wzzt/estimation/laminar_l1_milp.py:476-486`：1-SE选最简路径点；`:1164-1184`：扩界、容差、验证块及gap默认值。
- `research_experiments/wzzt_dr_selection/end_to_end/wzzt_adapter.py:200-211`：v1明确为singleton-only，rnj_stage_called=False。
- `rnj_wzzt_core/README.md:28`：30秒是显式示例；`:34-49`明确普通入口默认完整RNJ+MILP，纯MILP必须显式增加。

主工作区`D:\0-github_workspace\Topo\rnj_wzzt_core\rnj_wzzt\pipeline.py`与本工作树版本经`git diff --no-index`核对无文本语义差异；文件SHA不同来自行尾。不能把这点误报为算法版本差异。

## 历史良好结果为什么不能直接作为现行默认结果

已读取以下原始报告和可解析记录，而非仅凭目录名判断：

1. `D:\0-github_workspace\Topo\outputs\mainline\report.md`、`metrics.json`、`rnj_summary.csv`、`milp_results.csv`。
   - paper15完整RNJ F1=1，选中{113,114,115}与{103,104,105}；联合结果F1=1，MILP约94.92秒。
   - 但config为daily_demean、contract_blocks=true、deembedding_weight=.5、candidate_pool_mode=rnj。
   - 该行initializer为rx75_rnj_contracted_plus_milp，MILP只使用11个伪终端、4个有限候选。
   - 因此它是旧缩并＋有限候选域的历史结果，不是现行原终端、零偏置、自由支撑搜索的正常结果。可用于解释历史来源，不能用其F1=1承诺N0。

2. `D:\0-github_workspace\Topo\outputs\core_refactor_20260927_095726\baseline_rnj_default\metrics.json`、`milp_results.csv`。
   - 实际配置仅1场景×4点、replicate7/8、bootstrap1次、块长2、最多1块、30秒。
   - 仍为daily_demean和rnj有限候选模式；paper15只恢复1/6内部真块。
   - 目录名default不代表正常数据/预算，是旧重构烟雾测试，不能与本轮v1的1/6混用或互为解释。

本次在上述D主工作区outputs及core outputs的metrics.json内未找到support_search_mode="unrestricted"的现行正常主批记录。此为本次检索范围内结论，不声称所有其他存储都不存在。

## 两阶段基线验证

**N0：当前正常paper15 CLI。** 只将案例列表缩为paper15、输出改为新隔离目录，其余计算默认完整保留。运行时应保存实际函数默认值，不能从README示例复制30秒。

```powershell
& 'D:\apps\miniconda3\envs\Topo\python.exe' rnj_wzzt_core/run.py `
    --cases paper15 --output <新的隔离结果目录>
```

主任务已启动N0，对原函数的委托包装仅保存输入、RNJ输出和进度，不改决策。实际记录位于scratch的`runs/N0_default_reference`。这是一项条件，不是四网络批量，也不启动纯WZZT对照。

**N1：固定v1无奖励观测，完整RNJ→WZZT。** 重用v1的同一72行历史＋24行行动、24行验证，P/Q/V/根电压和时间/列标签逐值不变。用现行默认选择参数和求解参数替换v1纯WZZT入口；真值仅在返回后评价。N1保留96/24行是为隔离调用方式差异，不得称已对齐N0的288/288行数据。

N0与N1算法默认相同，数据来源、样本量和时间分辨率不同；N1与v1观测相同，但初始化、后端和预算同时恢复默认，因此收益不能只归因于RNJ。若需要进一步归因，应先检查此次路径、初始块、求解界与数据诊断，再另锁定最小对照，不能事后逐项调到成功。

N1的bootstrap块长4沿用算法默认，在小时数据中覆盖4小时；N0覆盖1小时。这是数据网格差异，应披露，不能声称时间相关尺度完全相同。

两阶段均须保留RNJ全部块、筛选块、拟合诊断，WZZT每步目标/界/gap/状态，完整路径及1-SE选点、有效内部块命中和额外块。完成搜索不等于恢复真树。若未决或恢复差，报告原结果与具体失效位置；停止价格比较，不自动增加预算、降低噪声、放大阻抗或恢复旧白名单。
