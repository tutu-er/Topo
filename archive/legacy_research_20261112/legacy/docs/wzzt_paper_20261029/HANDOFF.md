# RNJ+WZZT 全面测试交接

本轮目录为 `docs/wzzt_paper_20261029`，原始输入、物理网络、checkpoint、求解器与 AC 记录为 `outputs/wzzt_paper_20261029_full`。最终完成状态以 `delivery_verification.json` 和原始 `workflow_state.json` 为准；原 15 秒失败与 60 秒补充重试分别保留。

## 当前估计流程

1. 输入为同步端点 P/Q/V 与根电压，训练、验证、测试使用独立物理日。训练 N 是相同三天内的总采样量。
2. 训练数据先拟合有序约束 R/X，分别生成 raw/demean、R/RX75/X 的 NJ/RNJ 树与有限候选簇。
3. 对 15 个完整 RNJ 候选联合拟合 L1 平方压降响应，保留整棵树模型；此步骤可直接胜出，来源单列。
4. 原始 RX75/.16 初值搜索，加至多两个额外初值；每条局部路径预算 25 秒。修正提案可以删除与新簇交叉的旧簇，然后重新拟合整个固定家族。
5. `ridge_union` 用两档无量纲正则化强度（strength=.001/.01）扩充候选池，从多初值预选模型追加 25 秒搜索。实际QP参数为 `alpha = strength * max(||[P,Q]||_F^2 / (2*n), 1e-12)`，n为端点数；raw与demean分别计算，逐项alpha保存在fronts中。`restart_control` 在相同位置与预算继续搜索原候选池。
6. 在原始完整 RNJ 库及路径点中，按“验证 MAE 不超过最小值 3% → 非平凡簇最少 → MAE → 固定顺序”选模，最后应用 .16 强度过滤。R/X 系数界为 [0,2]；这里是响应系数，物理阻抗需包含因子 2 和标幺基准换算。
7. `ridge_guard0` / `multistart_guard0` 是预定后向删簇对照：从相应收缩前预选模型继续执行一轮、最多删一簇、15 秒预算，仅将验证误差不增的删除点加入选模。两步联合修正宽度 5，原单 bundle 对照保留，另补十 bundle 的固定父模型库组件测试。
8. AC 为选定拓扑后的训练参数重估与独立测试，不参与本轮拓扑选模。局部最小二乘设置 `max_nfev=12`，数值Jacobian所需的额外潮流调用不计入该值；另记全部测试点潮流计算是否成功及局部重估收敛状态。本轮未配置线路热限或运行电压限值。

固定家族 LP 原限 15 秒，要求最优状态；13 组触时时保留原始错误和已有 checkpoint。60 秒重试仅为独立补充，名义路径预算及观测数据保持相同，不能用重试成绩覆盖原分母。自由支持 MILP 为额外 80 输入对照，单步 10 秒、整条路径 100 秒、MIPGap=1%；可行暖启动、原生上下界和未决停止均保存，单步界限不扩展为整棵树全局保证。

## 论文证据怎么用

- 主文首先使用 [README.md](README.md) 的预定主比较与 `primary_statistics.csv`。主面板 640 输入由十个独立数据 bundle 构成；统计按 bundle 配对，噪声和 N 条件不当作 640 个独立重复。
- `by_case.csv`、`by_noise_N.csv`、`by_condition.csv` 展开网络、噪声、数据量和十类物理/计量条件。`physical_regimes.csv` 给出实际无噪声 AC 电压与根到端点压降，防止只以噪声参数描述实验难度。
- `data_diagnostics.csv` 保留442个物理/噪声条件的3978条逐日诊断，包括实际噪声、动态平方压降、每天96点设计矩阵的秩与条件数。
- 新拓扑 144 输入包括十二棵新树；case33 的二十输入来自同一棵端点化网络，不能充作二十棵网络。其额定负荷和骨干欧姆参数核对见 `case33_provenance.json`。
- `ablations.csv` 比较单独 lambda、25/100 秒、删簇和两步。每行仅与相同输入上的参考配对。主面板的改善不直接外推到新树、弱激励或现场三相系统。
- `joint_extension_comparisons.csv` 为两步组件的十bundle补充：160输入、480模型，加入新LP的相同追加预算单步对照。该扩展在主结果后提出，复用相同数据，属于次要描述性证据，设计与冻结时间单列。
- `joint_extension_paper_comparisons.csv` 再列相对已有删簇策略的增量；`joint_selected_diagnostics.csv` 保留最终模型来源、成对提案及第一步训练损失，支持逐例解释收益，避免把所有变化都写成跨越局部障碍。
- `unidentified.csv` 和 `failure_modes.csv` 保留未恢复、漏真簇和交叉错误簇。`greedy_events.csv` 才用于检查完整扫描中错误正权簇严格优于已测真簇的一步目标偏好。
- `postfilter_guard_*` 是探索性过滤保护，保证同一预选模型的验证 MAE 不增，但主面板和场景 F1 均有下降。其有时间戳的补充检查排除了动机预检数据。不得将该规则写成通用拓扑增强。
- 自由域、LP60 与 AC 分母见各自验证 JSON；误差均值与模型可用数、物理可行数、原始失败数同时报告。
- `case33_recovered_coverage.csv` 为单列的补充覆盖：N48沿用原LP15，原失败N288取LP60重试，每行标注来源和LP上限。它支持全部二十输入的后续检索，原始失败分母仍保留；N档之间的计算预算有差异。
- `claim_evidence.csv` 将论文主张与证据及适用域对应。完整记录数量和所有局部求解状态都不构成拓扑正确概率。

## 查找和调用

```python
from research_experiments.wzzt_paper.records import find, load, load_trace, load_guard, load_retry

catalog = r"D:\0-github_workspace\Topo\docs\wzzt_paper_20261029\catalog.sqlite"
rows = find(catalog, panel="main", case="flynn16", voltage_sigma=.001, N=288)
model = load(catalog, panel="main", case="flynn16", condition="reference",
             voltage_sigma=.001, bundle=3, N=288,
             variant="ridge_union", with_observations=True)
R, X = model["R"], model["X"]
```

`load` 读取原 15 秒批次；不可用模型会明确报错。`load_retry` 显式读取同一条件的 LP60 补充，并在 `original_record` 中保留原始失败；`load_guard` 读取单列的探索性过滤结果。`load_trace` 返回绑定哈希的原始路径、候选及终止记录。`with_observations=True` 自动按 N 读取训练子集并验证观测哈希，验证/测试各为完整 288 点。

原AC辅助函数的 `refit.scope` 中含有历史标签“all 96 rows”；实际训练行数以输入N及 `ac_converged_points` 为准，最终核验逐项检查其等于N、两类测试各为288行。该字符串不参与计算。

大表、SQLite 和逐模型 JSONL 在本机直接保留，在 Git 中通过 `records.zip` 压缩保存。新检出目录可用 `records.restore(directory)` 恢复；已有文件哈希不一致时拒绝覆盖。原始 NPZ 和每一步求解日志位于输出目录，复现时使用 `producer_source.zip` 和冻结协议生成。

## 恢复与核验

入口为 `research_experiments/wzzt_paper`。`prepare` 创建新输出目录和冻结源码；`run` 运行主估计；`workflow` 依次完成补充重试、自由域、AC、统计、独立计时、报告和核验。已有批次恢复时使用同一个输出目录，不修改源输入和结果：

```powershell
& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_paper.workflow outputs/wzzt_paper_20261029_full docs/wzzt_paper_20261029
```

若某分析阶段失败，修复后指定 `--from-phase <阶段名>`，历史日志另存，不覆盖旧尝试。独立计时需在其他估计批次结束后进行。最终以 `delivery_verification.json` 的逐项检查、`audit_summary.json` 的可用证据核验及原始失败分母共同说明状态。
