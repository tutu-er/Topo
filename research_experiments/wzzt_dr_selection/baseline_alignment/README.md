# 正常RNJ→WZZT基线入口

先读[当前统一选模与诊断报告](SELECTION_REPORT.md)，原默认核验保留在[REPORT.md](REPORT.md)。本目录用于正常无奖励基线；价格比较暂停。`end_to_end`保留为v1纯WZZT流程复现，不应继续用它代表默认主线。

`mainline_adapter.fit_mainline_observed(training, validation, root_label, output_dir, overrides=None)`处理已生成的观测。默认值从生产`pipeline.run`签名读取，RNJ调用生产筛选函数；传入WZZT的参数与生产`_fit_milp_variants`相同。接口验证沿用P/Q/V/根电压标签完全对齐和pu单位约定，真值字段剥离。随后完整RNJ族经固定支撑L1重估，与全部WZZT路径点使用同一生产1-SE规则比较。`selected/summary/path`仍代表原WZZT，`final_selected/final_summary/final_selection`记录新的最终模型；超时/未决不因RNJ可用而隐藏。正常生产CLI自身仍负责生成原paper15数据，因此外部实测负荷使用这一数据适配层。

以下命令从仓库根目录运行。已有结果目录不得重复使用；本批已经全部完成，无须再次运行。

```powershell
$topoPython = 'D:\apps\miniconda3\envs\Topo\python.exe'
$baselineRoot = 'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a\wzzt_baseline_alignment'
$sourceRun = 'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a\wzzt_data_calibration\flow_runs\paper15_iflex_v1'

# 原生产入口；只限定paper15和新输出目录，保留其余默认值。
& $topoPython rnj_wzzt_core/run.py --cases paper15 --output '<新的输出目录>'

# 正常CLI包装：保留原WZZT输出，额外输出final_selection.json。
# 前阶段N0使用旧版只记录进度；当前需要复现旧版时加--wzzt-path-only。
& $topoPython -m research_experiments.wzzt_dr_selection.baseline_alignment.run_default_cli `
    --repo (Get-Location).Path --output (Join-Path $baselineRoot 'runs/N0_default_reference')

# 已保存的iFlex无奖励观测；不重新规划或模拟，不读取价格策略。
& $topoPython -m research_experiments.wzzt_dr_selection.baseline_alignment.run_observed_baseline `
    --source-run $sourceRun --output (Join-Path $baselineRoot 'runs/N1_iflex_no_reward') `
    --normal-reference-run (Join-Path $baselineRoot 'runs/N0_default_reference')

& .\scripts\test.ps1 -PytestArgs @(
    'research_experiments/tests/test_wzzt_dr_mainline.py',
    'research_experiments/tests/test_wzzt_dr_pooled_selection.py',
    'rnj_wzzt_core/tests/test_pipeline_reporting.py', '-q'
)
```

`baseline_protocol.json`保存两条件边界。N0是真实正常288/288点；N1保留旧96/24小时观测，不伪造15分钟数据。两者算法参数相同，数据配置差异保留在报告中。

`diagnose_v1.py`只分析旧记录和矩阵乘法，0次优化调用。`audit_rnj_family.py`是两次已完成的完整RNJ固定支撑LP诊断，输出已存在时拒绝重复；迁移归档不重跑它。`export_alignment.py`只导出已完成的路径和配置，不求解。

运行锁仅防止重复启动，默认CLI没有从MILP搜索树恢复功能。中断或失败应保留当前结果、协议和状态；不要将部分路径当成完成，不自动重试或延长预算。

两个主流程各12次扩展、全部达到数值最优判据；结构准确度另列。来源、源码和已保存观测的哈希在证据中；原始家庭负荷与完整私有观测继续留在仓库外。


根部短馈线方案见 [ROOT_CASE_DESIGN.md](ROOT_CASE_DESIGN.md)。已完成三个比例的精确 R/X→RNJ 几何核验，未运行新的带噪声 WZZT；复现入口如下，必须使用新输出文件：

```powershell
& 'D:\apps\miniconda3\envs\Topo\python.exe' -m research_experiments.wzzt_dr_selection.baseline_alignment.audit_root_geometry `
    --output '<新的输出目录>/root_geometry_audit.json'
```

原结果来自内联 Python；该保存脚本为事后整理，已做语法检查但没有重复运行。原执行与新脚本哈希的区别记录在 `../evidence/baseline_selection_v2/root_geometry_reproduction.json`。


当前两组已经完成统一选模，不应重跑上述N0/N1目录。`replay_pooled_selection.py`从已有路径和 `evidence/baseline_alignment_v1/rnj_family_audit.json` 复算，其命令参数为 `--runs <baselineRoot/runs> --source-run <sourceRun> --rnj-audit <已有RNJ证据> --output <新的JSON>`；拒绝覆盖已有结果。该入口不调用优化器。当前结果为26/26候选复算、两组均选择完整RNJ；原WZZT结果仍0/2完整恢复。

`wrong_block_diagnosis.py`已完成19/19事后固定支撑LP，协议与结果都在 `evidence/baseline_selection_v2/`；不要重复运行。它使用真块仅作失败机制诊断，不能作为部署候选选择器。根部困难配置见 `ROOT_CASE_DESIGN.md`，目前只有三个精确几何结果，没有带噪声WZZT效果结论。
