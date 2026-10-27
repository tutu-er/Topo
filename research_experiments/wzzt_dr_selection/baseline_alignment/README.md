# 正常RNJ→WZZT基线入口

先读[核验报告](REPORT.md)。本目录用于正常无奖励基线；价格比较暂停。`end_to_end`保留为v1纯WZZT流程复现，不应继续用它代表默认主线。

`mainline_adapter.fit_mainline_observed(training, validation, root_label, output_dir, overrides=None)`处理已生成的观测。默认值从生产`pipeline.run`签名读取，RNJ调用生产筛选函数；传入WZZT的参数与生产`_fit_milp_variants`相同。接口验证沿用P/Q/V/根电压标签完全对齐和pu单位约定，真值字段剥离。正常生产CLI自身仍负责生成原paper15数据，因此外部实测负荷使用这一数据适配层。

以下命令从仓库根目录运行。已有结果目录不得重复使用；本批已经全部完成，无须再次运行。

```powershell
$topoPython = 'D:\apps\miniconda3\envs\Topo\python.exe'
$baselineRoot = 'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a\wzzt_baseline_alignment'
$sourceRun = 'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a\wzzt_data_calibration\flow_runs\paper15_iflex_v1'

# 原生产入口；只限定paper15和新输出目录，保留其余默认值。
& $topoPython rnj_wzzt_core/run.py --cases paper15 --output '<新的输出目录>'

# 同一生产CLI，加只记录进度的包装；本批N0使用此形式。
& $topoPython -m research_experiments.wzzt_dr_selection.baseline_alignment.run_default_cli `
    --repo (Get-Location).Path --output (Join-Path $baselineRoot 'runs/N0_default_reference')

# 已保存的iFlex无奖励观测；不重新规划或模拟，不读取价格策略。
& $topoPython -m research_experiments.wzzt_dr_selection.baseline_alignment.run_observed_baseline `
    --source-run $sourceRun --output (Join-Path $baselineRoot 'runs/N1_iflex_no_reward') `
    --normal-reference-run (Join-Path $baselineRoot 'runs/N0_default_reference')

& .\scripts\test.ps1 -PytestArgs @(
    'research_experiments/tests/test_wzzt_dr_mainline.py',
    'rnj_wzzt_core/tests/test_pipeline_reporting.py', '-q'
)
```

`baseline_protocol.json`保存两条件边界。N0是真实正常288/288点；N1保留旧96/24小时观测，不伪造15分钟数据。两者算法参数相同，数据配置差异保留在报告中。

`diagnose_v1.py`只分析旧记录和矩阵乘法，0次优化调用。`audit_rnj_family.py`是两次已完成的完整RNJ固定支撑LP诊断，输出已存在时拒绝重复；迁移归档不重跑它。`export_alignment.py`只导出已完成的路径和配置，不求解。

运行锁仅防止重复启动，默认CLI没有从MILP搜索树恢复功能。中断或失败应保留当前结果、协议和状态；不要将部分路径当成完成，不自动重试或延长预算。

两个主流程各12次扩展、全部达到数值最优判据；结构准确度另列。来源、源码和已保存观测的哈希在证据中；原始家庭负荷与完整私有观测继续留在仓库外。
