# 2026-10-18 完整流程冻结批次

源码分支 `codex/ev-train-validation-design`，科学实现提交 `dfe24edf`。
执行副本 `D:/0-github_workspace/Topo/outputs/ev_full_pipeline_stage_20261018`。
输出 `outputs/full_pipeline_v1`，输入 `outputs/full_pipeline_inputs_v2/prepared.json.gz`。
先读本目录 protocol.md、math.md、checkpoint.json，再核对输出 controller.json、controller.lock.json、controller_workers 与日志。已有12个worker；禁止重复启动。

冻结指纹 `eae436295601f57f758d4dcc304152b1ca24c12ca9d3c40110fcc24c1b513fe5`，包含864份源码/数学文件；完整源码快照、运行环境和窗口hash都已保存。正式顺序为24开发、48校准、64评价条件，统计单位为6/12/16个日期簇。开发后不按评价结果改策略。87组件测试和30个既有AC梯度/系数测试通过；两份额外开发pilot仅用于接口/成本验证，不混入正式分母。

若controller存活，只检查进度。若已结束且无活跃worker，可用同一命令续接；失败raw保留，不自动重新抽样。命令须在执行副本，设置PYTHONPATH为该根及rnj_wzzt_core，BLAS线程1：

```text
python -m research_experiments.mobile_ev_mle.full_pipeline.controller --output outputs/full_pipeline_v1 --prepared outputs/full_pipeline_inputs_v2/prepared.json.gz --phases development calibration evaluation --workers 12
```

仅在controller为statistical_complete后运行：

```text
python -m research_experiments.mobile_ev_mle.full_pipeline.analysis --output outputs/full_pipeline_v1
```

分析先仅用校准文件冻结thresholds.json，再读取评价。实际阈值不使用未知源类别。检查audit.json、summary.json和results.md；保留物理失败、优化失败、漏真和未决。若需修正后处理，仅修改analysis.py及独立工具，记录新hash；不得修改冻结科学文件。先前outputs/full_pipeline_preflight_incomplete_hash仅是源码清单检查失败的预启动快照，没有科学样本，不能混入正式批次。

后处理工具在outputs/full_pipeline_tools，证据包需包含全量raw哈希、全部统计重算输入、配置/来源/源码、预定4份开发物理回放样本；完整raw保留在执行副本，不把大型重复轨迹全部提交Git。交付需核对原生WZZT/RNJ、全树候选覆盖、固定/联合线路、候选更新和不同EV策略的独立贡献。真实部分仅为OPSD建筑进口功率形状；线路、Q、DER与表计为合成。

按阶段提交并普通push已授权origin/codex/ev-train-validation-design。保护原PDF和pyproject修改，不创建新科学批次，不更新长报告。尚未完成的研究不能写成完成。
