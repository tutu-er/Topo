# 本轮冻结实验续接：数值与统计完成，逐记录AC复核中

当前策略、阈值及115份科学源码保持冻结。192校准和96独立评价均已完成；controller在2026-10-16 09:18:27 UTC置为`statistical_complete`，两worker已退出。禁止重新启动规划、改阈值或追加新批次。

## 先核对仍在运行的后处理

执行目录`D:/0-github_workspace/Topo/outputs/ev_rx_stage_20261015`；Python为`D:/apps/miniconda3/envs/Topo/python.exe`。以下均是已启动的主机权限隐藏进程，数值线程为1；先看日志和主机PID，禁止重复启动。

|任务|PID|状态或输出|
|---|---:|---|
|全AC a组106槽|45900|`outputs/mobile_v2_fullreview_a.log`及同名`.json`|
|全AC b组106槽|47152|`outputs/mobile_v2_fullreview_b.log`及同名`.json`|
|全AC c组104槽|14224|`outputs/mobile_v2_fullreview_c.log`及同名`.json`|
|成本4槽|34648|已完成，`outputs/mobile_v2_fullreview_cost.json` success=true|
|缓存所有权→证据打包监督|1004|`outputs/mobile_v2_evidence_run_v1.json`|
|缓存所有权288槽子进程|664|`outputs/mobile_v2_evidence_owners_v1.log`|

AC启动配置、分母和工具SHA在`outputs/mobile_v2_physical_review_launch_v1.json`；每组完成后须读JSON的`success`，进程退出或CLI状态本身不足。a/b/c加cost共320槽，与旧pilot4、原开发28共同覆盖352槽；开发重复seed不另算独立历史。监督脚本`outputs/mobile_v2_evidence_run_v1.ps1`按顺序owner→pack --plan→pack；退出后仍须核对证据清单与恢复验证。不要同时启动第二份打包。

## 已完成并可直接核验

- 两库`outputs/mobile_v2_formal_n3_analysis/`与`n4_analysis/`含全部3拟合口径、5规则、11方法的统计、成本、选择日志投影、预测排序诊断和Wilson/配对分析。
- 预声明8历史×4移动方法空间消融32/32完成、失败0。实际移动7槽对应4条不同计划；32槽中未移动者空间项为0。4条计划空间项均正，n3其中一条总收益仍负。`u0`为同净功率放回初始点的电气反事实，少耗.2kWh驾驶电量，不作为已校准可执行固定策略。
- `outputs/ev_mobile_v2_final_tests_v2.xml`：237 passed，138.88秒。首次XML保留，其中73项因basetemp父目录缺失而setup失败；建立目录后全套重跑通过，科学源码未变。
- 统计/空间/测试原字节压缩证据：`evidence/ev_mobile_v2/poststats_20261016/manifest.json`。未来报告不能把这一阶段当全研究完成。
- 115份源码在C、D均逐字节匹配freeze。原阈值两库先于首次评价启动保存，858组独立核查见`formal_thresholds/`。

## 下一步只完成交付

1. 等a/b/c全部完成，核对3报告success与分母，汇总352槽AC检查/失败。工具源码不在运行中修改。
2. 读取`outputs/mobile_v2_formal_source_cache_owners_v1.json.gz`：区分已知未付费缓存、重复查询、唯一键/费用及缺失日志。只更正已保存路径费用解释，不改冻结选择器或伪称独立重跑耗时。
3. 等`outputs/mobile_v2_evidence_v1`完成。14逻辑目录352槽，预选完整代表和n3/n4评价0完整raw。核对包SHA、≤20MiB及恢复代表的原文件SHA；随后复制C研究目录`evidence/ev_mobile_v2/delivery`。
4. 最终审计、空间、测试、成本和新工具字节快照纳入新证据清单。已有stage0/operations/formal_thresholds/poststats保持不改。
5. 合并`outputs/mobile_v2_final_report_draft.md`与空间/费用报告，写简明`ev_mobile_v2_results.md`，更新README、复算入口与检查点。分别说明物理裕量、校准决策和预算/预测问题；不预设移动改善。
6. scoped commit、普通push `https://github.com/tutu-er/Topo.git` 的`codex/ev-train-validation-design`。用户已明确授权，无需再问。核对无关PDF及pyproject的原SHA，不用`git add .`。
7. 全部核对、提交推送完成后，给出三个问题的完整回答，将`topo-ev`设为PAUSED。未完成时继续保留检查点，不写成完成。

旧故障、恢复经过和完整命令已在Git历史与operations证据中保存。主机查询与沙箱PID可见性不同，单次沙箱`Get-Process`为空时先检查日志和主机只读查询。
