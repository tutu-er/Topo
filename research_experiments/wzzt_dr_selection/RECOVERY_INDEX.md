# WZZT-DR 恢复索引

## 最新完成阶段：P15全部位置检查（2026-10-29）

先读STATUS.md顶部和dr_location_probe/REPORT.md。128/128格、124不同拟合已经完成，不重跑；evidence/dr_location_probe_v1保存聚合证据。私有目录为C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_dr_location_probe_v1。该目录summary complete、audit_final all_passed、运行锁已移除。103的局部收益来自起点收缩及重拟合；未增加完整族恢复，尚无WZZT交叉删除价格算法。下一步为历史候选竞争审查，尚未启动最优面LP或原跨日期扩展；没有恢复自动化。下方较早阶段的下一步保留作历史，不覆盖本状态。

2026-10-28。原A2价格指标路线已收口，自动化已删除。用户随后授权的数据环境与正常WZZT基线审查按下方新增记录接续；以STATUS.md最上方为当前状态，不重启原批次。

工作树：C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo
分支：codex/wzzt-dr-selection-20261027
远端：origin，https://github.com/tutu-er/Topo.git
最后数值阶段提交：b6b2d49b78413466fddf81a18c838279d0bc7b82。

## 首先读取

1. [LITERATURE_CLOSURE.md](LITERATURE_CLOSURE.md)：最终贡献判断、六篇最近邻、否定范围、重启证据条件。
2. [literature_closure_sources.json](literature_closure_sources.json)：正文版本、阅读位置、稳定来源和限制。
3. [A2D_REPORT.md](A2D_REPORT.md)：同历史反例与宽度指标停止依据。
4. [STATUS.md](STATUS.md)：当前终止状态。较早报告中的“下一轮”是历史计划，已由收口结论替代。

## 已完成材料与完整分母

| 阶段 | 协议／报告 | 已完成记录 | 可恢复归档 | 提交 |
| --- | --- | --- | --- | --- |
| A1/A2 | protocol.json、README.md | 6 组反例；4 历史×6 菜单×32 实现=768 评价；每状态仅32独立实现 | evidence/a1_a2_final_v4/，rows.jsonl.gz 及 manifest、summary、validation、源码／配置哈希 | 01f9399c |
| A2b | protocol_a2b.json、A2B_REPORT.md | 7 历史×4权重=28价格问题、224响应；保留有限批量失配 | evidence/a2b_face_final_v2/run_records.zip、summary、posthoc_checks、design_lock；该目录没有 validation.json | f8759d42 |
| A2c | protocol_a2c.json、A2C_REPORT.md | 17历史96块：69合法替代、18下界必要、5松弛未决、4整体失配；另列4次X方向事后检查 | evidence/a2c_completion_final_v2/run_records.zip、summary、validation、posthoc_x_summary | 28992078 |
| A2d | protocol_a2d.json、A2D_REPORT.md | 96块、318最大值主问题+204射线核验、189宽度LP；复用16唯一模式、新增16模式。三真模型24格含8复用，不额外计数 | evidence/a2d_presence_width_final/run_records.zip、original_run_manifest、manifest、validation、summary | b6b2d49b |

A2d 新增实际评价：完整恢复8/16、首次正确12/16、停止4/16；失败/超时0。概率0.82等来自声明的合成响应分布精确加权，不能用这些行数估计现场置信概率。A2c 的5个松弛未决保留；A2d 求解层失败/超时/数值未决为0。主结果为独立小系统参照，生产 WZZT、工程 AC 和真实用户行为均未验证。

## 实现入口

- core.py：完整层状族枚举、候选 L1 联合重拟合、数值原始／对偶检查。
- experiment.py、exact_patterns.py：A1/A2 与八模式精确积分。
- optimal_face.py、face_price_audit.py：固定支持最优面与有限批量比较。
- completion_bounds.py、completion_audit.py：共享块完整删除参照、结构性松弛下界。
- compatible_functionals.py、presence_width_audit.py：连续非负参数最大值、无界射线、未来相等预测下的宽度。
- tests/：累计88项测试，最近已存运行3.392秒；本次文档收口未重跑。

## 原始运行和报告修正的关系

outputs/ 为本地大表及断点，Git 保存精选证据和压缩归档。A2d 原求解在 outputs/a2d_presence_width_v1/；最终报告在 outputs/a2d_reporting_final_v3/，中间 reporting_v2 仍保留。修正是将4个不相容记录的空集合字段从0改为不相容，并补全无目标分支返回结构；优化和实际评价未重复执行。

A2d 归档同时保存原运行源码与最终汇总来源。复现旧数值用归档中的对应源码快照；检查当前修复实现用 validation.json 所列当前源码哈希。不可把不同版本的哈希混作同一次运行。

新环境只需阅读压缩文件即可查证。确需复现时，先检查进程及 run.lock，检查 zip 内相对路径，再按各 manifest/raw_run_files.sha256.json 恢复到原输入目录并核验哈希，保留原归档。A2c 依赖 A2b 的完整族拟合；A2d 依赖 A2c，并复用 A2b 主权重的实际响应。不要直接运行默认命令覆盖原输出。入口支持 --output 指定新目录；源码或输入哈希改变时须建立新的记录身份，不能绕过续跑检查。

## 未开始与数据边界

原A2收口时：B1 八节点闭环、B2 锁定测试和工程网络 C未开始，iFlex仅全文与小CSV已核验。后续已完成完整负荷校准及正常接口，详见STATUS.md与data_calibration/REPORT.md；原A2的未完成范围不能当成后续环境现状。没有上传家庭原始负荷、订阅论文全文或浏览器认证信息。

重新启动须先满足收口报告的数学／方法证据条件。当前没有待自动执行的实验。

## 2026-10-27：paper15需求响应环境

- 阅读end_to_end/README.md和REPORT.md；锁定config.json。主批private目录为visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_data_calibration/flow_runs/paper15_iflex_v1。
- evidence/end_to_end_v1包含3策略完整分母、16次扩展求解记录、运行检查及源码哈希；无奖励完成，两个有奖策略首步未决，不重跑以修改收益。
- CLI提供prepare/run/resume，实际resume已核验。正常tests路径为research_experiments/tests/test_wzzt_dr_*.py；原始负荷不入Git。
- 原A2宽度路线和自动化继续关闭。本阶段仅建立可用的环境与生产WZZT数据接口。

## 2026-10-27：现行默认RNJ→WZZT基线对齐

- baseline_alignment/README.md为当前入口；run_default_cli.py委托原CLI，mainline_adapter.py接收外部观测并复用生产RNJ和WZZT。v1源码/配置保持原样，仅README增加纠正说明。
- N0原reference、N1旧iFlex无奖励，均完成12次扩展；内部5/6+3误检、4/6+2误检。完整RNJ固定族两次LP均21/21、验证更好。不要将单步全获证误写成全树恢复通过。
- 私有完整结果：visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_baseline_alignment/runs；已归档证据evidence/baseline_alignment_v1。不重跑已完成条件，不比较价格。
- diagnose_v1.py与export_alignment.py为只读算术/导出；audit_rnj_family.py已执行恰好2次LP，输出存在会拒绝重复。正常主线没有自动恢复MILP搜索树的功能。


## 完整RNJ统一选模与错误路径诊断（2026-10-27）

- 报告：`baseline_alignment/SELECTION_REPORT.md`；规则：`pooled_selection.py`；正常观测入口：`mainline_adapter.py`、`run_observed_baseline.py`；正常CLI包装：`run_default_cli.py`。原WZZT与新最终结果分别保存，不覆盖。
- 已完成复算：`replay_pooled_selection.py`，26候选/2条件，结果 `evidence/baseline_selection_v2/pooled_selection.json`。新最终恢复2/2，全部来自RNJ；原WZZT0/2。
- 错误诊断：`wrong_block_diagnosis.py`及`WRONG_BLOCK_DIAGNOSIS.md`；19/19固定支撑LP，不重跑。证据含执行前协议、原输入/源码hash、独立损失核验和loader版本边界。
- 根部难例：`root_case_protocol.json`、`ROOT_CASE_DESIGN.md`、`root_geometry_audit.json`。3个长度比例的精确几何检查已完成；带噪声观测及实际WZZT未运行。保存原协议hash，不把几何结果当带噪声效果。`root_case_controls.json`另锁定简单RNJ容差对照，未运行新数据。
- 39项相关测试通过；无未完成求解进程，无DR新比较，无自动化。


## 2026-10-28：固定观测、上界与初始化

- 先读 `baseline_alignment/BOUNDS_INITIALIZATION_REPORT.md`。证据目录 `evidence/baseline_bounds_init_v3/`：7/7无上界固定族LP已核验；原四个合法单真块仍均输给已知错误候选。
- `initialization_audit.py` 只新跑RNJ-only前4次接受扩展，旧模式直接复用。已完成记录 `initialization.json` 不可覆盖；4/4次新扩展最优，新增1真3错，锁已移除，无待续跑进程。独立复算见initialization_verification.json。前缀不是完整恢复试验。
- 正常研究入口的 `initialization_mode` 明确支持 `leaf_plus_rnj`（旧默认）及 `rnj_only`。连续权重不锁定、原始终端与残差不聚合，完整RNJ最终选模仍与WZZT搜索单列。
- `PQ_CONTROL_PLAN.md` / `pq_control_protocol.json` 是尚未执行的三种输入依赖关系对照。根馈线缩短与需求响应比较暂停。
- 本轮46项相关测试通过；没有上传家庭原始负荷。N0输入继续使用此前私有完整基线目录，哈希记于协议中。

- 本轮补充5/5 RNJ-only首步无上界LP，同样排除cap2解释，结果rnj_only_bound_results.json；与原7LP分开。原默认作为下一步三波动条件的固定初始化，仍未执行该控制。


## 2026-10-28 DR有限候选池完整矩阵

- 主报告：dr_pool_v2/REPORT.md；运行和数据入口：dr_pool_v2/README.md。
- 已完成48格、42unique；全部结束。不要重跑prepare或重新求解；--resume仅用于核对已完成任务复用。
- 代码提交a58cf7ea；完整结果、协议、逐格结构、配对评价和独立复算：evidence/dr_pool_v2/。
- 私有根：C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_dr_pool_v2。physical_v1含新观测；identification_v1含42fit_cache与48任务；audit含首批和全量只读回执。家庭原始数据不上传。
- 恢复历史原始来源：同scratch下wzzt_data_calibration/flow_runs/paper15_iflex_v1。
- 本批DR无恢复收益；多初值搜索在高噪N24改善内部块但叶零权仍保留。继续方向是预声明跨日期/映射/同菜单位置对照，不恢复旧A2宽度代理。
