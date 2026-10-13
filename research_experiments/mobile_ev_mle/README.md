# AC 最大似然拟合与 EV 模型判别设计

## 本轮优化式自适应研究（2026-10-13，阶段交付；研究仍有未完成项）

先读 adaptive_milp_protocol.md、adaptive_milp_checkpoint.json、adaptive_resume_plan.md。完整方案和验收条件已经保存；下面旧章节记录前期动作库研究，不代表本轮主选择器。本轮通过时间展开移动网络和连续功率变量配置 EV，未枚举完整路线。主框架允许竞争线路、历史状态及同点逐时源在冻结局部域内调整；完整 AC 回代和实际 H+新增联合拟合分别核查代理与统计效果。

- adaptive_history.py / symmetric_ac_local_math.md：完整 AC 白化导数、历史 profile 信息、补偿、活动面和多个展开点。
- ev_operations_milp.py / ev_information_milp_math.md：接入、移动、SOC、效率、行驶能耗、互斥与充电任务。
- ev_information_milp.py / ev_information_milp_math.md：support 下界、固定竞争见证及平方弦上界，QP 与 MILP 的界方向。
- adaptive_milp_study.py、adaptive_rolling_study.py：冻结的一轮与滚动独立历史实验；全部失败保留。
- adaptive_statistics_math.md、adaptive_result_audit.py：独立历史校准、实测 AC 重放、求解界审计。局部代理域不等于统计覆盖。
- empirical_design_analysis.py、impedance_component_audit.py：代理与真实排错关系、逐类边的 R/X 误差；分析不改策略。
- wzzt_export_audit.md：WZZT 默认 Source=None；当前提供可修正结构提议，尚未把内部阻抗权重作为 AC 初值。
- domain_nonlinear_evidence.md、nonlinear_taylor_stress_v2_evidence.md：宽解释域、完整 AC 竞争见证及代理失败。
- ac_reflection_counterexample_math.md：总表已知且局部根线路 profile 满秩，仍可有两个全局等价参数点的完整 AC 反例。
- history_cost_centering_math.md、independent_math_review.md：共同常数归一化与改变判据的逐对消融；v2 证据更正字段语义，原 v1 保留。
- weak_branch_information_math.md、weak_branch_peer_review.md：完整 AC 的弱内部边信息上界及因果自适应量词；不适用名义混合门控的统一 size 声明。
- nonlinear_surrogate_bounds_math.md：代理界推广到完整 AC 所需的统一误差条件；当前仅有点值回代，未满足该条件。
- expected_scenario_design_math.md：独立加权情景 MILP 原型及解析小例，尚未接入 C/D；一次微小数值界冲突保留为失败。
- continuous_predictive_eig_protocol.md：有限 prior 失败后的连续 proper 积分尝试，独立诊断，不改正式策略。
- future_prediction_math.md、control_gradient_math.md：追踪与背景不可分、proper 预测的待实现模型、补偿后的控制梯度与零梯度情形。

C 冻结 128 个校准历史、256 个独立评价历史；D 使用自身的 64+96 条完整两轮轨迹校准/评价。种子、源码哈希、物理/优化失败和原始负 gap 均保存。名义风险仅针对声明模拟分布下冻结的整策略，不能迁移到现场、任意参数或模型失配。有限 prior 的 4 条历史适合度诊断均拒绝，故不把这次有限 EIG 当作连续模型的概率证据。

C 已完成：128 校准（1 条拟合失败仍以∞保留）、256 独立评价及7168候选 AC 重放。唯一正确保留为重拟合 MILP 71/256、恒充62/256、原库81/256；全七方法联合误剔除14/256。收益主要在强 cherry，弱结构仍未决；代理增益与实测裕量相关较弱。详见 formal_C_independent_review.md 和 evidence/formal_C_independent_numbers.json。

D 已完成64校准、96评价及768份保存拟合的AC重放：最终唯一正确37/96，整轨迹任一轮误拒10/96（CP95 [5.11%,18.32%]），两轮选址平均25.04336秒。C/D不同历史与门限，不能据两表成功率差判断反馈收益。四压力批次16/16/16/32条全分母核验；最后一批漏真10/32。压力结果不继承名义人口风险保证。

共享功率平方新上界利用精确时间块及单EV互斥：两历史旧主问题变量23194/57754和17434，共享为322/466和322；696QP、16MILP、96AC保存见证重放通过。两共享动作相对恒充的代理增益区间均跨零。它尚未替换冻结C/D，不据速度改善推辨识改善。

proper连续AC密度与逆变换已证明并验证，但五版历史积分全部未过原七成分验收条件。最终总体ESS虽达437.317/259.058/193.939，另六成分ESS约1；未接受后验近似，未计算可信预测EIG或排名。API中的posterior_available仅指有限对数证据的数值样本库非空，不能代替原验收条件。详见continuous_eig_sampling_math.md、evidence/continuous_eig_compact/。

预算后配对续接重复同2pilot，科学源码、数据、动作、目标和调用数与旧pilot一致，但墙钟均值74.58163秒超过冻结45秒扩批门槛；保留失败，正式校准/评价为0/32和0/48，未取得反馈vs固定首轮计划效果。具体变慢原因没有同期负载证据。原八小时协议和预算后01:54UTC续接分别保留，不能说新续接在八小时内完成。

原报告在原路径更新、编译为49页并全部视觉核验，证据为evidence/adaptive_final_delivery.json。内置编译器因Windows沙箱设置不可用；原位XeLaTeX三遍成功，无越界或未定义引用。此前40页交付快照仍保留。outputs/和staging保存详细JSON/NPZ；evidence/提交无损压缩逐候选证据、版本/哈希与独立核验器。研究没有整体完成，未扩网。真实Git归档的640份文件逐字节匹配，四个核验器全部通过；第一次受全局outputs忽略规则影响的失败及修复另存，未修改旧科学记录。

可续接/只读核验入口（工作树根目录）：

    python -m pytest research_experiments/mobile_ev_mle -q -p no:cacheprovider
    python -m research_experiments.mobile_ev_mle.shared_power_square_audit
    python -m research_experiments.mobile_ev_mle.continuous_eig_recheck_compact
    python research_experiments/mobile_ev_mle/verify_adaptive_compact.py research_experiments/mobile_ev_mle/evidence/adaptive_d_compact
    python research_experiments/mobile_ev_mle/verify_adaptive_compact.py research_experiments/mobile_ev_mle/evidence/adaptive_stress_compact

本目录conftest.py排除evidence/outputs中的冻结源码镜像；原第一次532项通过的记录含10项历史toy镜像，另保留其日志。活动研究测试522项通过（126.65秒），实际日志见evidence/adaptive_final_test_validation.json。不要把历史快照重复当新增测试覆盖。
配对续接命令位于paired_d/run_continuation.py及paired_d/continuation_v2.md；按现有freeze继续会停在失败耗时门槛，不能直接跳过它启动正式批次。下一轮须先固定新预算/实验协议及分母，保留旧记录。

本目录已实现新研究原型。旧路线见 ../mobile_ev_probing/README.md。
旧外 LP 的未决不用于判断 MLE 路线。

## 前期运行入口（动作库阶段，2026-10-12）

科学 Python：D:\apps\miniconda3\envs\Topo\python.exe。
在工作树根目录运行，OPENBLAS/OMP/MKL 线程建议设为1。

    python -m pytest research_experiments/mobile_ev_mle -q -p no:cacheprovider
    python -m research_experiments.mobile_ev_mle.run_study --phase pilot
    python -m research_experiments.mobile_ev_mle.run_study --phase formal --output research_experiments/mobile_ev_mle/outputs/formal
    python -m research_experiments.mobile_ev_mle.run_study --phase stress --output research_experiments/mobile_ev_mle/outputs/stress --thresholds research_experiments/mobile_ev_mle/outputs/formal/thresholds.json
    python -m research_experiments.mobile_ev_mle.run_study --phase scale --output research_experiments/mobile_ev_mle/outputs/scale

配置、源码SHA-256、种子、逐候选阻抗/取得的最大局部对数似然、
重启损失/优化状态、物理残差、局部秩、动作/失败和计量哈希均保存。
outputs/保存详细JSON与NPZ，Git只提交本任务简明证据。

## 模型与限制

likelihood.py：平衡径向常功率AC；R、X分别估计并跨历史/新增共享。
末端真实总P/Q逐时拟合，EV已包含其中。源P/Q可逐时变化，经同一个
公共干线下端物理接入点作用；总表P/Q或独立源计量必须锚定。

根电压与固定非零P/Q/V偏差先独立校准。可选Calibration模式用独立
校准数据联合拟合跨时偏差，校准似然只计一次；主实验不放开偏差。
已知对角Gaussian误差及跨通道/时间独立是合成假设。
根电压噪声、相间不平衡、动态/平均计量和未知源位置未实现。

量用pu；base20kVA、0.4kV，Zbase8ohm。R/X搜索域[1e-6,.1]pu；
末端P/Q[-1,1]pu；源P/Q[-.5,.5]pu。优化后检查电压[.85,1.15]，
动作额外做参考预测[.9,1.1]检查。多初值只有最佳已发现局部拟合，
没有全局损失下界，也未枚举其他AC平衡点。失败不作为拒绝，
似然不是拓扑概率。

## 候选与设计

candidates.py复用passive_bridge_v9.py的RNJ/WZZT提议并允许替换。
三末端约化类为星形及三个二末端clade，使用预定义完整局部修正库；
大网络为有限候选，不自动包含真拓扑。RNJ的X/R=.5仅为提议代理，
不约束AC拟合。WZZT预算/失败原样保留。

design.py按Dette–Titoff(2009)第2、4.2节，通过历史+未来联合重拟合
计算竞争模型的历史似然代价及未来2KL差异。竞争R/X、逐时状态/源
共同重拟合。参考和平均负荷/源预测来自历史，没有真拓扑或未来噪声。
局部内层不提供精确T分数保证；负历史增量、失败或电压越界使动作无效。

初始12历史点+8未来点，每点15分钟。充电6kWh、最大6kW，
电池60kWh、效率.95、SOC[.1,.9]、初始SOC.4，所有车从地点0开始。
接入按时段给定；移动至少1时段不充电，每次消耗.2kWh。
Q/P固定0；不假定普通EV可独立控制Q。
有限库上限11，包括各站点恒充、早/晚脉冲和两个移动路线；
库截断及不可接入保留。无EV与四种策略观测长度相同，
四种充电策略电量相同。

## 统计验证

每个强/弱场景各20校准种子与20独立评估种子。每个校准种子重跑
被动候选、历史拟合、选址、实际联合拟合。用每试验跨5方法的最大
真候选损失差与绝对损失设阈值(alpha=.1)。校准与主要评估都要求
配对5方法完整；失败另外计数。检验仅覆盖声明模拟分布/候选/优化器，
不提供现场或任意参数保证。固定移除clade01检查候选漏真，
大固定电压偏差检查模型失配，另有小规模扩展。阈值、误拒绝、
排错、未决、阻抗/源误差和时间随结果公开，不预设EV必然改善。

## 文件

- protocol.md：运行前协议。
- theory.md：符号、公式、Gaussian KL及历史代价证明和识别边界。
- literature_evidence.json：指定正文阅读位置、版本、来源和哈希。
- test_likelihood.py：独立AC/导数/无噪声/校准/失败验证。
- test_design.py：历史约束作用、物理约束、预测及失败保留。
- test_pipeline.py：配对计量、EV一次计入、信息隔离及阈值限制。
- docs/mobile_ev_probing_report_v2.tex：在原路径更新研究主线。

生产rnj_wzzt_core未改。既有pyproject.toml及旧未跟踪文件不属于本任务。

## 已核验结果（2026-10-12）

80个正式试验、400个方法记录。强支路评价有1个固定脉冲竞争拟合超预算，
因此主要配对分母19；弱支路分母20。强支路全排错次数：
无EV14/19，恒充17/19，固定脉冲17/19，随机动作17/19，设计19/19。
设计比恒充仅多2个seed，不能据此断言显著或普遍优势。
弱支路五策略均0/20次全排错；设计错winner14/20，误拒绝真候选1/20。

校准真值仅来自三种cherry的混合，star只作为竞争候选，未校准star真值。
0/19误拒绝的95%两侧区间上界约.1765，不能称零风险。
漏真stress主要配对3seed：无EV绝对损失告警0/3，各EV3/3；
另1配对随机方法计算失败。大固定电压偏差0/4告警却增加阻抗误差，
说明低损失不能证明计量模型或参数正确。

五末端1seed包含真候选且所有拟合完成，但五策略局部似然赢家均加入
错误额外支路；真模型差值.3318--3.1806。没有迁移三末端阈值。
更丰富结构/极弱额外边会拟合噪声，报告给出稳定AC分支下的连续性证明。

23项实质测试通过。正式优化失败另做相同观测的恢复诊断：
cold初值15nfev取得loss1534.66025，而原warm120超预算；
不改原失败记录和阈值。SOC先扣行驶再检查充电前状态的边界已修复，
证明当前11个正式动作及选择均未受影响。

版本：正式科学流程434b2d3；种子恢复f741aac；SOC边界修复5080d4e。
正式记录对应修复前源码哈希，SOC修复对本算例可行集严格等价。
最新代码重跑会生成自己的配置/源码哈希；不要把不同优化配置的记录
混作同一校准分布。

发布证据：
- evidence/formal_summary.json：阈值、误拒绝区间、排错、参数误差、耗时。
- evidence/formal_fits.csv：1920行历史/联合候选阻抗及最大已取得loglik、失败。
- evidence/formal_design.csv：2640行竞争模型联合设计内层结果。
- evidence/formal_trials.json：80试验的动作、保留集合、覆盖、评价和来源。
- evidence/stress_* 与 scale_*：漏真/偏差/五末端诊断。
- evidence/failure_diagnostics.json、optimizer_recovery.json、soc_repair_validation.json。

详细JSON和NPZ仍保留在outputs/及本地evidence/*_records.json，后者约10MB
冗余展开数据不推送；CSV保留参数、目标和求解诊断，种子和源码可重建。
生成compact证据：先运行collect_results，再运行export_tables。
首次MLE交付为原docs/mobile_ev_probing_report_v2.tex及19页PDF；当前原位更新为27页，核验见evidence/three_direction_delivery.json，旧版本在Git历史中。

精确AC等价构造：无上游/源遥测时，改变公共干线阻抗与下游相位，可由同点逐时源补偿而保持末端P/Q及电压幅值不变。三种构造验证了该声明配置下的不可分离性；总表锚点仍不能自动证明全部线路参数唯一。

## 数学表达修订

第三节先解释固定拓扑内的“线路与共同源能否分开”，用不改变下游状态的公共干线扰动逐步证明等价；总表限制该构造但不证明全部参数唯一。
报告第2–6节给出维度、已知/未知量、完整AC可行状态、正态密度到似然的推导、共享线路联合拟合、g/R/T/J/评分之间的等式和含义。
该次修订时 EV 从11个地点/曲线/移动组合中，先对每个竞争者重拟合原历史与预测均值，再取最小竞争分数、最大动作分数。
恒定历史均值预测下，早/晚时段置换的理论分数相同；不能把返回某个时段解释为最佳实际时段已找到。
补充了期望与重新拟合不能交换的反例、真实种子51014的评分示例，以及局部参考基准与全域最优基准的区别。
本次只完善文档和交付核验，没有重算正式试验或改变算法、种子、阈值和既有失败分母。

## 三方向推导与两轮小算例（2026-10-12）

先读 three_direction_protocol.md 与 refined_protocol.md。
完整数学稿：symmetric_math.md、power_math.md、bayes_math.md；第二轮为
symmetric_ac_local_math.md、bayes_rb_math.md。原报告第10节汇总符号、
证明、结果和MILP/连续概率模型的后续方案。

    python -m pytest research_experiments/mobile_ev_mle -q -p no:cacheprovider
    python -m research_experiments.mobile_ev_mle.run_three_directions --histories-per-case 3 --continuous --stress
    python -m research_experiments.mobile_ev_mle.collect_three_directions
    python -m research_experiments.mobile_ev_mle.run_refinement
    python -m research_experiments.mobile_ev_mle.collect_refinement
    python -m research_experiments.mobile_ev_mle.confirm_local_actions

94项测试通过。第一轮8个历史（强/弱各3、域外和固定bias各1），每历史
512未来noise；第二轮复用6个原历史、新future noise。两轮实际连续
联合拟合共130次，全部完成；144个局部有界QP全部完成，程序累计计时
约126秒。确认脚本另对局部候选动作执行H+预测未来均值的完整AC竞争
重拟合，六例选择一致，仍只读历史和动作。所有结果保留局部优化限制。

第一轮共同48atom family有比例线路结构和两条预声明latent path，
其穷尽/后验/秩检验只属于该有限族；并非逐边连续MLE的统计保证。
历史cutoff12及0/4/36敏感性中每topology均只有1支持atom，表明初始
距离没有检验充分的参数调整。强案例中两次finite设计排序未迁移到
连续模型；弱案例全部未决，power训练最坏平均分全部0（并列规则恒充）。
strong有限后验exp下溢为0/1，对数质量保留；不说数学概率严格为0。

第二轮局部soft-history QP放开逐边R/X，但30%线路范围和源±.02仅为
局部求解/预测域，不是仪表误差盒或校准置信范围。三个strong案例移动
动作实际局部损失差比恒充大；weak仍未解决，67003代理21.351而实际
宽模型差-.313。后验KL积分8192×3seed选择均late_site0，独立确认对
恒充增益约.0023-.0024 nat，仍没有推出排错。完整AC设计复核保留相同
weak选择，说明参考MLE参数/预测不确定性也须检查，不能仅换求解器。

history_gate用完整finite null历史秩检验：strong剩1、weak剩2；校准
RNG为history seed+80000，gate.seed记录历史标识。alpha_H+alpha_N
联合界仅属于该有限模型。绝对Gaussian历史支持域的chi-square原理
已推导（原始噪声维数，不给MLE后损失套Wilks），尚未作为连续QCQP
实施。MILP支持平面/分段线性搜索和proper连续先验积分留在修订方案。
当前不扩网；先解决历史参数长平坦方向和未来baseline/source预测域。

紧凑证据：evidence/three_direction_summary.json、three_direction_fits.csv、
three_direction_scores.csv、refinement_summary.json、refinement_qp.csv、
refinement_fits.csv、nonlinear_confirmation.json。详细JSON/NPZ保留在
outputs/three_directions与outputs/refinement。原80试验、原阈值和失败
分母未覆盖；delivery_verification.json是先前19页报告的验证快照，
本轮编译与视觉核验另存three_direction_delivery.json。

H中不含本次受控EV；若历史已含它，baseline预测应先扣已知历史控制
轨迹，观测后的AC拟合始终直接使用末端计量总P/Q。
