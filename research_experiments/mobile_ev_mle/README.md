# AC 最大似然拟合与 EV 模型判别设计

本目录已实现新研究原型。旧路线见 ../mobile_ev_probing/README.md。
旧外 LP 的未决不用于判断 MLE 路线。

## 运行入口

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
报告为原docs/mobile_ev_probing_report_v2.tex及13页PDF，旧版本在Git历史中。

精确AC等价构造：无上游/源遥测时，改变公共干线阻抗与下游相位，可由同点逐时源补偿而保持末端P/Q及电压幅值不变。三种构造验证了该声明配置下的不可分离性；总表锚点仍不能自动证明全部线路参数唯一。
