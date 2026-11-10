## 冻结交付余量，2026-11-10

- dr_paper15_delivery_margin/REPORT.md 为模型与结论，model_test.py 与 results.json 为两轴确定性积分，ac_replay.py 与 ac_results.json 为 72 个封存输出的复核。conditional_floor.py 与 conditional_floor.json 另列确知一条交付轨迹时的 90% 条件概率口径。
- 仅读取原 wzzt_dr_p90_stage/run_v1 的 paper15_b2_v0005、价格冻结及已封存观测。共用原设备控制器、背景矩阵与径向积分核。执行范围、SHA-256、库版本、左右端和每次积分点记录在标量结果中；没有新的独立物理观测。

## 最新恢复：90%有界小批全部完成，2026-11-03

- dr_p90_cost/REPORT.md与METHOD_REVIEW.md是结果/数学及文献入口；public evidence/p90_cost_v1仅标量。新源码cost_only.py/run.py使用原Probability、设备、observe和evaluate，不重写辨识器。
- 私有C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_dr_p90_stage：code为冻结执行版本，run_v1的prepare/optimize/qualify/observe/evaluate全部完成；protocol/source_snapshot、3例solver/plans/price_receipt、price_freeze、4096概率、360物理逐条、observations/predictions_frozen、truth_private、summary、audit5472与ended/complete锁齐。不重跑。audit_export.py按私有根读取；第一后处理字段错误单独留档，无物理失败。
- 主3case各12新合成整段背景，共36，配对两law；去重观察204不是360独立样本。paper90价52.1695相对旧53.3829降2.2731%，主11/12、失配10/12，与旧逐路径同；同费轮换5/12。flynn无90起点保留92.1773诊断，均0/12；peng零价全12/12。
- 名义MILP仅旧candidate射线：paper零价、flynn.18981655、peng无名义解零fallback；新工作概率26.98%/3.48%/0，不能称90概率MILP已经实施。只有连续原概率法进行了90降费；规划与实际恢复两种分母分开。
- 下一项如继续是同目标/预算比较条件概率与损失差下分位数求解；后者未实现。原95%三例结论不改；不自动新抽路径/扩网络/强行调资源求正。原设备约束保留，轨迹/RX/u/报价私有。

## 最新恢复：共同误差方法比较已结束，2026-11-03

- 先读dr_joint_loss_methods/REPORT.md、MODEL.md、LITERATURE.md与evidence/joint_loss_methods_v1。不是新生产报价器，原辨识/设备代码未改。
- 私有C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_dr_joint_loss_methods_stage：code为实际运行版本，run_v1含12记录、parts/checkpoint/summary/protocol/source_snapshot/ended，quadratic_v1含6记录与协议/hash/source_snapshot/complete锁。两批结束无待续worker，不重跑。主结果9数学计划、11计算，二阶5计算；保留2次signed-zero重复。
- audit_initial.json保存171项中8个未预锁谱精度门槛失败；audit.json为修正解释后的183项算术/来源核验，不是积分认证。delivery仅修正缓存有符号零及追加回归，不能据此更新旧运行protocol。public明确无轨迹。
- 概率为控制前仿射Gaussian；sphere方向积分保留原L1，quadratic是二阶近似，2/6普通CDF达容差、4/6未决。严格95%可行报价0；零新增AC/参数拟合/W/部署。此前原控制后4096工作模型概率和576AC结果仅作已有对照，不能重复作独立分母。
- 下一项若继续：先核验原设备控制后的联合概率与梯度，二阶近似只产生候选，不直接作为95%认证。需另锁小优化/未看实际路径。全部数学、数值、设备边界见MODEL/REPORT，不自动增预算/噪声/网络。

## 最新完成：无抽样SOCP损失排除条件的固定报价筛查，2026-11-03

入口dr_socp_loss_screen/{README,MODEL,REPORT,PLAN}.md与evidence/socp_loss_screen_v1。复用probability_refinement的3例×4冻结报价，12/12完成、失败0；两符号×两行风险分配共48/48个界，0通过。pengwah四价均零，合计9个算例＋名义计划组合，不作为独立随机实验。抽样/新报价优化/AC/参数拟合/W搜索全部0。8项不同测试、338项确定性重算通过。

- 推导的固定符号SOCP界为均值G−高斯尾部−F翻号上界≥原h，主风险.04/翻号.01。已用当前报价均值符号及固定报价最优行风险分配核验，不只是均分风险失效。candidate三例翻号扣除9689.422/11000.638/4204.509，对应h9.179/14.543/39.314，下界−9631.353/−11012.758/−4237.990，尺度严重失衡。
- paper仿射工作模型精确均值D=86.807+537.980−576.898=47.889；原两候选共享误差在翻号项中抵消，而所提界丢掉正C项、将负F项做逐项高概率上界，破坏抵消。不能删修正以伪造原L1概率保证。此均值是执行控制前的解析值，不是旧AC/控制后均值或新选线收益。
- 原名义计划全部可行，但删除执行控制器后，paper/flynn旧新价最大单槽功率越界概率50%；6非零计划的移位/EV末端高斯能量等式279/279方差为正，严格等式概率为零。不能用旧设备保证支撑无控制高斯模型；储能模式切换未新核验。
- 本轮按预锁门槛未接入新价格优化，不改原3%/物理库存/旧结果。结论仅为这条充分条件在既有报价上过于保守，不证明所有价格或原概率问题无解。若坚持SOCP，先构造能量相容的仿射误差，并保留候选共同误差的抵消；不重复放大此逐行F翻号界。
- 私有wzzt_dr_socp_loss_screen_stage/run_v1保存预检参照精度失败及源码快照；run_v2已complete，协议/hash、12 receipt、48界、末端诊断、summary/audit/ended齐。独立SLSQP参照缩放后通过，kernel不变。公开仅源码、标量、hash，不上传轨迹。旧probability_refinement/run_v1未改；无运行worker。
## 最新恢复：概率可行化与15页算法文档已完成，2026-11-03

- 读dr_probability_refinement/{README,REPORT,PLAN}.md、algorithm/main.tex/main.pdf、evidence/probability_refinement_v1。源码复用原设备控制、条件积分、AC与冻结选择；新的tail.py只作可行化初值，不是新辨识器。模型含方差/相关性/95%事件，但0/3实际条件未找到95可行点。
- 私有C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_dr_probability_refinement_stage/run_v1已结束。protocol/source_snapshot、3例pool/problem/solver_private/plans/price_receipt、price_freeze、actual576、observations_frozen、scores、predictions_frozen、truth_private、scored_private、summary、audit25240、policy_probability_summary与ended/complete锁齐。optimize/observe/evaluate勿重跑；全部未决/迭代限/时间限保留。源码实际运行自private/code，快照/hash未回写。
- 旧wzzt_heterogeneous_pilot_stage/run_v2及其18背景保留。新批72背景、576记录、360不同观察、55296AC点全完成。17不同测试；cost分支用明确合成可行算术问题覆盖，不当作第4个物理算例。后处理初次AST遇Windows GBK解码失败已以UTF-8修复，仅检查命令失败，无实验重跑。
- 三例95规划均未达。paper新主21/24、失配20/24，旧均20/24；同费轮换均3/24。新价88.2076更贵，不建议替换旧53.3829。flynn新旧均0/24；pengwah均24/24零价但历史仍有争议。不能把0/3改成只测试两个争议例。
- 4096同流独立工作概率用于新旧比较，不复用旧128点：paper .922895/.933590，flynn .341636/.361234。价格冻结后才读新实际路径，无统计真实性/恢复保证。完整历史、原噪声、全库存、w2/common3%/.16及R/X固定，0新fit/搜索。
- 文档15页全部目视核验，数学别名已统一，源包包含main/results/figures/BUILD。原始P/Q/V、RX/u/报价/AC只私有，公开白名单无pickle/原轨迹。下一阶段应先用现有数据检查参考预测失配与阈值可达，不自动再扩计算。

## 最新恢复：异质资源实际测试结束，2026-11-02

- 读dr_heterogeneous_pilot/{REPORT,MODEL,README,PLAN,REPAIR_PLAN}.md与evidence/heterogeneous_pilot_v1。原17网络库存被三例小批调用；不再称资源执行尚未接通，不宣称全17网络已测试。
- 私有根C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_heterogeneous_pilot_stage。run_v1仅原规划，superseded_planning引用v2；run_v2有protocol/source_snapshot、case/pool/problem、solver_private/price_receipt/plans、price_freeze、actual144逐次receipt/result、observations_frozen、scores、predictions_frozen、truth_private、scored_private、summary、aggregate_diagnostics、audit18523、ended、complete锁。勿重启已完成optimize/observe。
- 真实执行入口driver.py直接复用原AC和weighted_selection w2/tau3；并非primary_rule旧tau2，也不是新RNJ/W拟合。模型/输出映射从完整历史freeze；原.16保持。devices.py因果控制可还EV分段能量和储能损耗；不同资源Q/P由库给出，非统一.35。
- v1有6个未达标启动；v2唯一附录为planning_repair.py：同一D解析均值只作初值，两次原概率可行化，只有>=95%才硬约束降费。三例最终均未95%，保留诊断价53.3829/72.2039/0。源码记录repair seed=原optimizer_seed+caseindex+30。128概率诊断未回流选价。没有第三次搜索。
- 三例每6新背景、两相关响应律配对；144记录/90不同观察/18独立背景，全部AC和选模完成，0失败未决。paper15主/失配恢复zero2、rotation2、cost_first2、optimized6；flynn全0；pengwah全6且全零动作。同费只保证rotation；cost_first满额费用更低。paper纠错发生在历史候选回选，raw=final，无新结构发现。
- tests.xml（10项）、repair_tests.xml（2项）与独立审计齐；post_audit/summarize/export_results仅只读算术及公开汇总，未重跑AC。公开source_hashes分物理规划源与后处理源；发布文件不能回写旧snapshot。完整轨迹/随机数组/价格/RX留私有，公开PNG/PDF为每条路径标量D图，已目视检查。
- resource库存使用full_inventory，不是旧2kW matched域。用户/EV/储能成本/相关SD为合成假设。背景仍旧三个训练日rank2，资源库5–10%预测误差未用。自动服务回补不等于手动用户参与校准；当前未查线路热限。
- 若新工作：另锁报价和新的独立路径；优先paper15费用/可行化、flynn受限门槛诊断，保留pengwah。当前本批无待续worker，不自动扩大网络/预算或重复已看18日求正。


## 最新恢复：异质DR资源配置完成，尚未接入报价优化

- 读取dr_heterogeneous_resources/{README,REPORT}.md、library.json、case_allocations.json与node_resources.csv。17网络332节点、81旧训练P输入，0新优化/AC/辨识。公开证据evidence/heterogeneous_resources_v1。
- 私有C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_heterogeneous_resources_stage/run_v2：protocol/prepared/ended、configuration_checks、source_snapshot、public齐；run_v1为序列化失败保留，输出库存与v2完全相同，不重复计算当新实验。
- 全库存域与matched_enrollment_kw受限对照域须显式选定；成本最小化无货币预算上界。库存来自预锁历史P尺度及合成设备组成，不是真设备校准，不看错误标签。所有对照用相同资源/能量/价格；保留无奖励及旧零收益例。
- next：复用现有条件积分/原AC，接入异质P/Q和EV/储能状态可行交付；现有逐槽clip不保证能量服务，未实现的接口不能称已支持。原生产默认、原3%与候选参数冻结未改。尚无本配置下的实际效果结果。


## 最新恢复：饱和响应配对批已结束，2026-11-02

- dr_saturation/{MODEL,REPORT,README}.md，evidence/saturation_replay_v1；代码8d321847。现有dr_spatial_correlated/ac_bridge增加可选delivery_rule=device_clip，原默认unbounded不变，5新增+8原桥接纯测全过。
- 私有根C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_saturation_stage。src/runner.py的prepare/observe/evaluate已经全部完成run_v1，post/audit.py只读重放，post/export_and_plot.py只发显式标量。勿重新调用observe；无未完任务。
- run_v1协议SHA d551c8520f49c32b374c7cee2c5f4a67ced09e703d3c5d90aecf6d2133fd0ac1；原价格SHA cd6d531bef1147f687c2b8efae591f8c544f8ca76a7f6c05ee7567f4de8c0cf3逐字相同。problem_frozen引用spatial_correlated_stage/run_v1/problem.pkl，不能假定本地有problem.pkl。
- source_snapshot224文件、parent旧144路径引用、观察/预测freeze、scored、summary、audit5652、ended及complete锁齐。evaluation.log保存清单尚未完成时的失败；evaluation_completed.log为随后只重试评分成功，未重跑AC。已完成144/144及13824AC时点，不重抽。
- 开放主9/12、失配8/12恢复与截断前逐条同，费用44.9505；调节27.57→26.82及25.06→24.86kWh。固定两点主9→8是唯一输出变化。严格设备与总2kW界均满足；原同费轮换恢复0/12、同配置4/12。配对两law共享12背景，不能算24独立背景。
- public_release为配置/标量/图/hash白名单，PNG已目视核查；完整price、P/Q/V、u、RX、Z和AC轨迹不上传。后处理源与物理源快照分别保存，不据报告回改运行hash。
- 当前未优化饱和价，旧95%不沿用；数学可通过条件化Z复用一维L1积分。要继续必须保留原规则并另冻结新价/新独立路径，当前已看12路径只作开发。用户未授权扩大算例。


## 最新恢复：开放节点相关DR小批结束，2026-11-02

- 最新入口dr_spatial_correlated/{MODEL,REPORT,README}.md；evidence/spatial_correlated_v2。源码提交1c3242a2及0ce1dc02。31原纯测+6约束修复纯测=37（另7子项）；两次运行同测试不叠加分母。
- 私有根C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_spatial_correlated_stage。src是原冻结实现，refinement是读未来前的附录；audit/post是只读复算/导图，勿用后处理源码追改旧manifest。
- run_v1保存原4起点有限罚规划及price_freeze、planning_superseded，无实际未来；不重启原优化。run_v2经refinement/refined_runner.py prepare/run另冻，引用run_v1/problem.pkl；不能假定run_v2有本地problem.pkl。原src/runner.py observe/evaluate在v2执行144记录，已ended。所有原失败/墙限/迭代限保留。
- run_v2协议SHA a2e4ee23d2f97456e10f47070637c41e9138e2de677768b861d17051370c41ca；价格SHA cd6d531bef1147f687c2b8efae591f8c544f8ca76a7f6c05ee7567f4de8c0cf3。source_snapshot/protocol/problem_frozen/price_freeze/observations_frozen/predictions_frozen/scored_private/summary/ended齐。actual/dayXX_law_policy有receipt及原result.pkl；scores保存完整原weighted_selection输出。
- 12新背景×6方案×2配对law=144/144；独立背景仍12。主恢复0/8/9/9/0/4、失配0/7/8/8/0/4（zero/旧两点/固定相关/开放/均匀轮换/同配置轮换）。开放费44.9505，与固定91.4457逐日错误相同；实际设备越界主12/12、失配9/12，不能忽略。同费轮换0越界。全部AC收敛、原3%与冻结映射已核。
- audit/audit_results.py --run run_v2只读重放，7794项通过；policy_actual_metrics和resource_excursion_summary给执行总量及越界。public_figures/comprehensive含计划u仅私有，aggregate带未裁剪越界脚注可公开。public_release为显式标量白名单，不上传轨迹/完整价格。
- 无待续批次。不再读新日或重复优化；下一次需另锁设备可交付约束/饱和与误差校准方案，之后才考虑用户暂缓的跨场景评价。

## 最新恢复：条件连续DR的AC批已结束，2026-11-02

- dr_conditional_ac/{REPORT,README}.md与evidence/conditional_ac_v1为最新入口；核心代码2415e10a。12新背景×5策略60AC日全部完成，四噪声共240选模，不重跑、不补seed。
- 私有根C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_conditional_ac_stage，run_v1/{protocol,price_freeze,source_snapshot,observations_frozen,predictions_frozen,scored_private,summary,ended}齐。每个actual/dayXX_policy含receipt和原始result.pkl，scores保存原完整weighted_score。协议SHA4758419f45fe60596d174406fe345ee27c34e7662377a81d38663691324009f6。
- source_snapshot是物理执行前冻结闭包；post_analysis/artifacts/audit属完成后的只读汇总，不要据此更新旧协议hash。src执行版本保持原样；公开源和配置足以定位，私有P/Q/V与价格/随机数组不上传。
- 高价continuous/rotation费用81.0211、低价47.20046来自旧条件积分price_freeze，未重新优化。主结果both中2/10/7/6/5恢复（zero/连续高/轮换高/连续低/轮换低，各12），有实际AC选模收益但95%未证；完整历史和所有R/X保持冻结，无新W搜索。
- audit_summary核验240次直接评分/噪声/源hash/费用；load_audit55、post_analysis1271检查；repo/pure测试各同6项，不合计为12独立测试。图已渲染核看。
- 当前没有未完数字批。后续需要新的明确协议及信息时序，先改进规划背景/误差参考再冻结独立验证；不能将当前12日当未看测试、不能改3%或删资源越界来维持收益。

## 最新恢复：条件积分DR两批均结束，2026-11-02

- 先读dr_conditional_gaussian/REPORT.md、MODEL.md；公开evidence/conditional_gaussian_v1含两批protocol、汇总、执行源zip、审计与效果图。实现0721a1ef。
- 私有根C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_conditional_dr_stage；run_v1主批、delivery_v1降额诊断均有ended.json，无待续worker。各2000独立整日；6000/8000观察选模，不重复生成。
- run_v1原执行源与当前修复后源码不同，仅无效条件因子guard/固定零项边界；executed_v1及两个source_snapshot.zip保留。严守各protocol hash，不直接拿当前源码重跑旧phase或覆盖原协议。src/是现行纯测通过版本。
- audit_v1.json 212项、audit_delivery_v1.json 286项；guarded_amplitude_plan_v1.json为解析降额而非新激励优化；price_freeze/evaluation_draws/evaluation_scalars/predictions_frozen/summary齐。P/Q/Y、R/X、价格与随机轨迹不得上传。
- 95%排除未达；原高强度全部越界，降额有条件选模收益。下一有界问题见REPORT末段，不开新网络/新种子扩批维持正结果。

## 最新恢复：Gaussian裕量SOCP批已收口

- dr_gaussian_margin/REPORT.md；公开evidence/gaussian_margin_v1。私有wzzt_gaussian_margin_stage/run_v1，protocol SHA 3599a32c62f6d33e517a25f07316a576d330a1df7fba1e366a4b509d512f2709; problem/源zip/价冻结/三求解/2000共享路径/8000选模/ended齐，所有任务完成。
- 当前正式corrected报价不可行；naive仅诊断，max及同费rotation所有路径资源越界。不得转旧worker当作新AC交付。没有自动新AC待启动。
- audit_history25、audit_solver26、audit_evaluation67、audit_labels16008；cost_bound给naive近最优下界。完整源码和失败回执保存。Gaussian预测仅3日rank2且708/2000自然负P，不能称实证校准。
- 本轮新合同及scope见MODEL.md。保留全历史/零弱参数/frozen输出；真块只后评。不得复用此2000路径调价后再称独立。

## 最新恢复：独立AC报价效果测试已关闭

- 先读 `dr_day_effect/REPORT.md`，公共 `evidence/day_effect_actual_v1`含完整9规则、12日标量、所需/实际gap、CI/p、机制和图。主zero及两opt各6/12恢复、总6错，两个匹配轮换5/12总7错；只避免一次轮换伤害，无主规则相对zero新恢复。预设w1敏感性低价有2真纠错，不能改写主结果。
- 私有根wzzt_day_effect_stage/actual_v1：protocol SHA f1e0f5ac0e798481888600c3f37b57aa001efd0a83dce7a5169df6b932659ee7；freeze、source_snapshot261、requests60、preflight、observations_frozen、predictions_frozen、scored、summary_private、ended齐。12独立source/case/profile/replicate新身份、5策略CRN，60/60成功5760 AC点，失败/重抽0。勿重跑。
- 方案设计源提交a212f761(worker显式结算/zero修复)、6237518b(driver)，精确运行字节以zip/hash为准。报价完全冻结、原288/288、R/X/.16/newday2%w2不变，无新fit/价格solve/W接受搜索。只有原5中1个规划资格case执行，其他4保持原不可行/未决。
- audit_selection_v1预审856、actual_receipt7194；audit_actual_v1/results_v1/receipt5017及mechanism_rms_public_candidate。先完整观测选模后truth；meta source/gates/费用/4角/AC/CRN/9规则/真假块均独立核过。branch_ampacity_checked=false，勿叫完整工程安全验证。
- public_release_v2和public_figures_v3为发布；plot_failure_v1/public_figures_v1保留WindowsGBK的SVG后处理失败，只重绘没重模拟。低费ΔM12/12正但原6误日缺口未补足；旧残差迁移RMS .536116远大于transport差.004538，先改善预测残差与需求gap，不简单增DR随机维度或刷批。
- 当前所有worker结束，run_lock complete，恢复只读结果，无待续计算。P/Q/V/AC/整日控制及R/X留私有；公开白名单可重绘。下一批须新锁规划/规则/种子/物理身份，已看测试不能重称首次独立。

## 最新恢复：无支付上限24h最低费用模型结束

- 先读 `dr_day_curve/REPORT.md`、MODEL.md，主配置budget=None，费用只作目标/输出；原1.6批次见CAPPED_PILOT_REPORT及capped_summary，勿混淆两域结论。
- 私有根wzzt_day_curve_stage：model_pilot_uncapped_v2含5个原条件的冻结协议/源zip/raw/修后/费用与模型评价，ended已完成；audit_uncapped_v2为1108独立复核，uncapped_export_v1为标量和原单位矩阵审核。所有solver结束，不重跑。
- paper新23/24费.5831下界0；旧可行witness新域24/24费.228748构成更好上界，不冒称新返回值。validation仅86/256与zero相同；不是实际90%已证。b4/b7无货币约束目标数值不可行，b5/peng未决，诊断高费用不作所需预算。
- 原model_pilot_v1、paper_adapter_retry_v1、combined_evaluation_v1/v2、audit_v1及源码zip独立保留。发布版runner先存raw再适配，旧运行以快照字节为准；原raw丢失限制和7.81e-12kW恢复单重试不覆盖。
- 发布版28单元过；公开代码/标量/源码hash，u及P/Q/V/AC/完整R/X私有。无上限DTO不能直接交给仍固定1.6的旧物理worker。后续应先锁有效跨日误差/响应模型，非继续增加费用；当前无待续实际试验。

## 最新恢复：DR距离与时长敏感性图表全部完成

- 先读 `dr_signal_analysis/REPORT.md`、`README.md`，公共 `evidence/signal_analysis_v1`。新增仅图表和固定模型反事实算术，无新增AC、fit、LP/MILP、价格优化或实际观测。旧实际数据独立分母19/20；新240设置行228有效不能当228独立日。
- 私有 `wzzt_dr_visual_stage`：scalar_extract_v1与visual_release_v1保存标量/绘图；dose_run_v1保存广播错误；dose_run_v2完成246内审与5296独审；working_law_v1复用原每例1000工作场景、192核验。installed_tests_v2.xml为16/16通过。协议/源码/结果hash见公开audit_summary和manifest。
- D为同输入两冻结模型L1平均预测距离，DeltaD是前后范数之差；g90取1000增量降序第900个，gamma只为原激励幅值目标。实际M=0正反例和旧主最终DR净0保持。
- 原2h/4h/8h平均DeltaD7.1243/9.6539/11.0238微单位，同1.6预算；6h/8h旧工作场景原gamma五例均1000/1000，仍非独立长时交付。12h部分下降，不据曲线改主方案。alpha sqrt2需3.2预算，单列超原预算。
- 五图PNG/SVG/PDF可用公开JSON直接重绘；通常无需重做私有核算。新长时动作尚未实际运行；若继续，应历史选时形与价格、冻结后新独立核验，保留旧2h和无奖励。所有当前批结束无worker，原资源失败日不填0、不重抽。

## 最新恢复：新日样本加权五条件测试全部结束

- 先读 `dr_newday_weighted/REPORT.md`、`README.md`；公开 `evidence/newday_weighted_v1`含完整九规则结果、正反例和费用分列。主w2/tau2额外DR收益0，部分预锁敏感性有真实纠错；不将敏感性替代主版本。
- 私有根 `C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a\wzzt_newday_weighted_v1`：run_v1为未部署raw-pair历史诊断，run_v2为唯一实际新日测试；不可混成两批独立试验。v2复用v1前2份完全相同问题报价，另3对改为真正不同的最终结构；原参数均历史原样。
- 实验源码0248a357、38bf5dd4、5c0b9d21；精确字节以run_v2/source_snapshot.zip为准。protocol SHAc4f74eafdafca1e017818d66d839eb9b6eb57677ad89ac8272e9b15017fef288；price_freeze SHA534153bf3c7bb8d44bf29f71ea5491ab916b70dd90cdc5a25d883899423e5592；snapshot SHAc03e8a2cbaab153fc634dcd142ca0dbc66b254647e5a3abb943e11a09a534416。
- 入口controller prepare/holdout/observe/score；全部已结束，勿重跑旧记录。57有效/60记录、19有效/20条件日、5472AC时点，新增fit/LP/W0。b7/path1540三策略前置资源资格失败，无AC；未决保留，不能靠填0、降库存或重抽补成60成功。
- 所有原288训练/288验证、R/X、原.16图固定。评分只改新日样本w1/2/4与新日tau3/2/1%，历史权重1与3%不变；原未门控累计3%单列。物理数据只生成一次，九规则重评分不重新仿真。
- 主要数字：主三策略各6错/13恢复/19有效，原三策略8错/11恢复；主DR净0。w1tau2总错8/8/7，paper1501粗族裕量-.570949/-.433748/+.407332；反例w2tau1总错3/1/2且flynn1512轮换距离负而纠，设计距离正未纠。完整九规则读rules.json，不择优报告。
- 审计run_v2/independent_planning_audit.json4951、independent_selection_audit.json10730、independent_physical_audit_v2.json9753均通过。首个independent_physical_audit.json为schema读取失败回执，勿当成功。补充independent_physical_supplement.json分有效/未交付费用；失败模型账项不是交付支付。所有审计新增solve/AC/fit均0。
- 源码已最小参数化旧one_day_worker的协议绑定nodes/noise，默认旧.0005/201/213保留；旧frozen_pool增加可选competition_selector、默认旧选择保留。旧实验当前文件hash可能已变，精确重放需各自snapshot/commit恢复，勿改历史hash；本轮外部审计后归档脚本不冒称事前核心代码。
- 下一项实质问题：价格需直接针对实际固定模型排除/选择而非仅D，误差生成依据需明确；柔性容量要与自然负荷联合约束，zero应有不依赖柔性资格的物理路径。当前批不追加参数/seed或重跑失败日。公开白名单无P/Q/V/AC/完整R/X，原家庭和聚合轨迹留私有。

## 最新恢复：冻结参数距离DR小测试全部结束

- 先读 `dr_frozen_distance/REPORT.md`、`README.md`，公开 `evidence/frozen_distance_v1/{configuration,planning,summary,physical_distance,audit_summary,manifest}.json`。已完成12/12物理记录、4独立日、1152AC；没有正在运行或未完成的新实验，勿重跑。
- 实验源码阶段 `c58d2631`（初版）和 `caf68f20`（可行初值/冻结凭证）；实际精确源码以run_v2/source_snapshot.zip与protocol.source_sha256为准。后续外部审计/报告源码不冒称事前运行逻辑。
- 私有根 `C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a\wzzt_frozen_distance_v1`。run_v1为60秒零初值未实际执行；run_v2价格先冻结后1000工作留出及4新日。两个版本problem/draws/affine scenarios逐字节相同，不重复算作两个独立科学条件。
- run_v2/price_freeze SHA407f1ea4ca24df648dbb9bd0774fca715be2c3c505de4c27f2d35dd90444c2df；protocol SHAaf607d4c2760c4a8195ad5b1830a2150da43e29f8df41643a369a1d62a5741e5。环境environment.json；参数frozen_pool.json、物理problem.json均私有勿上传。
- main288历史验证＋96新日；历史训练288只供已存R/X，未改权重。选池仅历史已选点19和同路径下一接受点20；新日完整RNJ＋W重跑为0。原规则select/contract直接复用，不更改原生产默认。raw/output粗21/21、细22/22。
- 结果：设计DR4/4扩大预测距离且物理P/Q核验同向；各策略最终均1遗漏/0恢复/0累计排除。1000工作留出达冻结gamma846/1000，90%幅值目标失败。主求解TIME_LIMIT、返回价恰为最佳初值，无连续精调或全局最优结论。
- 审计run_v2/external_gate_before_score.json268、independent_price_audit.json979、independent_score_audit.json755、independent_physical_audit.json1332均过。核心输入/源码不变，审计新增求解和AC均0。审计脚本已归档研究模块；公开白名单导出evidence/frozen_distance_v1/export_public.py只导标量，不读出私有参数数组。
- 后续先从M384=.75Mh+.25Mday与所需新日M>29.085610微单位检查可购买的排除空间；不能继续凭距离增量改写为纠错收益，也不以超时未达标否定所有DR。旧两零收益对照与本例失败完整保留。

## 最新恢复：连续价格与独立实际测试全部结束

- 先读 `dr_validation_excitation/{CONTINUOUS_MODEL,CONTINUOUS_REPORT,CONTINUOUS_PLAN}.md`、`evidence/validation_continuous_v1`。原3%/.16、资源/1.6预算、完整历史、误差及零弱权均未改。原实际全正确，但没有DR净收益；保留历史族的命名组合审计全部漏1，不能混用两口径。
- 私有主根 `C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a\wzzt_validation_continuous_v1`。
- `plan_v1`：13不同训练报价×10场景×17全族=2210LP；原all-noC强条件和Powell到评价上限均保留。`coarse_reprice_v2`：同13表重求当前F0精确退出报价，新增LP0；其price_selection SHA317ba149694d9d37e2dfa246923645d87e71b5f9b668637512303c2111ec37f9。合计56个价格求解阶段全status2；不称全域价格最优。
- `model_holdout_v2`：seed2026110132、10新工作模型路径，3种不同训练报价510LP，四策略均0/10；与真实AC背景不同，不校准现场概率。价格未据留出更新。
- `actual_v1`：先price_freeze/source_snapshot再8未来观测，4策略×2路径1200/1201；observe、两个fit worker、finish、aggregate全结束。8/8拟合824LP，原源与算法hash一致；32事件3072AC点，原480/480；全最终0错/原墙限。源冻结28d1b0e970d159852ed2a224152cf299d70c5c9d539fc45113f35bcb468b1d7e（price_freeze）；actual protocol f8926e94e25de85be00ed92d3cb5d87aa55a4e62543bc7399a9de2bc1c3d32d2。不要重跑，也不据已看未来重选价格。
- `retained_history_v1`：同8观察再加17历史族共同训练136LP，同原规则组合审计，8/8漏C且F0合格。原池F0出现0/8；历史全冻结预测器也8/8漏C。不是生产默认修复，未更改报价或原实际0错结果；原/追加/frozen三个selector及loss均独立重放过。该阶段源码hash事前锁定，source_snapshot在完成后归档且精确匹配，不冒称事前zip。
- 私有审计：`planning_audit.json`36279、`audit_actual_v1.json`812、`audit_retained_history_v1.json`1777全过；`audit_actual_mechanism_v1.json`明确path0准备初值修正2/8，16次接受未最终入选。独立脚本在visualization根或wzzt_validation_continuous_stage，公开证据附源码，轨迹/连续参数私有。
- 所有solver/worker已结束，无待恢复计算。若续研先解决“规划固定历史族与实际生成新族”一致性；可以明确比较保留历史族的算法版本，但不得无说明改默认或再换阈值制造收益。90%仍是激励条件概率，当前没有达标报价或统计恢复保证。


## 最新恢复：全族重调权后的3%排除审查已结束

- 先读 `dr_validation_excitation/{MODEL,REPORT,README}.md`、公开 `evidence/validation_excitation_v1`。没有新全局连续价格优化器或独立新价格实际效果，原3%/.16/接受/完整历史/预算/零弱权未改。
- 私有 `C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a\wzzt_validation_margin_v1`：`run_v1`（16训LP＋8面探针）、`run_v2_dual`（16训LP）、`mean_forecast_gate_v1`（2训LP）、`noise_scenario_gate_v1`（80训LP，10场景×4价）均已有summary。前两阶段另有ended；后两summary已complete。全部结束，不重跑。
- 原实际源仍为 `wzzt_conic_effects_v1/run_v1/main` 的8份观察，不是8新实际事件。新预测场景不读实际未来/真拓扑。两路径全族粗/细验证排除m仍负，接受/收缩裕量全正；当前价格未实现纠错。面探针8/8墙限保留，不用其宽松界宣称不确定性；精确对偶面16数值核验通过只覆盖声明域。
- 均值未来Y免费m=+8.374e−6，与实际免费−8.253/−9.331e−6相反。保留完整残差后的有限价格门控四者均0/10；代码使用返回最优点必要筛查，若有正点仍须最优面核验。没有新可行报价，不启动新物理或完整辨识。
- `wzzt_validation_margin_stage`保存4单元XML、MODEL/REPORT、导出审计脚本与public_evidence。独立261检查全过，0额外solver；原私有参数只用于重放不导出。四阶段`source_snapshot_recovered.zip/source_recovery.json`精确匹配协议5/5、6/6、46/46、46/46；来源当前未变字节或b0d713ca，恢复时点为事后。可据协议hash恢复而不换hash重跑。
- 下一实质工作是同病例/原资源域内对新验证排除条件做有界连续价格搜索，包含theta(c)重调权项，候选价逐一原LP复算。仅找到可行响应且通过原门槛时再冻独立实际核算；未找到就保留未找到，不改3%或用固定参数梯度替代最终条件。当前4报价失败不代表整个DR/连续价格域无效。


## 最新恢复：连续DR实际效果批已全部结束，无重跑项目

- 主报告 `dr_conic_pricing/effect_test/ACTUAL_TEST_REPORT.md`，事前 `PLAN.md`；公开 `evidence/conic_effects_v1/{summary,independent_audit,manifest}.json`。源码首阶段5838911f，原SOCP/MISOCP42c03737；最终记录见后续阶段提交。
- 私有根 `C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a\wzzt_conic_effects_v1\run_v1`。`price_freeze.json` SHA91bab2aa043802f470d7f1bea26cfc4eb5dcbb311cfaeb31b8d65355cf456a7c；`source_snapshot.zip` SHA25a6209527fed5aa86309d3c95af9a8742dceb834102df4cbf36a48c84912e01。
- 两例main/zero_control各protocol/problem/4plans、4报价求解回执、规划moment检查2000，test/records/physical及fit_cache/fits、worker_0/1_ended、fits_frozen、scored、summary全齐。16/16不同完整拟合、1511LP、64事件6144AC均完成；没有失败或待跑worker。完整历史288/288→480/480，不重跑旧基线或本16记录。
- 主例zero/rotation/SOCP/MISOCP最终同[1,1]、0/2恢复；对照同[0,0]、2/2恢复。付费相对zero与简单轮换均未减少最终错漏。主例价法提高固定参考激励，但freezero实际2/2且模型内2000/2000也达同阈值，现指标不支持纠错收益。
- 主例6/8得到且验证合格的正确候选，21粗族在3%带内仍回选。33accept/9cross/11删除未转化DR收益；收缩未丢正确解；16/16有墙限。不同辨识方法的自身修正收益与DR收益分开。
- 冻结runtime为prepare.py/physical_worker.py/driver.py和原audit.py，不能改后硬套协议恢复。audit_v1_schema_failure.json原错误保留；后验audit_v2.py只修预算字段读request，audit_v2.json SHA d8dcce2195a455abdb8edaf01f4f5ff34f25c143f0b6ff256e9400b2f40dc528，16/16独立审核过；physical_independent_audit_v1.json也留存。
- report.py是只读导出；public_summary.json旧v1与public_summary_v3.json均保留，v2增加模型配对zero达标数，v3核验两条件共享2个自然背景、4个响应流，没有重算价格/AC/W。公开汇总只复制v3，不把任何P/Q/V、物理时序或拟合R/X上传。后验电压信号/误差比仅描述诊断，非噪声消融。
- 这些仅是连续DR、合成响应乘子相关.8；无PV新增、无EV实测/行为校准。规划六日因子近似不等同实际producer。概率只针对激励，非恢复；两实际路径不提供90%总体认证。
- 下一项先审查激励条件与最终选模的一致性，需要有具体新差异再锁有界实验；不自动补种子、扩大网络或改噪声/预算找正。本批和所有旧批均保持结束。


- 独立性口径：4个case-path分层记录中只有2个不同自然背景，不能合称4条iid自然路径；各条件内的两条路径保持固定分母。background_reuse_audit.json保存数组哈希证据。

## 最新恢复：连续DR SOCP/MISOCP实现已完成，无后台批次

- 先读 `dr_conic_pricing/{README,MODEL,IMPLEMENTATION_REPORT}.md` 与 `evidence/conic_pricing_v1/{summary.json,audit_delivery.py}`。新主模型90%指激励达标；不使用Copula，不是90%完整恢复，也没有新增AC/W恢复试验。
- 公共入口：`PriceProblem.from_dict`、`solve_price(...mode='socp'|'misocp')`；CLI `python -m research_experiments.wzzt_dr_selection.dr_conic_pricing.cli`。SOCP有互斥类型时必须显式allowed_contracts；MISOCP自动选择，无强制分散。参数和概率模型见完整JSON表。
- 最终回归81/81（56新＋25兼容），MISOCP有全部9种固定类型SOCP参照。最终四正式run均两阶段最优/重算过，四时段各20000和96时段DR＋EV各2000抽样全达标/0设备预算违约；只属合成求解资格，不是实际拓扑收益。
- 私有根 `C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a\dr_conic_pricing_stage`：`release_socp`、`release_misocp`、`release_socp_day`、`release_misocp_day` 保存protocol/problem/optimization/validation/ended；`final_regression_v4.xml`为最终81项；`delivery_summary.json`为独立重算。
- 源码zip `release_source_snapshot.zip` SHA256 `2ccb82e0eef33e67d8b0b767a4288370edda742fda735f2c21822435be5c2853`。旧同根socp_run_v1/*_final、integrated_tests_v1/v2、final_regression_v3及misocp_numeric_diagnosis_v1保留数值修复过程；不把旧随机流再验证算新增独立样本。
- 无进程/锁/待续solver；每个CLI新目录独占，旧结果不覆盖。精确数据/源变更应另冻协议。Gurobi13.0.3，NumericFocus2、NonConvex0、BarQCPConvTol1e-8，硬科学约束未为数值修复放宽；独立未平方检查及失败回退必须保留。
- `resources.py`复用原ResourceDay/DayModule，输出P后Q、各time-major。实际幅值可直接生成增量；只加一次资源增量，kW/kvar需显式标幺换算，电压仍由原AC/表计提供。`identification.fit_full_history`追加全部观测并调用原fit_observed，未重写辨识。
- 下一项是锁定真实历史争议块对应g/β、预测共同因子与实际响应检验，再做独立最终错漏/费用比较。本次实现不能冒充已有新WZZT价格收益。所有旧价格/AC/W批已收口，勿据本文件重跑。


## 最新完成：目标漏块预计算与报价优化，独立选线收益未通过（2026-10-31）

本节覆盖下方旧“当前运行”。入口dr_targeted_pricing/ACTUAL_TEST_REPORT.md、MATHEMATICAL_MODEL.md与evidence/targeted_pricing_v1。本轮没有新的待跑批次。

- 从既有V=.0005的3/40最终失败中资格复核；paper15/b2完整错1但bank已0，保留不做DR。主flynn16/b4当前157LP、原288/288下bank/单路径/full均漏C=201–203。原两个零收益控制继续保留，不当新独立重复。
- 两完整任务（同步反向/简单分时轮换）、201/204、.2kW/.4kWh、0/1/2报价与整段参与0/.5/.75。四独立训验报价，硬预算.4sum(c)<=1.6，100合法联合列。历史全部保留，四新日同样追加480/480；原.0005/.005、25秒/3%/.16与零弱R/X不改。
- planning32请求/31unique/3303LP；独立test20请求/15unique/1568LP，合46完整调用4871LP、失败未决0。208物理事件19968AC时点，保存运行/预算检查过；ampacity未核。46unique均至少一条墙限，88接受2cross，均未当最终收益。
- 真实planning targeted16格全错1；rotation训练201执行时8格错0，其余错1，Gamma均0。单规划背景下最优rotation训练(2,0)/验证(0,0)，风险.25、期望费.6；同自由度优化轮换相同，非W独有结构设计。
- 冻结后4独立路径：zero与统一低价rotation均[0,1,0,0]，3/4恢复、均错.25；优化[0,1,0,1]，2/4恢复、均错.5。均实际费0/.7/.6；原绑定报价同2/4但费1.4。成功路径100/102本来zero已正，101未修，103恶化；不能称DR价格有效。
- 机制：planning成功依赖有限额外初值准入，粗21族仍3%合格；补回已存粗拟合点会8/8重新漏C（只读反事实）。test103 W确实补C、增益过实际接受门槛，但粗21仍3%合格，最后回选粗族；原bank/单路径同观察0错。最终收缩不是丢失环节。联合算法本身优于几何方法的证据与本次价格失败分开。
- 独立24857检查0差异，0新LP/MILP/AC；另保存点机制审查。公开只白名单标量/离散支持/hash，原PQV/AC/连续权重全私有。主case是已知错误块、已知生成模型的一背景参照，不是部署参考分布或行为校准。
- 本轮有界任务结束，不补种子/预算/报价/网络求正。下一步先解决最终选模目标一致性与自然/计量参考依据；没有可检验的新差异，不启动新物理或辨识批。

## 当前恢复：targeted_pricing_v1/flynn_run_v1实际规划拟合进行中

- 先读dr_targeted_pricing/MODEL_AND_PLAN.md。两个baseline已结束不重跑：paper15 run_v1免费bank已正确，仅资格；flynn_run_v1 full/bank均漏201-203，主机制例。
- flynn_run_v1/protocol.json/source_snapshot_targeted.zip/32物理请求31jobs已冻结；planning两个fit worker正在执行，脚本run_precomputed.py，命令fit --phase planning --worker 0/1。使用旧run_day_trial的fit/finish，不改识别器。
- 接下来run_precomputed finish planning→decision_layer optimize（四独立报价）→decision_layer test→run_precomputed fit test两个worker→finish test→aggregate→独立audit。不要修改冻结source以绕hash守卫。
- baseline选择在DR前锁为三例既有.0005失败中唯一bank也错的flynn b4；标准噪声.0002原40例均正确。仅201/204资源，active events0/2，其他两新日自然，全部480/480；预算1.6、原3%/.16不变。
- 源噪声继承worker已参数化，旧实验3ca版本在旧source zip，不能将新source当旧协议resume。当前actualdriver源码与risk模型全部入新protocol sourcehash。原physical basis收据价格1，test须用冻结quote的另plan_packet重算费用。

## 最新恢复：day_module_actual_v1全阶段已结束，不重跑

- 入口dr_day_modules/ACTUAL_TEST_REPORT.md。48请求/41unique fit/9565LP/18432AC时点全结束，无失败未决；planning18、test30（23unique）。oracle正式zero、paid-only rotation_010，test先锁后开，raw/final全0。
- private scratch/wzzt_day_module_actual_v1/run_v1：protocol SHA 26cef82997ecfa871f4f9b1d0208614f6a7fadfb901096ca91c13ecbf148b157、source_snapshot.zip、plans、planning/test的observations/worker_ended/fits_frozen/scored、oracle、summary。source d42250fe对应冻结驱动；不要改源hash后冒充resume或重跑终态。
- 同级audit_independent_report_v2.json/execution_v2/repair_v2及v1原失败完整保留。v2源码d4d34408...、报告8fa0ea7c...，18007检查0差异；repo audit_day_trial.py可在新out路径只读重核，不新增AC/fit。公开evidence/day_module_actual_v1为标量白名单，家庭/节点PQV/AC/连续参数私有。
- 完整路径与Gamma结果均为当前固定单控制：一条未干预背景、一个条件4格背景。40/41有原路径墙限，不称全树最优。几何RNJ额外假块由现有族库选模修正，完整W最终未选新路径；不改记为DR恢复收益。
- 原两例继续零收益对照，不扩该例费用/种子求正。新协议必须有新的历史依据与响应律边界，旧NEXT下方只作已执行计划记录。当前无研究进程和待恢复数值阶段，授权Git目的地仍为github.com/tutu-er/Topo任务分支。

## 最新恢复：全天模块/历史输入阶段结束

- 先读dr_day_modules/{README,MODEL,HISTORY_REPORT,RESOURCE_REPORT,ACN_DATA_REPORT,AMPLITUDE_NOTE,NEXT}.md。只实现历史审计/资源运行域；尚无全天价格优化器、新AC/完整W恢复结果。旧两例作为零收益对照保留。
- 历史审计e092cb91，40/40源、80hash、560窗口/60620对已结束，不重跑。2x双通道全天0/4330、全历史夜间19/4330。该标准严格；无近比例不否定多节点补偿。未读真实结构/V/未来，未完成历史竞争关联。
- resource源码3855b348.../test6d8d2ecb...，完整29检查通过。v3局部6测1LP，完整回归7LP（6最优1预设不可行）分别保存，不累计成样本。AMPLITUDE_NOTE数学经独立审查已收紧正确族噪声底条件，固定历史反例保留。
- private scratch/wzzt_day_modules_v1: history_audit_v1的protocol/progress/ended/cases/aggregate及audit_independent_v1，full_stage_regression.xml；wzzt_day_schedule_resource_audit_v3含previous_v2/source_snapshot_v3/receipt/semantic_summary；wzzt_acn_public_session_probe_v1含官方5会话metadata原响应（私有）及public_schema_summary。
- 公开evidence/day_history_input_v1、day_resource_modules_v1、acn_session_qualification_v1均有manifest；不上传输入轨迹/会话/认证。新资源桥未核验一次注入/多日总预算，整段rho由调用方提供且无参与校准。下次先按NEXT锁最小资源/物理批，保留所有真实任务超能力/断连/失败；不得直接运行旧价格panel或把运行LP称价格优化。
- fit_observed、SharedFitter与旧物理默认没有因该阶段改动；显式Method2缓存身份仍可用。Git目的地github.com/tutu-er/Topo任务分支已授权，旧阻断不再适用。当前无研究数值进程，不创建自动重复。

## 收尾恢复：全部本轮有界阶段已结束

- 最终入口BUDGET_REFERENCE_REPORT.md（原参照）、NUMERICAL_REPLAY_REPORT.md（2数值重放）、CACHE_METHOD_REPORT.md（缓存安全）。原source版本c0282288、数值API af24ba98，精确原字节zip保留；缓存修复是下一独立版本。
- 全部研究数值与审计均结束。不要重启原32+32、3443LP、4LP探针或2fit188LP；原30失败、flynn参照未决、pengwah免费选择保留。零收益结论限于本两例。
- direct与SharedFitter均显式记录方法，缓存隔离修复12测试通过，0研究调用；累计25不同单元/回归检查。无新增家庭、PQV、AC轨迹公开。

## 数值重放恢复：2/2已结束，不重跑

- 入口dr_balanced_design/NUMERICAL_REPLAY_REPORT.md、numerical_full_replay.py；private run_v1/numerical_full_replay_v1的2fit/188LP均结束，score五方法0错。无新独立样本/AC/价格。
- 协议adb7c4df...、source_snapshot_new_config.zip 45源码SHA a4db539944222007a551c26034e57625586ae9849e4a7880cfe114822da90b17；还原时以zip原字节为准，不将新hash硬套旧记录。
- API default lp_method1仍保留，Method2仅两显式直接调用；原64批30失败、oracle及费用仍独立封存。公开evidence/numerical_lp_configuration_v1无轨迹/连续参数。
- 下一项仅SharedFitter缓存方法身份修正与假函数检查，防止方法1/2混用cache；不新增DR或完整拟合。

## 最新恢复：原预算内参照批已全部结束

- 先读dr_balanced_design/BUDGET_REFERENCE_REPORT.md与evidence/budget_reference_diagnostic_v1。原64未来＋2历史、66unique、3443LP均封存；未来34完成30首LP状态9。测试23/32未决；不要把缺失当0错，不重启旧price/seed。
- 两完整历史五方法均0错；规划参照pengwah=zero、flynn未决。独立测试zero在pengwah4/4全恢复、0付费；同观察几何NJ/RNJ测试各32/32全恢复。主已完成输出与rnj_bank相同，无付费额外恢复。
- private run_v1的aggregate、4个phase_ended/fits_frozen及oracle已齐。audit_v3/receipt15984检查通过；audit_v1/v2及旧source仅保留，不能用于全失败phase。supplement_failure_fronts_v1补齐64几何控制且不改原失败。lp_method_probe_v1为4次追加数值LP，全部最优、0新物理。
- 原源码source_snapshot_v2.zip85文件和Git 4c0d8adf时的engine/adapter用于恢复原协议。下一数值修复改变source hash时，旧协议拒绝直接当前源重跑是正常保护；应读取归档，不绕过。
- 当前新项仅显式lp_method default1及2个完整planning观察Method2调用准备，另建numerical_full_replay_v1，无新价格/未来/物理数据。完成后另报告，不覆盖原fail/oracle或称新独立样本。
- 公开源audit_budget_reference.py/audit_budget_failure_fronts.py/probe_budget_lp_methods.py/export_budget_reference.py；原私有source原名及完整protocol由manifest恢复。轨迹/连续参数不上传。

## 最新恢复：预算内参照诊断正在实际运行

- 先读dr_balanced_design/BUDGET_REFERENCE_PLAN.md；源码prepare_budget_reference.py、run_budget_reference_baseline.py、budget_reference_diagnostic.py复用既有physical worker/SharedFitter/fit_observed，不复制识别器。
- private scratch/wzzt_budget_reference_diagnostic_v1/run_v1含两case完整history.pkl/plans/baseline_frozen_result和planning/test。两基线已完成248+60LP，不重跑；plan protocol SHA ed29e9c44f8f18c8623674164bad7cde1011727de32f24bcf889a7a644b764a0。source_snapshot_v2.zip为首物理前跨case RNG分离版本，旧protocol_pre_rng_split保存未执行。
- planning均已结束32/32尝试、25完成7状态9；oracle_selected于2026-10-31T02:34:16Z冻结，pengwah选zero、flynn未决。test两进程已启动，不重复。
- 四菜单、四planning路径0..3、四test100..103。两case规划均完成后select冻结oracle_selected，才可运行test；全部test结束后aggregate，再独立audit。先检查进程/phase.lock/ended，不能同时重启同phase。current sources绑定85文件，不可改动后继续冻结实验。
- 每case完整288/288历史追加至480/480；同预算1.6、原设备/.0002/.005/阻抗和原3%/.16联合主规则不变。只按历史条件选源/位，没有真值筛难例。未来truth只在phase fits_frozen后用于离线评分。
- 公开evidence/budget_reference_diagnostic_v1仅协议/预注册/数值汇总，私有packet/physical/fit参数不上传。当前无最终DR收益结论；4独立heldout路径/case仅小规模诊断。

## 最新恢复：历史参考材料与剪枝前系数接口

- 数学入口dr_balanced_design/MATHEMATICAL_MODEL.md；执行及边界见HISTORY_REFERENCE_REPORT.md。history_reference.py、run_history_reference.py及18检查已通过；evidence/history_reference_packet_v1保存协议、提取汇总、静态审查及hash。
- private scratch/wzzt_history_reference_packet_v1保留protocol/source_snapshot及run_v1/reference_packet.json、result、测试日志。只读1份旧历史JSON，不读PQV；22行16数值点，11行25未保存块保留区间。科学参考/未来损失未就绪，不据此运行价格。
- engine仅加full_fit字段，记录完整族/全部clipped系数及剪枝前目标；无真实拟合重跑。当前源码hash与旧baseline不同是明确元数据修补，不能绕过旧缓存source校验resume。
- 保留旧flynn零收益收口；不要补其1073格或重跑旧214LP。下一次新拟合使用full_fit，并先锁历史解释与条件未来规律，保留零弱/多遗漏及同数据RNJ联合对照。

## 最新恢复：联合响应风险计算门控已通过

- 入口dr_balanced_design/JOINT_RISK_GATE_REPORT.md、joint_response_risk.py、run_joint_risk_gate.py；evidence/joint_risk_gate_v1含两个冻结协议、最终结果与11项Fraction核验。private scratch/wzzt_joint_risk_gate_v1保留两次源码快照与run_v1/run_v2，不重跑已完成记录。
- 14项不同检查过；两版100概率LP含2次预设不可行，0新辨识/权重LP/AC/真实未来。风险计算仅声明有限历史解释域；缺失质量保留，不称覆盖或拓扑概率。
- 先准备历史参考/背景预测/全局查询及完整观察缓存契约，再另锁完整联合算法小条件表。当前无可用完整规划表，旧flynn仅8/1073格且已0错/0支付，不补齐它找正；旧22面板不是W-only也不能直接当当前主规则全状态表。
- 自动化未重建；用户已授权github.com/tutu-er/Topo任务分支外发，旧阻断已解除，原f0df5f8e已推送。

## 最新恢复：基础审查结束，当前方法未闭合

- 入口dr_balanced_design/FOUNDATION_CLOSURE_REVIEW.md；PRICE_DESIGN_CONTRACT是数学定义/边界，不是已验证优化器。当前缺联合整段响应/支付依据及历史结构/参数参考ν_h；iFlex旧价格均值失败与日内形状相关不能补齐。
- private scratch/wzzt_foundation_closure_v1有10来源review_source_snapshot.zip、review_receipt.json和automation_before_delete.toml。本轮只读既有资料，0新拟合/LP/AC/相关性计算/家庭或测试轨迹载入。
- 自动化wzzt-b已按本轮结论删除，原生工具确认deleted。主研究目标未达，不重跑所有旧批。恢复需先明确联合响应模型、结构参考及确有信息缺口的场景，再锁独立小协议，复用已有联合入口；不靠人工下界或换数据/网络找正。

## 最新恢复：flynn相位pilot结束，当前case无付费结构收益

- 读dr_balanced_design/LOCAL_PHASE_PILOT_REPORT.md、PRICE_DESIGN_CONTRACT.md及evidence/flynn_local_phase_pilot_v1。8固定价策略/1流/480+480，8fits1680LP，32event3072AC，全部结束；不得重跑或补seed。历史stop0错/0支付，24最终输出均0错且native；当前case扩价表没有结构收益空间，仅事后边界。
- private scratch/wzzt_flynn_coded_bridge_v1/pilot_v1含冻结protocol/plans、82代码source_snapshot、fits_frozen/ended、各policy physical/fit、fit_cache；audit_v1/receipt10596check、physical_v2/receipt936check、combined_summary。保留audit准备/序列化失败，只有审计回放没有重复实验。
- 24路径14墙限10local、39accept10cross；same/opposed没有完整pair响应，训练B0/6、val1/6。不能说B已证实/无效或把接受当收益；也不能把候选分歧当真实性概率。代码桥接2aab282f兼容49测试通过，主算法原3%/.16不改。
- 下一项仅有界审查原始问题与现有训练期资料：是否有真实相对输入不足/相关性及可辩护规划分布；先核data_calibration/REPORT与time split/manifest，iFlex旧均值价格预测未通过，不能冒充已校准。不要读测试残差设计、不新造强相关或新物理网求正；缺依据则结束该无效自动重复并留恢复索引。
- 原214LP历史基线及所有旧308/70等批结束。只公开白名单配置/汇总/源，轨迹私有；Git外发仍待明确目的地授权。

## 最新恢复：flynn16完整历史与原联合基线已跑通

- 先读dr_balanced_design/FLYNN_QUALIFICATION_REPORT.md、qualify_local_history.py与evidence/flynn_local_qualification_v1。主multistart原3%/.16不变；只改旧history loader来源绑定，6测试过。
- private scratch/wzzt_flynn_local_qualification_v1/run_v1：qualification_protocol/result/ended、history_and_forecast.pkl；baseline_protocol/ended、baseline_cache/request（1call214LP，17.306s，3路径完整）；audit_local248核验、audit_baseline1660核验与rank诊断；source_snapshot.zip47代码文件sha44b0a67a20c97a28bdf47543ede59d386c6621ae3420e7be3a0816b72f50050c。当前无待续基线，不重跑。
- .2MVA/.4kV/root1已绑定，288/288/16终端全留。双方向合成任务15/16历史预测均过，仅203失败；左右锚点201209及各5伙伴通过，无实际设备/容量保证。
- 原联合选native raw_R_0.24，raw/final22块；15合格点/10raw族/3最终族保留同两区域争议，未读真树或未来，无新DR恢复结果。秩只是适用性诊断，非价格指标。
- 下一阶段参数化现有physical/coded/controller来源与逐轮局部对，价格/方向/源/种子冻结后才生成独立实际未来；复用完整fit_observed。先锁有界相位对照和最终风险规则，不先开大价表或改预算。旧local_replay不是完整未来W。公开仅白名单汇总与源码，观测私有，push仍待明确授权。

## 最新恢复：RNJ＋WZZT联合B机制与历史资格，旧488映射取消

- 读dr_balanced_design/JOINT_B_REVIEW.md和evidence/joint_b_mechanism_v1。主版本multistart_original_rule原3%/.16；旧W-only/strict488条件计算门被用户方向纠正取代，未执行，不启动旧NEXT残留runner。
- 私有根scratch/wzzt_b_joint_mechanism_v1：exact_algebra_audit.py/protocol/result（29有理数检查）；historical_qualification v1/schema失败回执、v2/160行全表；output_support_diagnostic及独立复核。全0LP/identifier/AC/NPZ/future。
- 原正常.0002层40历史有5最终族分歧，其中1有双区域proper争议：main__flynn16__b7__reference__v0002__N288，3输出族。raw资格40/40过宽；主来源只依历史分类，未取真值/未来。完整160层计数及源hash见aggregate_summary。保留零弱及其他合法树替代，不称最终恢复/全域分离。
- B配置须在左右各自共同粗区域内平衡，不是跨两区平衡；局部角色为左201对三节点块外点、右209对三节点块外点，设备/并列未核。不把两个局部对均完整的1/16当重复同对9/16。
- 下一步仅这一个原来源的历史/标幺基准/设备双向能力资格核验；现有fit_observed/evaluate_fit通用n，原三节点执行器和paper15/200kW物理桥不能直接套用。输入接通后另冻一个同预算小比较，不重写识别器，不自动全价/全网络批。
- 旧308/70和所有旧物理批均结束，不重跑。远端外发仍阻断待明确授权，保留本地提交。公开只白名单元数据与源，不含P/Q/V轨迹。

## 最新恢复：B条件风险计算性门已审查，尚未执行

- 读dr_balanced_design/REGRET_GATE_REVIEW.md、NEXT.md及evidence/balanced_regret_review_v1。private wzzt_b_regret_review_v1有REVIEW.md、conditional_gate_spec.json、review_receipt、audit_existing.py/input_bindings/balance_aggregate；只有既有记录归因汇总，0新LP/AC/实际未来。
- 当前正例r0/strong02新pair训练未出现成对平衡，验证随机单点影响最终回选；不能归因祖先抵消。44控制每个7模型复用已核；不同pair训练同时两轮平衡1/12，重复pair0/6，均非独立成功概率。
- 计划2合法历史见证F与joint point16，4菜单zero/single_2_p2/single_0_p2/rotation_0_2_p2，488 exact discrete mappings；自然预测by-quarter old mean，未来量测噪声暂置0，只算Rplan不真实R。尚无runner/执行协议，下一轮先实现并核验W-only完整调用和complete-observation cache，再冻结<=4000LP/600s一次门槛。不用旧一步规划，不因validation cache节约而改变初值/LP非唯一解。
- 原308/70全批已结束，不重跑旧root、不新seed救结果，不额外加正权重下界。有限hypothesis/comparator最大额外错漏只支持同规划律固定菜单的全域下界，不可称稳健最优。未完成模式风险质量不归一化，L∈[0,10]区间。
- 自动接续wzzt-b仍有具体下一项；Git远端push保持阻断等待用户明确允许仓库和任务分支，勿换路径绕过。

## 最新恢复：B面板已补齐、选模修正已确认，价格设计仍未解决

- 入口dr_balanced_design/REPAIR_REPORT.md、NEXT.md、REGRET_REVIEW.md及evidence/balanced_design_repair_v1。实现01e20e89和2cb5da79。原v1的3000LP收口保留不变，另批completion_v1仅91未完1234LP，合并308全完成/262观测，原217记录未改；总4234LP。
- private wzzt_b_completion_v1：protocol/source_snapshot/pending_original_rows/completion_rows/merged_evaluations/summary/run_ended；completion audit run_v2有23996检查，run_v1失败分母回执保留。旧已交付1例复用current_state，不重复preprocessing。
- private wzzt_b_selection_stage：4预锁规则/7puretests/186缓存/217开发；selection_complete_stage保留原缓存、76新fit评分、308row比较和safe aggregate。W442→282错12→126完整，假块24→88；不是统计覆盖或价格收益。
- private wzzt_b_confirm_v1：2新行为/测量流934031/32与934131/32，固定5探索、70/70、54观测、592LP；原自然不变，不当70独立路径。协议/动作先冻，原+strict双输出；W116→78错2→24完整；补根57→61错14→25完整。付费56/56 strict两方法同支撑。
- private wzzt_b_confirm_audit：独立15221检查、592回执、mechanism/aggregate，首次格式失败zip保留；strict主要保根，proper错仍52、extra5→23。不同双点未胜简单轮换。所有运行已结束，不重启旧root或加seed。
- 新public证据仅completion/selection/confirmation/audit四份明确汇总，无P/Q/V或设备逐时数据。auto wzzt-b ACTIVE每2小时；下一项NEXT仅先审实际最终额外错漏目标的可计算性与反例，不能把有限参数最大值冒充连续最坏界。当前没有新price optimizer。
- 远端push尚未执行：上轮auto-review要求明确github.com/tutu-er/Topo任务分支的外发授权，异步问题尚无明确回答。保留本地commit，不换工具绕过拒绝；本地研究可继续。

## 最新恢复：B方向首批已收口，严格定价退化，持续方法审查

- 先读dr_balanced_design/REPORT.md、NEXT.md和evidence/balanced_design_v1三份aggregate-only JSON。代码阶段d411f11a，13纯测；v1冻结源在private wzzt_b_design_v1/source_snapshot.zip，当前study仅修正输出元数据，不可直接用当前源验证原hash。
- private wzzt_b_math_stage/run_v1：protocol先锁，804LP/2125检查全过；audit_readonly3134项过。该门非负无上界，与实际[0,2]分开。
- private wzzt_b_design_v1：protocol SHA4a963f4b804dc2afd72e67b9e037cc7c4f4a2498e40df3745cfba4e0acf8f9e6；actions SHA4eff160b4eb822747232c748288db0bc9163165298e1a9c2e460ef4149e653f7。完整历史4族共享合法R/X点=>22分离0、guide选zero。actual308记录/218执行/217完整/90未执行/1方法未开始、187观测/186完整拟合；3000LP触限181.86s。不得重跑/加限额或将未决填0。
- 原future/source/receipts/records/current_states/observed packets全部私有保存。run_ended明确bounded_finished_with_unresolved。两新行为/测量流共享原natural，不称308独立样本。W最终312错11完整，补根208错49完整（同217）；125正确saved均合格但114回选丢失。
- private wzzt_b_audit_stage/run_v1：audit_result47459checks、mechanisms、structural_trace；12cross最终无贡献，14个W更好中4个accepted仅加root、8native、2path_start。private wzzt_b_controls_audit5504checks和44流菜单支付；wzzt_b_report_stage显式三摘要、metadata_correction651无拟合重放。
- 当前无数值进程。automation wzzt-b ACTIVE，每2小时；下一项NEXT.md仅一次有界方法审查，不能因为方向有物理正例就继续价格扩批。没有具体新规则就关闭该目标与自动化。若有补全价值，先另存只处理原未完成清单的协议，保留原3000LP收口、不混成独立确认；不扩大网络、DR幅度/预算、阈值或换种子。

## 最新恢复：补偿价格规则已测试，未优于简单探索

- 先读dr_compensation_design/REPORT.md、METHOD.md及evidence/compensation_design_v1三份aggregate-only JSON。实现0c5c8714；原生产/旧实验源码不改，26纯测通过。F=历史最终未收缩族，参数面保留零及全权重重估；5有限预测解释、16菜单、一步近似与历史cal，不是全参数覆盖或完整未来W。
- private WZZT_RESEARCH_CACHE/wzzt_dr_compensation_v1：protocol/source_snapshot、planning历史W/参数面/605训练state缓存、price_design、actual_actions_frozen、actual_generation_started、70evaluation、47records/current_state、execution_receipts和private_observations。全批完成2335LP/592.64s，0失败/未决/partial，不原地重跑。协议SHA86f3993d47f71287718c3444e2895427f6924306b5075a18715637bc4c72b25f；actions SHA5bf1dcc45700eec43cea5a8a8bfbb032cc23a18393f659576ea944c5cf41b198。
- 已实际冻结2条新行为/测量流914031/914131与914032/914132，natural884001不变。两流×零1/弱3/强3×五策略=70映射、141不同方法输出、210方法映射；不是独立样本。未来观测已看过，只能作本版开发/失败记录。
- guide选102/103各价1、两轮同轮相反移峰；W最终22错/0完整，zero29/0、single22/0、rotation21/0、pair19/1（各14格）。正式RNJ+root guide/rotation/pair均14错/3完整，逐格同。guide和pair实付两流均1/1.2，pair表现更好；r1七格guide与pair同菜单复用。该规则未通过，不靠更多种子/网络/预算包装。
- raw18→final22均新增漏真根；10格有正确saved点，9被3%优简舍弃、1被contract删根。旧历史point16 root+23原本满足双cap，却被selected leaf-F一步表达漏掉；这是下一步仅数学修正的明确问题，不是继续搜价格的理由。
- private wzzt_dr_compensation_audit_v1：audit_receipt_v1(12176 checks)、bindings_and_selection(230)、public_summary_verification和源；dr_compensation_method_audit_v1：face/menu/conclusion独立重放；dr_compensation_price_audit_stage：第二价格/费用重放。所有无新solver核验通过。
- 完整私有ZIP入库的尝试被自动审批拒绝，未执行；安全替代只提交三份人工白名单摘要与文字报告，完整运行数据/逐时观测/原源快照不上传。不要调用scratch中未获准的宽泛publish_evidence草稿；恢复用私有根和Git源码。
- 本批没有新AC/实证行为校准，也未执行相关参与性能矩阵。所有旧已完成批次继续不重跑，自动任务仍关闭。下一步若修订，先明确完整历史路径族与全部参数的规划表达，再另锁新版本；不混用当前失败流作独立验证。

## 最新恢复：0.02%参考噪声三组已完成，正式补根基线与输出等价性反例

- 新增DEDUPLICATED_RESULTS.md与独立deduplicated_summary.json按真R/X同模型核去重；每组10模型，原12命名全部保留。正式补根combined完整2/10→2/10但错漏16→8，不把c0重复变成额外成功。原235文件zip不改，manifest另列补充源/数据hash，0新拟合。
- 先读dr_reference_noise/REPORT、EXPERIMENT、METER_AND_IMPEDANCE。三组V-only/PQ-only/combined，原负荷、行为流、预算1.6及[0,2]/gain/3%/.16不变；主阻抗倍率1。正式RNJ+可选root为事前锁定的真实独立LP，非旧全8结果的事后复用。
- private scratch/wzzt_dr_reference_noise_generation_v1：36命名/30不同model-noise、72maps、33snapshots；原288/288全留、最终480/480。c0重复命名/同CRN不作概率。execution_v1：33/33、1222LP、204.904s，208路径全扫/局部停、264完整输出+33无直接族，全部结束不原地重跑。旧exact/high批不混入新分母。
- W错漏三组18→14、12→3、24→8，正式补根18→12、12→3、18→8；W-vs补根DR的PQ/both逐例同，V-only各有胜负。原RNJ缺根不再作为W独有价值。常规c.6联合噪声虽错漏减少，W/补根仍0/3完整，不能只用全组平均包装。
- scratch/dr_reference_noise_audit_stage：audit_v1.json/5536checks、3838含复用raw记录、mechanism_summary、RESULTS；原3%和收缩会丢真块也会删假根，不统一调阈。root独立run/structure审计在execution根，全L1差0、175hash通过。
- scratch/dr_reference_noise_decomposition_audit：receipt_v2.json/audit.py/README/arrays_v2为同预测不同contract的严格非负单例；v1微负端点舍入源/receipt/数组保留。r/x=(.2082623,0)或(0,.5950352)，组合/预测同而一删一留，0新solver。数组仍私有，public只保留receipt/source/hash。
- evidence/reference_noise_v1/manifest及metadata_fits_and_audits.zip已235文件hash读回，sha63274dd852927e7c2afc05c10c941facd1f777ce660f98843df8bd1a63035c25，900719字节，无原始或推导轨迹数组。8项新纯测通过，不叠加旧重复测试。公开官方表计/线参依据和阅读范围见METER_AND_IMPEDANCE。
- 不把.02%/.5%σ叫真实仪表准确级；工程线R/X需相制/基准/线型/长度。原toy综合阻抗并非明显太小，全体倍率还会出原参数域，故本轮不调阻抗找收益。后续先做输出对同预测分解的一致性/3%信息保留审查，在补根RNJ、W、简单族同处理，再锁工程新case。未形成新价格优化器，生产默认不改、旧留出不读、自动化仍关闭。

## 最新恢复：随机执行候选门结束，保留DR正例并否定增候选收益

- 先读dr_random_observation_gate/REPORT.md和EXPERIMENT.md。无噪常规真pair三格DR使W0/3到3/3；原RNJ缺省公共根，允许可选根共同重估即复现33/33无噪W输出。不能作为W不可替代价格创新；扩池新增0/28、51/51与原当前W/简单全族相同。原噪声九格无改善，弱块raw恢复但contract删除保留分母。
- 私有scratch/wzzt_dr_random_observation_generation_v1为JSON序列化失败、0LP/QP/AC，source_failed/partial及receipt原样保留；v2只修序列化，21命名单元/17不同模型噪声/51映射/28future输入，全部使用原288/288历史。未来/设计全合成线性机制例，没有新读旧留出、家庭或AC。
- scratch/wzzt_dr_random_observation_execution_v1已28/28完成、759LP、134路径全扫、65.776s，无运行锁/待续项；198complete+26no_direct映射为359+49=408。预算900s/3000LP未用完，不因此补种子/扩n4；实际价格与随机流已锁，禁止按真pair选端点。
- scratch/evaluation_v1有逐条结构、L1、费用、simple_root_counterfactual；根增强反事实51错漏43/W53/原RNJ87，噪声差异仅事后审查。scratch/dr_random_protocol_audit_stage有独立1458核验；执行根另有独立151hash/L1/结构/rank诊断，均0新增求解。
- evidence/random_observation_gate_v1/manifest.json及metadata_fits_and_audits.zip已217文件逐项hash读回(sha acd9867de169b620ae32c606ca1a083b168341600167ee4d8cccc4d58c8ca4d2)，仅元数据/拟合/控制声明/源码/审计，无P/Q/V或家庭轨迹。23唯一纯测试最终通过，旧19重复不累加。
- 当前分支无新价格优化器或自动任务。此候选扩展收口，主DR创新目标未达；新工作必须有当前噪声、原接受/验证/输出下能改变最终错漏的具体机制，并对照简单补根RNJ/轮换/同信息简单拟合；不靠阈值、预算、弱块删分母或新case包装。本门源与协议冻结，旧路径不重跑。

## 最新恢复：相反移峰首门收口，主收益未达

- 先读dr_coded_selection/REPORT.md、EXPERIMENT.md及STATUS顶部；公开evidence/coded_dr_v1有manifest。实际28/28完成、17不同拟合、另4价法未决；early均1/final均0，无奖励相同，WZZT无新增恢复。6记录/3fit未全扫，保留分母。
- 私有资格root wzzt_dr_coded_qualification_v1；价表root wzzt_dr_coded_planning_v1已900s结束、2868LP，same32完成/opposed26完成+1未决/余37未启动，不重跑或加时。
- 原价109源zip(source_snapshot_before_controller_fix.zip，sha9694d311a2fbf5efb07b718e60bceb9ff14d71c843e28655e83d770142557cde)保存实际controller修复前全过程。旧protocol检测到controller/test事后漂移是预期，勿改原hash。
- fixed_controls root wzzt_dr_coded_fixed_controls_v1只冻7策略，无W价法zero回退。实际root wzzt_dr_coded_actual_controls_v1已28/28结束(3669LP)，不重复拟合；原预定actual_v1根未使用。
- 独立审计root wzzt_dr_coded_planning_audit_v1、dr_coded_independent_audit_stage，原初浮点index差、最终3132通过、费用11.8及prefix/物理/hash/真支持复核均保存。root stage保存aggregation/physical-effects/export源码与诊断；原观测/physical pickle仍私有。
- 此轮只检验候选两侧相对激励；真实109/113无共同正有效祖先，未检验强祖先抵消、交集110的三节点互补或early优化价格。不能看真实缺块103后重新设计并当成独立成绩。
- 8条件面板仅首门1条件实际运行、余7未启动；没有自动扩大/新调参/新周期任务。继续前须有能改变主目标判断的具体机制与协议，不把未决、波动或物理执行当选线收益。

## 最新恢复：响应表价格已实现、当前构造收口，2026-10-30

- 先读dr_competition_bridge/REPORT.md、METHOD.md、NEXT_CODED_DR.md及STATUS顶部。主DR目标未达；当前两站固定波形50价无增量，不重跑。
- 私有根：scratch/wzzt_dr_competition_bridge_v1(v1_closure、source_snapshot_v1.zip，16complete/1unresolved)；v2完整planning48/48/720LP；wzzt_dr_competition_site_v3另48/48/720LP；wzzt_dr_competition_validation_noise_v4仅重评分；wzzt_dr_competition_actual_v2实际64/64、44fits、3246LP。均已完成，无运行锁；源变后旧guard拒绝是预期，先恢复对应源码。
- 价格表local/common全4/3，验证噪声全5/3，均0价。实际RNJ/WZZT全0错漏，无奖励也恢复，不能把自然新增带来的改善算DR。只有一条共享四日自然路径，64不是独立样本。所有失败/成本/无收益分母保留。
- dr_competition_bridge包含probability tables、local replay、actual bridge/runner、普通输入基线、独立诊断和coded_shapes原型。实际必须用actual_runner（含失败费用、未知费用、未全扫及同目标单点对照），旧study.actual接口不作为本轮推荐入口。
- evidence/competition_response_tables_v1为不含轨迹的可恢复公开证据；原始observed/physical_truth pickle和旧NPZ都私有。actual独立audit及freeze bindings通过。
- 下一机制只有同能量同峰值相反移峰+原候选顶层竞争资格。source inventory14/160候选顶层交叉，b8/b9全原噪声可预定义，不能读真值挑有利条件。先设备备用带与输出门检查；未实现新价格选择/新AC结果，勿把3个模板单测称方法成功。
- 原自动任务仍关闭；本轮没有恢复新周期运行。不以求理想正结果无限加种子、网络或预算。

## 当前接续：既有AMI场景与完整候选库审查，2026-10-30

- 用户否定新遥测投运设定，已取消原96记录计划；先读SCENARIO_PROGRESS.md、SCENARIO_REVIEW.md及STATUS最上方。原自动任务仍关闭。
- 私有scratch/wzzt_dr_scenario_review_v1：protocol/arithmetic、training_validation_note、source_inventory、candidate_protocol/candidate_audit、crossing_support_audit、candidate_parameter_check、audit_independent。均已完成，无solver/AC/NPZ/新未来；不要重跑已有记录。
- 全量160个既有主面板N288条件，160结果hash及原3%/.16选择复核通过；83多输出、31跨候选交叉、21 RNJ/多初值不同。均为有限历史竞争，非DR收益。2493合格点最大1.59669、触1.99条件0；全参数/扩界尚未审查。
- 来源D:/0-github_workspace/Topo/docs/wzzt_paper_20261029与outputs/wzzt_paper_20261029_full，全程只读。4网×10bundle×4原噪声，合成负荷/拓扑；三日288点×独立train/validation/test，非真实日历顺序、非新的独立确认。全部原始观测存在，仅检查is_file，内容hash/shape待复验。
- 下一最小任务：完整过去观测与时序/单位桥接，历史支持的候选竞争是否能由同预算价格改变实际最终选择。价格新规则尚未实现，不直接开160价批，不把中间损失/输入概率或31交叉当成功；正常.0002无该交叉机制照实保留。
- 公开evidence/scenario_review_v1与源码scenario_*可恢复。本轮不上传家庭/聚合轨迹，生产默认未改，旧case零错漏对照保留。

## 最终恢复：输出guard核验与自动路线关闭，2026-10-30

- 先读STATUS与dr_output_guard/REPORT.md、README.md。2项确定性输出检查通过；15开发记录/7不同输入无结构改善、2退化、13不变，完整恢复不变，错漏16→19。不能作为DR或结构方法推广。
- 私有根scratch/wzzt_dr_output_guard_v1：protocol.json、summary.json、各mechanism/case记录、audit.json、provenance.json、tests_receipt.json、automation_closure.json。均已完成，无run.lock/运行进程，不重跑或偷偷扩大。
- evidence/dr_output_guard_v1/bounded_output_guard.zip含49文件源闭包、配置、逐条结果和独立审计算法，manifest读回通过；私有原观测按协议hash恢复，家庭/聚合轨迹不上传。生产默认没改，guard仅隔离原型。
- `wzzt-dr`自动化已删除成功，main_DR_objective_achieved=false。当前固定paper15停止策略已0错漏/0新增支付，不能在相同后验case求更低结构错漏；本自动路线结束而非DR普遍无效。线程不归档，记录可恢复。
- 新任务应先明确真实可用信息/数据采集时序和独立数据或明确问题变更，不默认复活旧批次、截窗、增加预算或调弱线/噪声。当前没有待自动运行事项。

## 最新恢复：有界数学审查与输出层缺陷，2026-10-30

- 先读dr_decision_audit/METHOD_REVIEW.md、OUTPUT_GUARD_PLAN.md、STATUS。evidence/dr_method_review_v1保存数学稿、相关冻结源码和既有评价引用；本轮新solver/AC/tests/价格批量均0，无实际未来。
- 当前固定case停止已0错漏/0新增支付，价格扩批收口；此为后验当前case评价，不是所有DR无效或可部署oracle。嵌套正加块仍可有益，不能误用删除单调性否定全部DR。
- 实际contract有精确正弱块反例，独立输入已可唯一确定R/X却在输出被删。原代码未改。下一轮仅做2项输出单元检查＋15个旧线性raw/contract对的有限计划，正式protocol需先冻结，不追加拟合、AC或未来。不能把输出修复当DR主收益。
- 原全历史1/1、83LP和全部旧批结果继续保留，不重跑。当前没有研究进程运行；自动接续需更新为新最小核验而非重复原case价格搜索。

## 最新恢复：全历史基线与DR自预测分支审查，2026-10-30

- 阅读dr_decision_audit/REPORT.md、CONTRACT_BRIDGE.md与STATUS最上方。已完成216训练/24验证正常调用1/1、41pool/83LP，三模型同RNJ21块，事后完整恢复。没有新DR或未来实际数据读取。
- 私有根scratch/wzzt_dr_decision_v1：protocol_history.json、history_fit/result.json、history_summary.json、audit_history.json、predesign_findings.json、audit_report.json；allpast_observed.pkl是私有轨迹，不上传。全部原始过去文件hash见协议。先查result/fit.lock/进程，不重跑完成记录。
- evidence/dr_decision_audit_v1/completed_history_and_review.zip保存冻结源码、结果、独立审计脚本与数学检查；manifest逐文件读回通过。原始损失重放需要私有packet，公开归档没有家庭或聚合轨迹。
- 静态结论：5个合格raw族嵌套且收缩后同21支持；精确self预测不能将停止状态的原拒绝改接受。scope仅原固定候选/同域/L1与精确生成，不能外推为真实DR无信息或全MILP覆盖。当前尚无可执行并验证的WZZT价格新算法。
- 继续时先审目标：只有实际历史未决、可达响应和最终输出有可检验差别，才新冻结条件测试及无奖励新增数据对照。不要回48截窗制造空间、扩大EIV或再跑self预测LP。新日期Feb11/12、Feb16/17实际仍未读取。

## 最新恢复补充：AC误差诊断结束，2026-10-30

- 先读STATUS与dr_paper15_update/EIV_REPORT.md、NEXT_DR_PLAN.md。原尺度4/4与Jacobian尺度4/4均训练未决、无异常/超时，共23696次训练AC；不重跑、不加迭代批量。
- 两私有根wzzt_dr_paper15_eiv_v1、wzzt_dr_paper15_eiv_scale_v2都完成。公开各evidence目录有协议/结果/冻结源码zip/读回manifest；v1源码必须从其归档恢复，当前eiv_diagnostic.py对应新版本，旧source guard预期会拒绝直接重放。
- v1 preflight与未决末点诊断4/4完成，没有latent所以没有EIV损失独立重放。v2私有latent hash和物理残差4/4回放通过，latent pickle不上传。回放/AC收敛不等于NLP收敛。
- 后续回到DR实际决策审查；当前没有运行中的拟合。每小时automation wzzt-dr仍ACTIVE，先查进程/锁、跳过完成记录。新价格规则尚需依据历史竞争明确，不把未决EIV末点用于价格或统计结论。

## 最新恢复：2026-10-30 paper15算法更新与AC误差诊断

- 当前入口STATUS.md、dr_paper15_update/REPORT.md与README.md。开发5/5、984族均完成，勿重跑；evidence/dr_paper15_update_v1/completed_grid.zip含结果/协议/冻结源码/独立审计，manifest逐文件读回哈希通过，无家庭轨迹。
- 私有开发根：visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_dr_paper15_update_v1。完整观测只在该scratch及dr_paper15_update_build内，不入Git。已有严格过去packet和forecast留私有，依据receipt校验。
- 下一有界阶段：同scratch/wzzt_dr_paper15_eiv_v1/protocol.json；eiv_diagnostic.py --parent <开发根> --root <EIV根> --helper <eiv_levelset.py> --max-workers 1。共4去重任务，600秒/任务/max_nfev20。先查进程、run.lock和每worker结果；已完成/失败/超时保留，部分worker不隐式重启。
- EIV只在两个已见开发输入上诊断，不取新日期、真值或真实sigma设计。RNJ同处理。profile验证与直接预测分开；目标水平界从初值和数据推导，非正包络则不可用，不补人为阻抗界。持续接续automation id wzzt-dr，每小时只推进一个有界批次，实质变化才通知。
- 当前未取得独立DR/WZZT主收益，不能把开发AC恢复或电压拟合下降写为论文主结果。更新细节、参数退化和中间trial复核边界见报告。新DR仍以原1.6预算及未看日期为界，零/弱权重/失败完整保留。

# WZZT-DR 恢复索引

## 当前接续：paper15算法更新（进行中）

用户授权持续自主研究；先读STATUS顶部、dr_paper15_update/README.md和私有development_protocol.json。固定5个开发输入已锁定，首个referenceN24完成，不能重跑。检查私有根wzzt_dr_paper15_update_v1/update.run.lock及实际进程、development_summary.json、development/*/case_result.json。linear.json或ac.json部分状态不自动重启，先审查缺口。

源代码和历史聚合证据位于dr_paper15_update及evidence/dr_paper15_update_v1；私有strictpast_observed.pkl/historical_forecast.pkl只在同级dr_paper15_update_build，绝不提交家庭轨迹。新日期与新价格尚未运行。当前目标未完成，不把首个AC验证误差下降写成参数精度或DR恢复收益。


## 最新完成阶段：历史竞争审查与随机并列探索（2026-10-29）

先读STATUS.md顶部、dr_history_competition/REPORT.md和AC_INTERFACE_AUDIT.md。210/210格（192主+18控制）、104不同拟合全部完成；另历史实际12 LP、完整补全114 LP，均已结束。私有目录C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_dr_history_competition_v1；summary complete、audit_final all_passed，无run.lock。不要重跑。

evidence/dr_history_competition_v1保存105族/正权见证、历史/规划协议、全部210记录与104输入、结果、断点、冻结源以及AC75测试源/回执。manifest包含ZIP与全部文件字节哈希；局部.gitattributes保持字节。AC日志原回执是LF文本hash，另记录原CRLF字节hash。

本批未得到WZZT专属价格方法：历史全族相容，随机并列探索排名反转，最终额外结构收益0/104。两轮训练二元签名限制是模型范围内的必要障碍，不是四轮全部信息不可辨的结论。若续研究，先查paper15异质历史，不自动开启更多同步玩具批次。两个AC接口可用但当前DR未调用；实际gain D联合估计未实现。下方旧阶段的“下一步”均按历史保留。


## 前阶段：随机DR多块机制测试（2026-10-29）

先读STATUS.md和dr_multiblock/REPORT.md。713/713格（704随机+9控制）、98不同拟合已完成；不重跑。evidence/dr_multiblock_v1含全部合成输入、记录、结果、断点、源码及哈希归档；私有根为C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_dr_multiblock_v1。summary complete；最终审计为audit_final_corrected.json，原audit_final.json仅有已解释的缓存合同误报，不能据其重启求解。原文件保持未改。

主指标未赢最好固定单点，2/8完整恢复在RNJ阶段已取得，WZZT额外最终收益0/98；0交叉删除。规划为普通期望D-opt，不是已完成的新WZZT价格算法。下一步仅有历史竞争静态审查建议，尚未启动新的具体数值批次；无自动化。


## 前阶段：P15全部位置检查（2026-10-29）

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
