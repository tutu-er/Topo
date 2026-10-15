## 最新：固定线路目标与 I² 梯度核查（2026-10-15）

先读 `ev_fixed_gradient_results.md`，完整数学与阶段验收见 `ev_fixed_gradient_math.md`。当前主策略仍为一拓扑一组训练R/X、竞争源逐时调整的平均最近竞争者目标。历史进入候选/参数/预测，历史损失差与参数弱方向尚未进入规划；历史offset仅推导和解析验证，没有接入旧策略。

新增独立AC/无损/I²/锚定I²响应与梯度模块，61项测试通过。四份开发历史的768/768源拟合完成，32项方向差分通过，与旧AC源拟合48/48分数一致。I²距离/梯度相对误差中位数约0.0046%/0.0058%，192拟合累计10.74→7.45秒；这是同参数计算诊断，锚定版最坏误差反而较高。尚未改变选址、开展新策略校准或声称提高真实排错率。证据与94份源码/输入哈希见 `evidence/ev_fixed_gradient/`。原LaTeX/PDF未修改。

## 最新：R/X物理范围、强精度基线与最近竞争者（2026-10-15）

先读 `ev_rx_results.md`；数学见 `ev_rx_math.md`、`ev_rx_interpretation.md`。每拓扑仍只有一组训练R/X，验证和未来固定线路。新增统一物理X/R范围、同域平方功率精度MILP、平均最近竞争者目标；全部候选/源补偿和完整AC接受保留，规划不用未来真值。

完成4历史box/cone配对开发、36校准、12独立评价，全部1064方法槽完整。配对开发X误差均值69.15%→38.25%，共同EV电压增量误差降低。独立评价中，移动双向最近竞争者/同域平方基线平均M=338.48/257.61，6赢3同3输；校准后唯一真10/12对9/12，二者误拒1/12。最近竞争者较原全对均值无稳定额外优势；共同失败与未达代理间隙保留。原精度基线仅以电压方差倒数加权净功率，统一精度时还受循环损耗/驾驶补电影响；新增平方基线在本例有独立解析最优性证明。

完整AC回放及36749项独立统计核查通过。证据、原字节归档及续接命令见 `ev_rx_reproduce.md` 和 `evidence/ev_rx/`。下一步重点是固定参数的动作响应风险，尚未把训练Hessian当置信集。原LaTeX/PDF未修改。

## 最新：普通时变负荷与逐时预测（2026-10-15）

先读 `ev_dynamic_protocol.md`、`ev_dynamic_math.md`。新增 `ev_dynamic_loads.py` 生成共同/各户 AR(1) 普通负荷与轻微变化功率因数，预测只用训练，跨过验证间隔；不使用未来真值。每拓扑仍保留一组训练 R/X，仅竞争源逐时重新拟合。新入口 `ev_dynamic_study.py` 联合MILP配置地点、移动及连续充放电，完整AC目标统一接受新旧计划；代理误差门只诊断。旧默认及旧实验不覆盖。

先看 `ev_dynamic_results.md`；复核步骤见 `ev_dynamic_reproduce.md`。144项集成测试及14项交付/分析工具测试通过。4个独立开发历史在两精度条件下完成，36校准与12独立评价全部通过AC回放；开发21、校准39、评价15个原文件恢复SHA全等，重算阈值和评价报告逐字节一致。主场景统一计量精度，恒充、预设高低与四个匹配操作域的精度基线均保留。

独立评价中，固定双向判别/同域精度平均M为395.86/404.64，两者均9/12唯一保留真候选、0/12误拒；移动双向判别M为433.96，但唯一真8/12、未决4/12。移动相对固定双向1赢10同1输，收益集中。幅值集中和开放双向相对恒充有价值，拓扑信息尚无稳定固定域增量。规划与实际差异、R/X补偿及逐时价格推导见 `ev_dynamic_interpretation.md`；两条最差校准历史加大源拟合重启后负M保持。完整反例、未达代理间隙及旧记录缺口均保留。原LaTeX/PDF本阶段不修改。

## 训练参数固定与独立验证（2026-10-15）

分支 `codex/ev-train-validation-design`，从 `1c71cb2` 建立。先读 `ev_fixed_train_protocol.md`、`ev_fixed_train_math.md`。用户已确认P/Q直接作输入；训练段估R/X，验证及EV新数据固定这组线路，仅估每时段共同源P/Q。首版训练8点、验证4点、未来6点；验证不进入规划，未来预测先用训练均值。旧72源和既有结果保持原样。

已完成固定输入拟合、源QP/MILP及整策略开发，56项集成测试与4项源码恢复测试通过。weak8、rich8、rich32、rich96共16个条件对应4个配对独立资产：32/96点训练两主实际裕量均4/4正，原弱训练仅1/4；训练支持不足是重要失败原因。两主相对质量价格MILP尚无稳定新增优势，平坦预测的时间置换不变性、cap20封顶及完整AC门分别定位。用户已确认保持每拓扑一组参数，验证不连续调线路。正式校准/评价0/0。入口为`ev_fixed_train_results.md`、`ev_fixed_train_interpretation.md`和`ev_fixed_train_next_protocol.md`。原12H改善分解与MILP物理自由度见 `evidence/ev_fixed_train/previous_gain/milp_gain_analysis.md`。上限目标相对平均目标的M增量51.99中，未来电压项49.41；空间与曲线的逐历史机制不同，求解精度影响仍需消融。

新增完整AC标量分解与插值界推导，4项解析/极小MILP测试；定向54节点/648任务通过，粗距离插值反而较差。后验残差插值平方较准204/288，但未接选择器；门审计找到10项被拒的AC预测改进（8条去重曲线），尚未评价未来效果。总计66项针对性测试通过。源码从实际Git对象导出后，78/79冻结源恢复精确字节，16条件全部AC/QP复算通过。下一步先核查完整AC接受准则；仍保持每拓扑一组训练参数。

冻结数学稿保持实验时原字节；输出高斯工作模型、理想最小值J与局部取得值的补充口径见原报告第18.2–18.4节。原报告第18节已原位更新，109页PDF编译及改动页视觉检查通过。

以下保留原研究分支的已完成结果。

# AC 最大似然拟合与 EV 模型判别设计

## 最新：紧凑 MILP、确定性继承与预测解释（2026-10-15）

先读 `ev_compact_protocol.md`、`ev_compact_checkpoint.json`；数学见 `ev_compact_math.md`、`ev_original_shared_square_math.md`、`ev_prediction_ensemble_math.md`、`ev_policy_calibration_math.md`。本阶段主方法是 MILP（混合整数线性规划）：二进制变量配置接入、移动、充放电模式及平方分段，连续变量配置功率和电量。完整 AC 竞争解释仍可重新拟合。随机价格曲线只作独立对照，不向主方法提供初值、继承计划或选择候选；这一要求替代下文旧阶段“随机初值”的待办。

共享控制平方已接入，主例四段时 18 个控制最多 72 个平方分段二进制，操作变量另计。相同真实历史见证下，移动双向主问题的总二进制由 1860 减至 204，2 秒主问题预算的代理上界由 1424.19 收紧到 1071.12。阶段提交 `6ca0f85` 与 `a3dedf0` 已普通推送原研究分支。

已实现较小操作域的确定性计划继承、历史平坦方向的合法参数解释及负荷/同点源/功率追踪预测情景。修订版 v2 的 4 条开发历史全部完成；4063 个保存拟合状态、44 个真实未来响应、47 个历史解释通过独立 AC/操作复算，15696 个竞争 QP 通过完整矩阵与上下界复算。开发中移动纯充电、精化平均目标、上限20目标的平均实际 AC 裕量分别为 128.37、107.45、123.37；这些数值用于开发诊断。平均任务分数会牺牲当前最难竞争者，源预测及参数解释仍有偏差，负例完整保留。

v2 按72份科学源码、配置和种子冻结。36校准与12独立评价全部完成，阈值先于评价启动保存。校准36288保存AC状态、396真实未来、154584QP；评价12384状态、132真实未来、48816QP，均通过复核；371项集成测试通过。统计由独立程序重算80项全通过。

最新结论与下一阶段见 `ev_compact_results.md`。上限20目标的平均实际AC裕量158.3，恒充7.38、随机112.8；唯一正确为12/12、5/12、10/12，三者误拒0/12。平均目标106.3仍略低随机。两主方法最终计划12/12来自MILP；上限本层新6、继承6，平均新1、继承11，随机进入主方法为0。全部11方法的风险、逐对分数、阻抗、来源及失败保留。X误差较大、无EV3次误拒及固定充电1次误拒均保留，4例扩大局部搜索后成本变化仅浮点量级。

24条件同波形位置消融通过，上限目标实际位置收益9正/2零/1负；该消融隔离电气作用，与计入驾驶/接线/同终SOC的主比较分开。无损恢复入口为 `evidence/ev_compact/delivery_calibration_v2/README.md` 和 `delivery_evaluation_v2/README.md`，原报告第17节已原位更新。完整raw在原D阶段目录保留，紧凑交付带原始哈希及覆盖范围。滚动、多轮风险、proper预测EIG和扩网尚未完成。

下面各节保留旧版本、旧分母和当时结论。

## 最新：原参数域平均任务MILP与移动充放电（2026-10-14）

先读 `ev_original_results.md`、`ev_original_checkpoint.json`；完整方案在 `ev_original_protocol.md`。数学见 `ev_original_domain_milp_math.md`、`ev_original_interpretation_math.md`、`ev_mobile_service_math.md`。旧阶段下文保留原版本和原分母。

已实现全部4候选、12有向任务的平均价值及原物理参数域。共享R/X与H/未来状态始终可重拟合，同点源逐时变化，EV计入总P一次。时间展开MILP联合配置位置、连续充放电、接线、移动和SOC，没有预列完整路线/曲线库。QP下界、MILP上界及完整AC点检查分开记录。主例合法共移动见证严格消去其第一层控制乘积，残差平方分段实际运行；通用非零乘积分段已独立测试。

4开发历史验收后冻结68科学源，12独立历史评价完成。真候选取得最小联合AC损失：恒充9/12、固定充电10/12、移动充电12/12、固定V2G10/12、移动V2G11/12。随机方法有1个操作失败保留；移动V2G相对随机的11有效配对5正6负。48优化建议中11达到代理间隙，移动间隙仍宽；这些排名与分数不是风险或正确概率。

全12H×2方法的24同波形位置消融完成：移动充电11正1负，平均ΔM=39.846，主要来自未来电压通道；移动V2G6正5零1负。搬回初始点保持逐时总P/电量/吞吐/噪声，取消驾驶会改变终端电池电量，故该电气隔离与主研究同服务比较分别报告。两条明显负例及5个未移动缓存保留。

另完成同4开发H的宽域对照、2条2m短边的asset/numeric压力。结构覆盖与真实R/X入域分开：2m在资产域外，过高下界可放大错误偏好。公开电缆参数和25–150m模拟长度族的物理依据在结果稿；没有实测长度资料。三个新增目录均完整独立重放。

当前219项针对性测试通过，2项真实Windows链接创建因权限跳过。主评价3384保存拟合、3516QP、107真实响应，位置消融432拟合/48响应及所有域压力均通过独立算术回放。7个紧凑包61/61文件在新目录恢复后原字节一致（46.48MB）；raw完整保留。恢复入口为 `evidence/ev_original_domain/compact/README.md`，68/70源仅换行恢复见 `ev_original_source_restore_README.md`。

原报告第16节原位更新并编译检查。下一步共享18个功率平方压缩主问题的推导和948见证/4620动作核对已完成，尚未接主策略；见 `ev_original_shared_square_math.md`。随后保留较小操作域及随机初值、完善历史解释和未来预测，再重新冻结评价。新风险校准、proper预测EIG、滚动及扩网仍待后续。

以下为先前阶段记录。


## 竞争响应、全候选曲线与物理位置（2026-10-14，本轮阶段交付）

最新入口：`ev_cross_stage_results.md`、`ev_cross_stage_checkpoint.json` 与 `ev_cross_next_protocol.md`。数学分别在 `ev_ac_moving_cross_math.md`、`ev_cross_price_math.md`、`ev_discrimination_aggregation_math.md`、`ev_correlated_load_excitation_math.md`；5篇原文算例核查见 `ev_related_case_review.md`。

控制同时移动参考/竞争总P/Q中心，保留功率×线路/状态交叉项。54份旧合法见证逐通道白化RMS≤.25由18/54提高到48/54；其余6个远解释失效保留，新批为开发而非正式风险评价。原数值R/X正下界1e−6pu保持；未以结果调高。现场资料仍缺，标准馈线等效功率例单列。

全4候选/12方向v2完成8独立H、16质量条件、112方法条件，全部联合拟合有效，漏真0。实际604拟合调用，22,450批量时序AC接口调用；1024保存状态、112物理响应、1728QP独立重放通过。首2H用于v1数值修复、后6H冻结后运行；formal0/0。v1两数值界冲突和两方法拟合失败保持。

最坏全对代理16/16在恒充处已达声明.01间隙，开放竞争主MILP调用0；固定MLE分段MILP16次。raw/guarded=恒充，简单质量/随机价格曲线在部分块内cherry条件得到更大真实AC裕量，块外和star仍弱。新的平均任务MILP已推导，尚未接AC主策略。各方法排名不是排错概率。

三末端相关负荷线性例开放公共干线与末端斜率重拟合，同电量恒充成本2.7、高低曲线7.2，MILP曲线及补偿失效反例均验证。标准case33两历史40候选拟合完成；同曲线三个位置会改变小信号，R/X仍不稳。同位置同输入EV/ESS AC差0；10驾驶/接线/SOC情景5不可行保留，净移动收益尚未验证。

139个唯一针对性测试通过。源码字节和求解界在 `evidence/ev_cross_all/`、物理位置在 `evidence/ev_ev_ess_spatial/`。新checkout先 `python -m research_experiments.mobile_ev_mle.ev_cross_source_restore` 只读核对33科学源；仅换行差时显式 `--restore-line-endings` 恢复，再按 `ev_cross_study_review --directory research_experiments/mobile_ev_mle/evidence/ev_cross_all/v2 --output <新review文件>` 重放。实际Git对象导出后33/33恢复一致。

原报告第15节原位更新，编译与视觉核验见 `evidence/ev_cross_all/delivery/delivery_verification.json`。严格模型门、平均目标AC部署、正式风险、扣成本的位置收益、滚动和扩网仍待验收；旧章节是历史阶段证据。


## 连续曲线MILP、阻抗尺度与交叉项（2026-10-14）

最新入口：`ev_profile_milp_results.md`、`ev_profile_milp_checkpoint.json`、`ev_profile_next_protocol.md`。完整推导见 `ev_profile_milp_math.md`、`ev_bilinear_support_math.md` 和 `ev_bilinear_exchange_math.md`；文献正文核查在 `ev_milp_profile_literature.md`。

连续平方分段MILP已实现，保存冻结二次目标的上下界。3独立基础历史产生21尺度/拟合域条件，84方法条件的完整AC联合拟合均完成。覆盖内15条件中恒充、竞争投影MILP的正确最小损失候选分别12/15、14/15；这是重复开发条件，未使用拒绝阈值。提高真实内部阻抗增强本批信号，提高拟合下界排除真实弱边时则放大错误偏好。固定参数曲线21/21达到简单计量质量LP的最优分数，该对照继续保留。

独立重放528份保存拟合及84份未来响应通过。54行事后四角AC诊断定位到参数与功率交叉项遗漏：有界线性QP最小成本376.641，而合法完整AC解释成本25.443。新的双线性方向下界及竞争解释交换MILP已完成解析原型，尚未接AC选择器；根代理72项针对性测试通过。先按新协议验证混合导数与同一参数见证，再用新历史评价，暂不扩网。证据、25份冻结源码字节、配置、种子、失败分母、求解界在 `evidence/ev_profile_milp/`。

原报告第14节已原位编译并检查为76页。原PDF在开始时已有修改，保持本地更新而不加入阶段提交；交付校验保存于 `evidence/ev_profile_milp/delivery_verification.json`。

新checkout可运行 `python -m research_experiments.mobile_ev_mle.ev_profile_source_restore` 默认只读核对25份冻结源码；若仅换行不同，显式加 `--restore-line-endings` 恢复原字节。工具经7项安全测试，真实内容变化或缺失会拒绝整批恢复。随后运行 `ev_profile_milp_review --output <新的复核目录>` 重放保存证据。

## EV电池服务与接入灵活性（2026-10-14，持续研究）

先读 `ev_flexibility_protocol.md`、`ev_flexibility_math.md` 和 `ev_flexibility_checkpoint.json`。新操作LP/MILP固定终端电池电量，明确充/放电、损耗、驾驶、接线和离站条件；旧21科学源码和既有结果保留。15解析+35操作+5AC接口通过；完整研究回归967通过，之后新统计外围29和重放23另行验证。

新开发6独立H/18质量条件、162方法条件完成；648保存联合拟合、72历史副本及162真实响应独立重放通过。连续曲线由优化产生，少量节点外层比较；固定点后续价格步8/18有正增益。移动/人员/V2G的成功和负例均保留，尚无统一优越性结论。同一H三个质量条件只算一条历史。紧凑证据和重放在 `evidence/ev_flexibility/v1_development/compact_full_diagnostics/`。

第一批39校准出现1条v2g_mobile拟合失败，原阈值∞、评价0/48完整保留。失败保留全部候选的拒绝分数经独立证明和37测试后，重新冻结safe-v2规则；新39校准984001--984039全部拟合有效，阈值6.034651101192644，9/39超过成本门。新48评价985001--985048及独立重放完成：恒充唯一正确28/48、固定曲线27/48、移动30/48、随机29/48、人员扩窗28/48、两种V2G各27/48。固定/移动V2G误拒2/1，任一9方法误拒2/48；最终AC拟合432/432方法条件有效，4未启用、2共同设计回退另列。评价选择>15秒11/48。实际拟合损失与失败记录保持原样；分数0仅编码未实施拒绝。本轮还未完成滚动、扩网和联合人员路线。

完整数学、结果与下一阶段门见 `ev_flexibility_results.md`、`ev_flexibility_next_math.md`。当前指标确能改善局部分数(固定22/42后续价格有增益)，但与实际裕量增量的描述相关仅.086；固定曲线未通过提高实际排错的主要验收。优先补正cherry竞争方向和合法历史展开点，保持弱边未决，暂不扩网。完整可续接紧凑证据位于 `evidence/ev_flexibility/v2_safe_formal/compact_full_diagnostics/`。

新检出仓库可先运行 `python -m research_experiments.mobile_ev_mle.ev_flexibility_source_restore` 只读核对；仅有换行差时显式加 `--restore-line-endings` 恢复23冻结文件的原字节。随后用 `ev_flexibility_safe_review --directory research_experiments/mobile_ev_mle/evidence/ev_flexibility/v2_safe_formal/compact_full_diagnostics --phase evaluation` 独立复核，无需本地raw记录。该流程已在只含提交文件的专用目录验证；恢复工具6测试通过。原报告原位编译为71页并检查更新页。

## 疑似支路信息价格（2026-10-14，正式阶段完成）

先读 `structured_information_results_review.md`、`structured_information_checkpoint.json` 和 `structured_information_next_protocol.md`。完整推导在 `structured_information_math.md`、`targeting_math.md`、`information_power_link_math.md`；预声明阶段与规则在 `structured_information_protocol.md`、`structured_information_protocol_v2.md`。

完整 AC 白化导数构造疑似内部 R/X 的信息，投影其他共享线路和 H/N 各自真实 P/Q、同点逐时源的补偿。连续功率由 LP/MILP 价格步优化，再检查完整 AC 信息；少数接入节点逐个比较。全部四个 star/cherry 候选参加完整 AC 联合拟合。LP/MILP 界属于仿射子问题，原目标仍是局部多初值非凸搜索。

解析、秩亏、活动边界、梯度和24条小支路尺度 AC 审计已通过；最小尺度 SSE/二次预测比为1.00014--1.00039。工程方向的随机单位改进均值及固定线性 Gaussian 小信号功效联系已推导，21项功效数值检查通过。纯 X 弱方向、角点停留和远处非线性补偿反例保留。

v1/v2各为同一6独立H、18质量条件的开发 pilot。v1等权增强错误任务的负例推动 v2 的 H-only 任务权重和第二个可行历史解释点；新规则在正式数据前冻结。权重是任务优先级，可行扰动点不称 MLE 或统计置信集。v3仅补调用计数，动作/损失/调用计数等价证据保留，旧13份科学源未改。

正式39独立校准、80独立评价完成，所有119历史的19份冻结科学源码相同。阈值2.9617106149485366。唯一正确：无EV28/80、固定恒充49/80、简单全局质量选点53/80、信息设计57/80；互斥安全未决分别48、27、22、16，误拒分别4、3、5、7；恒充另有1次拟合失败。信息设计相对恒充配对裕量提高627.891（SE127.681，n79）；相对简单全局质量提高−29.757（SE147.447，n80）。二元完整80分母为9胜/1负，双侧精确p=.021484；该项为补充终点。相对质量选点、多中心、任务权重和后续迭代的稳定额外排错收益尚未建立。

任一11方法误拒11/80，95%区间[7.07%,23.27%]。有限秩结论属于冻结模拟混合和随机校准样本的边际风险；当前固定门槛的条件风险另需验证。star误拒4/10、weak仅2/12唯一正确，均保留。两处star展开支持失败属于展开模型范围，8条历史已分开而未启用。下一阶段优先处理合法正边竞争与弱信号风险，不直接扩网。

独立 AC 重放：39cal的1872保存拟合和80eval的3840保存拟合通过，实际唯一拟合分别1060和2224；首8eval完整投影735目标/2205矩阵检查通过。原恒充三个错误候选的max200失败在独立扩大预算诊断中3/3恢复（24/24初值ftol结束），原冻结记录不改。固定前16H相关平均误差诊断复用保存动作，任一方法误拒2/16变3/16，未增加独立历史样本；其348实际拟合及独立计量/AC重放通过。

结果表、每边R/X、已取得loglik、源状态、求解界、失败、源码字节、配置/种子和独立重放在 `evidence/structured_information/v3_frozen/`。原summary保留首次导出：其中旧二元配对仅计连续成功对；以独立review及evaluation/tables/analysis的全80二元分母为准。完整raw trial保留本地，manifest保存精确SHA；紧凑证据与冻结19份科学源码字节快照随提交交付。新checkout可用 `gradient_price_source_restore` 核查并仅恢复换行差，再按config/seeds重跑。原报告第13节原位交付理论与本轮结果，PDF交付核验另存delivery_verification.json。

正边竞争的两H固定尺度验收已完成：216任务、432次局部AC拟合、3639次AC调用和1728状态独立重放，全部任务保留。一个正边竞争可将部分模型距离减少约10.5%，但历史任务权重下总分修正不足0.2%，两条固定曲线的排序未变；据此保留推导与模块，尚未部署新选择器。见 `pairwise_cone_checks.py`、`pairwise_cone_ac_scale.py` 与 `evidence/structured_information/pairwise_cone_ac_scale/two_H_analysis.md`。新模块显式读取紧凑 `inputs.json.gz`，针对性测试无需旧未跟踪pilot原始记录。

固定门槛的随机校准集容差界与独立CSV核验已补全。预声明n39、alpha=.1、delta=.05给出rank39；对原80保存损失的事后诊断，任一方法误拒11/80降为6/80，信息设计唯一正确57/80变55/80。原rank36正式策略及全部记录保留；本诊断未提供新前瞻评价，star/weak条件风险仍待解决。见 `calibration_tolerance_math.md`、`calibration_tolerance_audit.py`。

上一阶段全套890项通过记录属于当时源码；随后紧凑输入与CSV核验修改的当前42项针对性测试通过。原报告已原位编译并检查为67页；整个研究仍有proper预测EIG、弱支路风险、滚动与扩网等未完成项。

## 单节点完整梯度价格（2026-10-13，本轮实施）

先读 `gradient_price_protocol.md`、`gradient_price_math.md` 和 `gradient_price_checkpoint.json`。
每轮只在一个节点短时接入，少数节点可逐个比较；连续功率曲线由 max-min 仿射 MILP 优化并经完整 AC 联合重拟合接受，不使用完整曲线库。
`gradient_price.py` 对新增计量计算完整 P/Q/V 与根总表响应价格。历史固定，竞争线路跨历史与未来共享，未来真实总功率及同点逐时源可补偿。半SSE损失与代码预测减读数的符号已核对。
`single_site_milp.py` 包含接入、移动前缀、充放互斥、效率、SOC及同电量任务；普通充电器默认不放电。
`gradient_price_search.py` 保留可行恒充，以完整AC实际局部分数决定接受，另用两条由优化生成的初值处理零梯度。恒充失败不直接删除节点。
均值预测是首版规划近似；固定投影下它与随机价格的梯度相同，活动边界改变时另作核验。边际价格是随机单位增量均值的极限，有限档位增益为价格积分。
文献正文定位见 `gradient_price_literature.json`；理论与实际实验分别报告。当前实施仍是局部多初值搜索，其MILP界针对仿射子问题。
本轮独立比较由 `gradient_price_study.py` 生成新证据，不修改下文冻结C/D策略或旧结果。

完整结果见 `gradient_price_results.md`；正式科学源码8478af6。6pilot+12独立校准+24独立评价已完成，696测试及1176份实际保存拟合的独立AC重放通过。
仅保留真拓扑：无EV0/24、固定恒充7/24、固定节点优化7/24、随机节点优化8/24、设计节点6/24；设计中star5/5与weak9/9均未决。候选漏真、必要拟合/物理失败均0/24；全六方法任一误拒1/24。
完整/PQ-only动作24/24相同；后续梯度对最佳操作优化初值的新增收益约为0。本批未显示设计优于恒充。同质仿射饱和功率定理解释初值有效性的条件，完整AC的全局最优仍待研究。
新证据在 `evidence/gradient_price/v2_supported/compact/`，逐场景/RX/有符号裕量分析在 `v2_supported/evaluation/analysis.md`。本地raw保持原样；compact只删两个SVD数组并保存raw哈希。原库仅在两条pilot作12计划独立对照，主selector未调用。
原报告本轮原位更新为54页；完整研究未终结，下一步的预测、随机价格和弱边验收条件见结果稿。
Git换行转换的首次归档失败已保留；13份冻结科学源码原始字节另存science_source_snapshot.json.gz。新checkout先用gradient_price_source_restore默认只读核对，需要时用--restore-line-endings；仅换行差可恢复，代码变化则停止。独立Git源码目录恢复后13/13哈希相符。



## 本轮指标修订与恒充保护（2026-10-13）

先读 ev_metric_revision_math.md、ev_metric_revision_protocol.md、ev_metric_revision_checkpoint.json。
主选择器继续用时间展开 MILP 联合配置路线与连续功率；原动作库只作对照。
新增 ev_baseline_guard.py 将指定恒充用操作 MILP 实现为完整可行计划，并以同代理 QP 增益区间决定采用或回退；失败和原搜索输出保留，尚未改旧搜索内部 incumbent。
审查发现旧代理固定未来末端 P/Q，而实际 AC 拟合会调整它们。新的方向性全计量指标及历史 offset MILP 已完成推导，尚待接入完整 AC。
future_power_metric_check.py 的两时段 MILP 验证固定/开放未来功率的最优时段可相反；固定指标所选动作按开放指标评价低于恒充。该结论限于线性解析算例。
本阶段 72 项针对性集成测试通过，其中 33 项新增；此前 522 项全套记录保留。证据在 evidence/ev_metric_revision/，新策略未进入 C/D，原报告待 AC 接入阶段更新。

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

本目录conftest.py排除evidence/outputs中的冻结源码镜像；原第一次532项通过的记录含10项历史toy镜像，另保留其日志。上一阶段活动研究测试522项通过（126.65秒），实际日志见evidence/adaptive_final_test_validation.json。不要把历史快照重复当新增测试覆盖。
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
