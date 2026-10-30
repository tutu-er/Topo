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
