# EV统一比较：既有算例库存与可执行矩阵

2026-10-16，只读盘点。canonical分支 `codex/ev-train-validation-design`，本次核对HEAD `1007559dfc17abc498df9885db96d5cc91aadae5`。没有修改旧源码、旧记录或C盘；本文件是新增盘点交付，没有运行拟合、选择器或新评价。

路径缩写：

- `C` = `C:/Users/23761/.codex/worktrees/36a1/Topo`。
- `D` = `D:/0-github_workspace/Topo/outputs/ev_rx_stage_20261015`。
- 下表源码均在两处的 `research_experiments/mobile_ev_mle/`；证据目录除另注外相对 `C/research_experiments/mobile_ev_mle/evidence/`。

## 1. 主要判断

已有算例远多于最新4份开发历史，但主要成熟流程仍是三末端。可以复用的范围包括：强/中/弱内部支路、star及三个cherry、相关/独立负荷、训练8/32/96点、统一/时间/空间计量精度、同接口源漂移、偏差失配、候选漏真、公开低压电缆及case33五线派生尺度、一次五末端扩网。旧五末端只有1份历史；case33派生仍只有3末端，均不构成完整IEEE33网络主动辨识证据。

新比较应使用同一个计量/拟合/操作/接受/评价流程，覆盖这些主要条件；旧分数只用于历史盘点和机制诊断。不同旧版本使用过联合重新拟合线路、固定训练线路、不同任务和不同精度，不能把各版本的平均裕量直接拼成新方法排名。

建议按16场景×4历史先开发，再冻结6个主场景组成的预声明独立混合分布，开展108校准和72评价。缺真、错误参数域和偏差失配单列压力结果，不并入模型正确主分布的校准。正式阈值先保存后读取评价。

## 2. 已有算例、实际入口及覆盖边界

| 算例族 | 可复用生成/运行入口 | 代表设置 | 已有数据及实际边界 |
|---|---|---|---|
| 最早强/弱三末端 | `simulation.generate(seed,case='strong'/'weak',n=3,history_count=12,future_count=8,truth_family=...)`；`run_study.one_trial` | 400V、20kW；干线R/X=.005/.0035pu，叶线R=.028+.003j、X=.011+.002j；内部strong=.012/.007，weak=.00003/.00002 | `evidence/formal_summary.json`、`formal_trials.json`；各20种子×强弱2条件，40校准槽+40评价槽；跨强弱条件配对，基础种子共40。旧11动作库仅作动作库对照，不再作主选择器。n3候选覆盖仅指star/三个cherry声明类。 |
| 偏差失配/漏真 | `run_study.one_trial(seed,'bias_misspec'/'missing_truth',...)`；CLI `run_study --phase stress` | V真实偏差改为[-.00065,0,.00065]pu；missing_truth预置真cherry01后从库删除它 | `stress_summary.json`、`stress_trials.json`，各4历史。属于模型/覆盖压力，和正常主条件分开。 |
| 旧五末端扩网 | `simulation.generate(seed,'strong',n=5)`；`candidates.propose(...,max_candidates=8)`；CLI `run_study --phase scale` | 默认真支持 `((0,1),(0,1,2))`，独立P/Q变化更大 | `scale_summary.json`、`scale_trials.json`；seed71001、1历史、8候选，真候选在库。五法最终赢家均错；没有扩网校准。候选是有限被动/局部修补集，不完整。 |
| 中间强度/精度族 | `information_study.configuration`、`information_study.make_path(seed,cfg,case=...,truth_family=...,quality=...)` | H12/F4；star/weak/moderate/strong；内部R/X分别weak=.00003/.00002、moderate=.003/.002、strong=.012/.007pu；逐边lognormal σ=.10 | `structured_information/{v1,v2_targeted,v3_frozen}`。统一/时间/空间精度及历史创新幅度.004/.012pu可复用；旧信息指标另列。 |
| 同接口源/负荷预测失配 | `adaptive_milp_study._evaluation_path(seed,stress='source_drift'/'load_trend'/'bias')` | 未来源P加.10pu、Q加.035pu；或未来8点P共同增加0→.05pu；bias加[-.0006,0,.0006]pu | `adaptive_stress_compact`、`adaptive_d_compact`及旧自适应交付。旧滚动规则/阈值与本轮单次固定线路流程分开。 |
| 400V电缆原域MILP | `ev_original_study.make_case(seed,family,short_edge=False)`；`FAMILIES=((),((0,1),),((0,2),),((1,2),))` | NAYY50/120/150：R′=.642/.225/.208Ω/km，X′=.083/.080/.080；干线40m，叶线80/100/120m，内部80m，长度±5%；H12/F6 | `ev_original_domain/{pilot_v2,evaluation_v2,domain_contrast,location_ablation_v1,compact}`。全部四候选；竞争线路/负荷/源可在历史+未来重新拟合。不能与固定训练参数结果混为同模型。 |
| 短支路与参数域失配 | 同上 `short_edge=True`；`fit_options('numeric'/'asset')` | 内部边2m；numeric R/X∈[1e-6,.1]pu；asset为25–150m电缆族范围 | `ev_original_domain/{pilot_numeric_v2,stress_asset_v2,stress_numeric_v2}`。2m真参数在asset域外；结构覆盖和参数覆盖须分开。star没有内部边，是无短边的对照。 |
| 紧凑MILP/预测解释 | `ev_compact_study`，底层仍复用 `ev_original_study.make_case` | 共享功率平方、合法历史参数解释、负荷/源/追踪情景；H12/F6，原时变精度 | `ev_compact/{development_ensemble_v2,calibration_ensemble_v2,evaluation_ensemble_v2}`；4开发/36校准/12评价。可复用资产与既有曲线，原多解释策略另作参数可调路线。 |
| 固定训练线路/训练支持 | `ev_fixed_train_study.run_trial`；`ev_fixed_train_training_support.make_training_inputs(seed,family_index,length)` | weak8；rich8/32/96：P=2.4±.6kW、Q=.8±.3kvar，各户独立；源2±.25kW、.7±.1kvar；验证4/F6相同 | `ev_fixed_train/development_v2`、`training_support_dev_v1/{rich8,rich32,rich96}`。16条件仅4配对独立资产；无正式校准。rich前缀的训练终点变化是该旧机制试验的特定设置。 |
| 普通时变负荷主链 | `ev_dynamic_loads.generate_background(seed,count,n=3)`、`fit_forecast`；`ev_dynamic_study.make_case(seed,family_index,cfg)`、`actual_curve` | H96/V4/F6，每槽15min；P均值2.4/2.8/2.1kW，共同AR(1) SD=.12、φ=.9，各户SD=.25、φ=.8；PF=.97±轻波动；源P均值2kW、SD=.15，Q均值.7kvar、SD=.06 | `ev_dynamic`的开发/校准/评价raw.tar.xz，4开发×两精度、36校准、12评价。每拓扑一组训练线路、P/Q输入、当期只拟合同接口1源；AR仅使用训练。 |
| 物理X/R约束及四域强基线 | `ev_rx_study.train_bank`、`run_trial`，仍调用dynamic生成；`ev_rx_fit.fit_rx_inputs` | ρ=X/R∈[.083/.642,.080/.208]，原R/X盒保持；四域固定/移动×纯充/双向，平方精度/均值/最近竞争者 | `D/outputs/rx_{development_box,development_cone,calibration,evaluation}`；canonical `ev_rx/*raw.tar.xz`。4配对开发、36校准、12评价；全部三末端四候选。 |
| 梯度完整流程 | `ev_gradient_flow_study.run_trial`、`select_plans`；`ev_gradient_flow_design.ProfileEngine` | 同dynamic/RX；13法、固定/移动双向，AC/I²梯度，训练前二pairac只规划聚焦，最终仍评价四候选 | `D/outputs/gradient_flow_{development_v2,calibration,evaluation}`；4/36/16历史。前二漏真在评价为1/16，原全库漏真0；不能将pairac称严格二候选试验。 |
| 严格二候选/源投影指标 | `ev_response_metric_study.declared_bank`、`run_trial`；dynamic生成、rx训练、actual_curve评价 | star/clade01预先固定，仅site0双向；9法，联合源投影/源盒泄漏/负荷情景 | `D/outputs/response_metric_{development_v1,calibration,evaluation}`；4/18/8历史。没有WZZT shortlist，覆盖由该两族声明构造保证。 |
| 新标量L2弦MILP | `ev_scalar_distance_milp`；`ev_scalar_distance_probe.run(directory,output)` | 3/5/9/17网格；旧4开发历史，严格2模型、site0双向；完整AC接受 | `D/outputs/scalar_distance_development_v1`。仅开发4历史/16网格条件；尚非独立校准/评价，网格累计计时也非独立同预算政策比较。通用多候选版另行实现。 |
| case33标准尺度派生 | `ev_physical_cases.case33_local_standard()`；`ev_physical_spatial_fit_audit.generated_history(seed)` | 原5条Ω线路；12.66kV/1MVA，末端bus4/19/23，120/90/90kW；H12/F4，源10±2kW、3±1kvar；同(6,0,6,0)kW | `ev_ev_ess_spatial/physical_fit_v1`。2历史、40最终拟合；真结构clade02，未选址。原图删支路并将保留隐藏节点负荷归零；不是完整case33。 |
| 公开低压矩阵参数族 | `ev_lv_parameter_family.public_lv_parameter_family(length_m,cable_codes=('C201','C250'),base_kv=.4,base_mva=.02)` | 固定revision公开4×4电缆矩阵，验证平衡电流后投影；显式长度 | 只提供线路参数与来源，不生成结构、负荷或计量；没有推断现场长度或下界。若替换NAYY，应新命名来源，不悄然替换旧生成器。 |

## 3. 计量、操作和评价的共同口径

`simulation.model(n)`单读标准差为P=.0006pu、Q=.0004pu、V=.00012pu、根表P/Q=.001pu。20kW基准下为.012kW、.008kvar、根表.02kW；1000kW基准下则为.6kW、.4kvar和1kW。标准化后仍须报告实际单位、EV/当地负荷比和电压信号/噪声比，不能只比较6kW或pu。

旧H12/F4的 `information_context.quality_models` 为统一5次、时间[1,9,9,1]、空间矩阵 `[[9,1,1],[1,9,1],[1,1,9],[9,9,9]]`，三者总电压读数均60。H96/F6主链为每末端各槽5次，总90读数；时间对照[1,9,1,9,1,9]也总90。固定偏差不随重复平均消失。新空间条件如沿用旧F4，明确F4，不能直接冒充F6同预算。

当前电池任务来自 `ev_fixed_train_design.CONFIG`：60kWh容量、24→26.85kWh，ηc=ηd=.95、每槽.25h、命令上限6/1.05kW、吞吐≤7.5kWh；最多移动1次，驾驶和接线各1槽、驾驶.2kWh、人员30min，抵达后有最小吞吐要求。四操作域完全共享同一任务，固定点是site0。跨纯充/双向另报吞吐，跨移动另报失去的时槽、驾驶补电和人员。

当前固定训练路线的实际损失是验证SSE+未来SSE，训练R/X固定，P/Q读数直接输入，逐时同接口源重拟合。训练负责估参、候选和预测；验证不进入规划。`actual_curve.clean_future_M`仅去除未来噪声，仍含原噪声验证误差，不能称完全无噪声风险。

建议统一保留none/constant/highlow/square/nearest/grad/scalar及声明的history-offset变体，fixed/mobile分别比较。随机及旧11动作库只作附加对照。主比较同时报：完整AC M、相对none及square的配对增量、校准排错/误拒/未决、候选漏真、代理界、实际AC接受、动作来源、失败、资源和独立部署耗时。冻结代理界与最终继承的计划分开保存。

## 4. 本轮可执行开发矩阵：16行，每行4历史

除表内变化外，保持NAYY四真结构、四候选、普通AR96/V4/F6、统一精度、同接口源、每拓扑单组训练线路和RX物理范围。四历史通常循环star/cherry01/cherry02/cherry12；第15、16行的结构例外显式列出。每行固定/移动及各算法在同一未来和噪声下配对。每行内四历史独立；训练长度、参数域等跨行配对会复用同一资产与未来，因此64是场景—历史槽数，独立基础历史数以seed/asset清单去重另报，不能直接写成64个独立样本。

| ID | 改变的条件 | 生成/适配路径 | 覆盖及验收重点 |
|---|---|---|---|
| S01 | 普通AR96，统一精度 | dynamic原入口 | 作为所有变化的共同锚点；完整四候选。 |
| S02 | 同轨迹训练末32点 | 先生成96+4+6，最后截训练末32 | 训练终点、验证、未来及噪声与S01相同，forecast gap仍4。 |
| S03 | 同轨迹训练末8点 | 同上截训练末8 | 保存训练失败/弱信息/响应外推失败；不换有利种子。 |
| S04 | 高负荷相关 | 新adapter复用cross的高相关构造，独立创新.00025pu，或明确移植到AR | 只改负荷相关参数；若移植AR，标为新变体而非原cross逐字复现；R/X补偿。 |
| S05 | rich32独立P/Q训练 | `make_training_inputs(...,32)`保留原配对holdout | 复用已存在的训练支持家族，评价同域新方法；与AR主行有多项背景差异，分别报告。 |
| S06 | 时间精度[1,9,1,9,1,9] | dynamic cfg公开repeat_counts | 总采样30/末端；分清精度择时和拓扑增量。 |
| S07 | 真实源为0，竞争源仍原自由域 | 新adapter将生成sp/sq置0并重算计量 | 这是无真实源但未知源解释，不等于已知无源；后者另行命名。 |
| S08 | 同接口未来源漂移 | 移植 `_evaluation_path(...,'source_drift')`：未来P+2kW/Q+.7kvar | 预测只用训练，源位置不变；保留源补偿和误差。该行可作预声明名义预测失配族。 |
| S09 | 2m内部支路，numeric域 | `ev_original.make_case(short_edge=True)`取得资产，再用统一动态数据adapter | 真参数必须在numeric域内；star无内部边作为负对照。保存短边未决。 |
| S10 | 同2m，asset域 | 与S09完全相同资产/计量，只换拟合域 | 明确参数漏真压力，排除出主校准分布；不把窄域效果算作EV增益。 |
| S11 | 未校正电压偏差 | 移植bias_misspec真实偏差，拟合校准不变 | 模型失配压力；主分布之外；固定偏差和随机误差分开。 |
| S12 | 预先删除clade01候选 | 新driver显式声明bank={star,clade02,clade12} | 四真族中只有clade01实际漏真，保留1/4漏真及其余对照；不要按未来选择删谁。 |
| S13 | 接入及人员窗口限制 | `MobileServiceConfig.access/worker_available/setup/travel_*` | 时间展开路径直接配置，不枚举路线；无可行服务保存explicit failure；同域基线共享条件。 |
| S14 | 较长移动窗口 | 同一AR过程，未来12槽，公开任务/时长/精度预先固定 | 必须使用通用平方MILP而非六槽解析曲线；分清更长观测与空间价值；F12同长度none。 |
| S15 | case33五线标准参数尺度 | `generated_history(seed)` → 统一train/fixedθ/动作/eval adapter | 原来源只给clade02，4个独立历史均保留该真结构；不能伪称四结构。若H12切train8/V4，保留其弱训练；扩H另新命名。 |
| S16 | 五末端嵌套结构 | `simulation.generate(seed,'strong',n=5)` → 通用meter/fit/context adapter | 4独立历史均可先用旧nested真族，候选最多8、逐历史记录漏真。动态n3入口不能直接用；无法适配时保留明确失败。 |

可再增加两个便宜敏感性条件，但不暗中更换上述矩阵：未来负荷0→1kW趋势，及旧H12/F4统一/时间/空间精度复现。若要求优先覆盖全部旧质量条件，可用空间精度替代一项尚未适配的扩展，但必须在运行前确定并留修改记录。

正式6主场景建议优先S01/S02/S03/S04/S06/S08；S07可替换S08，但须在冻结前决定。将case和真结构按预声明权重独立抽样，108/72是独立历史总数。阈值若按混合总体校准，只支持该混合总体的边际口径，不能沿用旧“每真族9份”的条件阈值说法。若仍按真族分层，明确每层实际校准分母和有限样本分位数规则。S10/S11/S12不进入该主分布；S15/S16是结构/尺度外推，单独报告。

## 5. 现成接口的限制，实施前必须处理

1. `dynamic.make_case`硬编码3末端和原base资产。改 `training_points` 会移动holdout索引，不能直接得到S01–S03配对。新driver应先生成完整96轨迹，截训练末尾，且fit_forecast使用相同训练末时刻。
2. `ev_dynamic_loads.generate_background`读取全局CONFIG，普通/高相关/源变化不是所有参数都能从study cfg注入。新adapter显式接收声明参数，重新生成物理量和计量；不要只改标签，也不要为新策略修改旧冻结全局配置。
3. `ev_response_metric_study.validate_config`强制2候选、6槽、uniform、site0；`scalar_probe`也是strict2开发工具。新统一driver应调用底层fit/context/MILP/evaluate接口，不能假装旧CLI已支持四候选、多种horizon或n5。
4. H12/F4工具、F6平方解析初值和通用操作MILP的限制不同。F12/n5须有统一可行初值与显式失败路径；原六槽解析最优值不外推。
5. 多候选scalar应保留全部竞争者；最近竞争者聚合和全对均值分别声明。共享lambda节省规模，仍要分开记录冻结MILP全局界和完整AC接受。不能通过仅保留历史前二解决规模而漏记真候选。
6. case33派生依赖原`terminal_case33/data/case33bw_raw.py`，公开电缆矩阵依赖`data_external/lvnetworkdataset_20260920/cables/cableparameters.json`。本次实查C两者均存在，D-stage两者均缺。应明确复制来源/哈希或显式缺依赖失败，不替换成NAYY假称标准参数。

## 6. 实测成本与本轮预算依据

以下只读取保存记录；没有重新运行选择器。近期D-stage记录为实际线程1环境，跨版本工作流内容不同，成本仅用于预估，不能作算法公平速度结论。

| 已有批次 | 历史及方法数 | 整研究工作流每历史 | 单策略成本示例 |
|---|---:|---:|---|
| 最早动作库formal | 80强弱条件槽、40基础种子、5法 | 校准1241.92s/40槽，评价1308.92s/40槽 | 平均选择13.94s；16内层动作失败保留。 |
| 原域EV评价 | 12H、9法 | 1147.12s/12≈95.59s | 全选择953.18s/12≈79.43s；含开放参数解释，非当前固定线路成本。 |
| 普通动态评价 | 12H、14法及消融/价格诊断 | 978.26s/12≈81.52s | 包含AR/均值消融和回放，不能当部署耗时。 |
| RX开发cone | 4H、19法 | 均值68.35s，范围60.90–81.03s | 没有完整统一runtime新字段。 |
| RX校准/评价 | 36/12H、19法 | 54.25/50.35s；评价范围14.93–68.61s | 旧MILP时间限额相同，整策略额外AC成本另计。 |
| gradient正式评价 | 16H、13法 | 均值12.40s，范围10.95–14.41s | 公共训练4.42s；固定square/nearest/grad独立规划=.164/1.362/.585s，独立端到端4.665/5.865/5.087s；mobile nearest=8.592s端到端。 |
| strict2 response评价 | 8H、9法 | 均值4.851s，范围4.484–5.157s | 公共训练2.073s；square/nearest/grad/contrast_joint端到端2.284/2.938/2.395/2.335s。 |
| scalar开发网格 | 4H、4网格 | 全网格规划每H .993–1.249s | 包含新开发路径；不是各网格独立同预算部署时间。 |
| case33局部位置诊断 | 2H、4动作场景/四候选 | 24.29/19.06s | 共40最终拟合、1665拟合AC批调用，未含主选择器。 |
| n5旧动作库 | 1H、5法/8候选 | 68.37s | 单历史、H12/F8，不能预测新H96、多候选精网格总成本。 |

新64H应先做各类型1个pilot检查，保存训练、上下文、每方法MILP、AC接受、未来拟合/回放的成本；普通三末端可参考约15–60s/H整比较，弱训练/短边/标准尺度/n5可能显著更慢。多候选源QP数随有向对数增长，n5不应一开始全部17网格。按实测成本再冻结统一策略的分段/预算；预算不足保留已完成矩阵与显式未运行项，不删最慢或失败条件。

独立部署时间必须把公共训练、操作/平方初值、父域继承和上下文计入各方法一次，祖先去重；不以研究工作流缓存隐去必要成本。同历史动作去重可节约离线评价，但不把共享计算后的低成本写成单方法独立成本。

## 7. 本次盘点的完成范围

已核对源码入口、实际已有证据、各模型的参数/候选覆盖、近期保存runtime和旧结果成本。盘点本身没有编写driver或运行研究。下一阶段的验收是64开发槽逐项有结果或明确失败，全部方法动作/固定θ/源接口/噪声配对一致，完成后再冻结正式6场景协议。

## 8. 盘点后新统一程序的实际设置

随后root新增 `ev_systematic_cases.py/study.py/refit.py`，实际配置以新协议及freeze为准：长窗口是8槽；强源是全程4倍偏差而非未来阶跃；漏真行bank为star/clade01，真结构仅clade02/12，因此4/4漏真；弱边行4个历史中只使用三种cherry，含一个重复结构。case33及n5仍各用固定真结构的4个独立历史。

训练长度三行和弱边域两行复用种子，64开发槽对应52去重基础seed。所有regime先生成108期、白噪声按channel独立流，未来按T截取；纯预测接口为 `(training,meter,fits,config)`。case33标准参数依赖已由root从C复制到D，哈希进入新freeze。

新增独立 `test_ev_systematic_isolation_audit.py` 的17项检查通过：预测/driver信息边界、未来真值篡改、全部训练/验证/未来通道配对、各通道30读数预算、EV物理单位只加一次、实际真结构、开发/正式seed分母与新依赖freeze。3个接口pilot另完成104保存状态/312物理块回放，不作独立策略效果证据。
