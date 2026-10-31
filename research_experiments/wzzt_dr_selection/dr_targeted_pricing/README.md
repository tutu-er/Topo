# 目标块：实际影响预计算与报价优化

本批先从既有最终失败筛选机制子集，保留原零收益控制。完整识别器仍fit_observed，不重写搜索/选模。

- MODEL_AND_PLAN.md：完整方案、执行状态、预算和严格有限模型；最上方独立训练/验证四价补充优先。
- prepare_baseline.py：已完成paper15资格与flynn主例当前288/288复核，不重复。
- run_precomputed.py：基础prepare/observe planning、复用fit/finish、目标机制跟踪及最终aggregate。
- decision_layer.py：独立source绑定的四价扩展prepare/optimize/test；不增加基础执行状态或W调用。
- optimizer.py：训练/验证重复两价的受限对照。
- event_optimizer.py：四独立报价主模型，50价向量/方案、100可行联合列，先risk后费用。

主例flynn16/b4/v0005，C=201-203；现有bank和完整方法均漏C，主障碍在验证3%带。201/204执行，202/203被动。物理和参数保持原.0005/.005、288/288→480/480、预算1.6、25秒/3%/.16；明示合成参与规律不称现场EV校准。

实际顺序（root为private wzzt_targeted_pricing_v1/flynn_run_v1）：
1. 基线已完成；run_precomputed prepare/observe planning已完成；decision_layer prepare在读取影响损失前已完成。
2. run_precomputed fit --phase planning --worker 0/1正在运行，检查锁/ended，勿重复。
3. run_precomputed finish --phase planning，全fit冻结后评分；decision_layer optimize从真实16表取风险，冻结oracle。
4. decision_layer test生成新实际观测；run_precomputed fit --phase test两worker；finish test；aggregate。
5. 独立审计和最终白名单汇总，原轨迹/连续权重仅私有。

不要调用run_precomputed的旧重复报价optimize/observe test入口。该入口仅保留基础源版本兼容，真实测试使用新的四价决策层/整数2×2报价。

四价扩展前的base protocol、32物理状态和source保持不可变；新decision_protocol与event_price_decision_lock绑定新层。执行状态完整，因此价格扩展只重新积分，不补造或重跑损失值。

这是有限影响库的精确优化，不是全连续DR全局最优。影响表不可按孤立模块收益相加，多资源多轮的高阶交互需要完整联合状态或有验证误差的近似。
