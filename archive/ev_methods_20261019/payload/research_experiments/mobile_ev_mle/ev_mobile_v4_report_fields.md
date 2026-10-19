# v4最终报告的只读字段映射

本映射只读取分析器/审计器源码，未读取任何部分评价。等8条冻结评价全部写完并由root通知后，才执行以下提取。

## 输入及完整性

使用完整的analysis JSON：phases.evaluation。要求history_denominator=8、records_present=8；overall的history_regime_denominator=16、independent_history_count=8、method_slot_denominator=192、candidate_slot_denominator=768；planning.expected_plan_slots=96。缺失仍保留分母，若不完整则在结论写明，不只取成功子集。

独立audit的exact_case_coverage及source_hash_checks先核对，再用totals/maxima、recorded_failure_counts。物理核查成功与优化达到全局最优分开。

## 关键配对表

四组by_bank_regime：n3/regular、n3/shift、n4/regular、n4/shift。每组pairs的键为a+'__vs__'+b。四列依次为：

- guarded__mobile_bidir 对 guarded__fixed_bidir；
- guarded__mobile_bidir 对 guarded__mobile_charge；
- guarded__mobile_bidir 对 aligned__mobile_bidir；
- guarded__mobile_bidir 对 square__mobile_bidir。

提取delta_M.mean/median、wins/ties/losses、complete/missing/denominator。正文表优先均值与胜平负，证据JSON保留全部。每组4条历史；整体16个未来情景相关，不报告基于16个独立样本的显著性。若要整体按历史概括，先对同一历史两个regime的配对增量求平均，再对8条历史汇总；同时保留regime分层，不能只报更有利的一层。

补充同域4个guarded−aligned、guarded−square，以及各族FD−FC、MC−FC，区分放电、移动和保护规则的作用。若两方法动作key相同，差异应为0，不能计作多次独立验证。

## gap分类

- 取得值继承：monotonicity，8历史×3方法族×4包含边=96。square承诺Q，aligned/guarded承诺同一完整Φ。
- 已接受保护：planning.guarded_accepted_floor_checked、violations、unknown；失败/未知保留，不能把没有接受新提案当作大幅提高辨识。
- 聚合取舍：rows[].parent_comparisons中is_selected_parent=True的average_metric_sacrifices_true_reference。此名义事件在两个regime重复；按历史index、family、child去重后计数，不能重复算两次。
- 名义到实际：每regime分别统计selected parent的predicted_mean_gain_actual_loss、predicted_true_reference_gain_actual_loss及对应comparable分母；两个regime是同历史配对未来。它们包括预测、未来计量/输入误差及数值拟合，不单凭这些字段作因果归因。
- 预定因果诊断只用index0、6，guarded MD对FD，三因素8组合。来源为另外完成的离线归因，不使用真值重新选动作。不把总残差项全部称为预测均值误差。
- 全动作排序可用overall.ranking及分组ranking，报告严格对/反转/并列/缺失，但动作对并非独立历史，不给伪样本量。

## 计算与失败

planning的QP_attempts/completed、原物理域line/source failure、master_attempts/completed、master_gap、compression_used用于代理求解情况。真实完整AC目标未有全局证书。

family_costs的standalone_seconds及soft_budget_exceeded按每策略整条四域180秒解释；公共参数拟合/预测费用单列。报告每历史成本，不能只报最后一次cache命中的时间。

planning.top_planning_failures及histories[].planning.rows中的step.failure按历史×方法×域计，不重复两个regime。overall.methods中的physical_status、fit_status和candidate_slots按192/768实际槽计。guarded回退但实际成功时，优化失败与物理失败同时如实区分。

最终报告由ev_mobile_v4_results_template.md填写约500–1000汉字及关键表；所有精确数据留在完整JSON证据。没有稳定优势时直接写未建立优势，说明被解决的指标不一致与仍存在的预测/计算问题。
