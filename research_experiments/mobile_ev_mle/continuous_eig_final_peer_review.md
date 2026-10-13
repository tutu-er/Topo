# 最终 continuous proper H-only v5 独立审查

结论：未发现改变目标分布、重复使用 H 或破坏 L32/L128 配对的实现错误。最终数值结果必须保持 **failed**；该审查确认实现与失败报告一致，不接受 posterior、拓扑排序或 EIG。

审查仅阅读指定 continuous 源码、数学/计划、已有测试和 H-only v5 数值交付。没有读取原 trial、实际 future/truth/noise 数据，没有运行实际 H 或 predictive MC，没有修改运行源码、primary 或 Git。源码 snapshot 审计包含配置列出的依赖文件字节哈希；未读取该依赖的研究数据。此处不重新证明整个 inverse 域定理，而是核对已接受定理的调用边界及数值失败处理。

1. `continuous_eig_sampling.py:601` 与 `continuous_eig_final_helpers.py:38` 都直接构造 `log(mode_mass)-log(B)+logπ-logq+log L_H`。prefix 仅复用 fresh β/logq；不读取旧 logweight。六个 raw bank 的有限权重逐元素重算误差为 0，非有限值模式完全一致；重新汇总 logZ 的误差为 0。未把 pilot 的 `Lhat^κ` 权重带入最终 estimator。
2. pilot 只输出工程 proposal 参数，并明确 `E[Lhat^κ] != L^κ`。final driver 只加载 frozen proposal 的均值、协方差和 uniform mass；完整 mixture q 包含顺序条件 normalizer 和 β Jacobian。用 frozen q 重算六个 bank 的 logq，最大误差均为 0。source snapshot 的九个源码哈希及 final plan、frozen q 哈希均匹配预先配置。
3. `guided_latent_likelihood` 从一次 nonlinear kernel 的最后轴切出前 32 项；full 和 prefix 分别取各时刻均值再乘时间项。三重复的 L32/L128 β 与 logq 在 raw bank 中精确一致。独立重跑的 prefix 与 chunk 测试通过；切块不重新采样，不改变配对。时间独立云保持 joint likelihood 的无偏性，跨 β/mode 的 CRN 不要求独立。
4. 所有七个 mode 都保留。mode gate 直接检查每个 mode ESS，不依赖显示质量是否下溢；zero finite-MC mass 也不解释为精确模型拒绝。原门槛 global ESS≥100、每 mode ESS≥20、最大权重≤0.05 未降低、豁免或删除。
5. 在 prior-supported β、有限 background density 和 master rectangle 内，inverse/branch/certificate/正物理 V/正 Jacobian 的失败会 raise。`invalid_inverse_count` 因此只统计 background/master 支持外的不可用 inverse 点，不能当作支持内 numerical failure。v5 各 mode 的支持内失败计数为 0，master rectangle 支持外计数为 0；非零 invalid inverse 事件位于 background 支持外。真正 source-prior 支持外的点保留为零目标；Gaussian 测量负 V 仍允许。
6. θ 权重 sample SE 是条件于固定 common latent cloud 的诊断。由随机 H 权重选择的 latent ESS/relSE 是样本矩，不是误差界；joint sample-moment relSE 不等于 unconditional evidence SE。由于共享 latent CRN，不应把两层诊断直接相加或平方和合成为总 SE。当前代码/报告已明确这些限制。

|指标|L32 三重复|L128 三重复|
|---|---|---|
|global ESS|427.868 / 232.348 / 200.340|437.317 / 259.058 / 193.939|
|最大权重|0.021487 / 0.054495 / 0.050630|0.020579 / 0.049621 / 0.052114|
|其他六 mode conditional ESS|均约 1|均约 1|
|logZ 重复范围|2.346539|0.504244|

配对 `logZ128-logZ32` 为 −0.183144、1.659152、0.076554。每 mode 配对 log evidence 差异覆盖约 −0.320 至 25.210，不能只看 dominant mode 或 global ESS。L128 第三重复还违反最大权重门槛。failed 结论由 mode ESS 门槛直接确定，不需要把这些波动解释为置信区间。latent doubling 改善部分样本诊断，但没有解决所有模式的积分退化。

独立测试命令：`python -m pytest -q test_continuous_eig_sampling.py test_continuous_eig_proposals.py test_continuous_eig_guided_latent.py test_continuous_eig_tempered_proposal.py test_continuous_eig_final_helpers.py -p no:cacheprovider`；结果 **56 passed in 2.78s**，数值线程为 1。测试检查代数、归一化和数值边界，不证明实际 H 积分充分。

唯一非阻断用语建议：`continuous_eig_sampling.py:578` 的 `posterior_available=True` 只表示 log evidence 有限、数值粒子 bank 可构造。failed 交付中应继续以 `posterior_accepted=False` 和 `posterior_values_credible=False` 为准；将来可把前一字段改名为 `numerical_particle_bank_available`，避免读者误解。此建议不影响已执行的目标或失败结论，本次没有改动源码。
