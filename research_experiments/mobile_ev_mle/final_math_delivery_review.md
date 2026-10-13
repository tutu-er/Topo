# 最终数学交付独立审查

审查日期：2026-10-13。对象为 research/ev-history-t-design 的共享功率平方上界、连续 proper H-only 积分和名义弱支路信息界，以及正在整合的 mobile_ev_probing_report_v2.tex。结论：在下列声明条件下，未发现改变结论的数学或目标分布错误。连续积分结果必须保持 failed，不能提供可信 posterior、EIG 排序或拓扑正确概率。

本次仅阅读当前数学、源码和已有审查材料，并保存审查文本与输入 SHA-256。没有启动优化、AC 求解、Monte Carlo、测试或逐粒子数值重算；没有读取实际 future/truth/noise NPZ，也没有修改 docs、冻结 C/D、旧 outputs 或 Git。已有实验数字及测试通过数由报告和现有审查材料提供，本次不重新验证这些数字。审查主要核对代数、量词、信息流和实现公式；没有重新证明完整 inverse 域的全部常数，也没有核验外部论文全文。纯数学/源码材料已经足够完成本次范围，因此无需额外文献下载。

## 1. 共享平方图

- 分解成立需要两种不同条件：实际 B 每个残差行的精确非零支撑只属于一个时段，使跨时段列内积为零；单 EV 单位路径及 stay 功率绑定，使同一时段不同站点的净功率乘积为零。同期 B 列不必正交。certify_time_block_support 对 row != 0 精确检查，连 1e-300 跨时段非零也拒绝；operation_model 的 DAG 单位整数流与功率绑定提供第二种条件。多 EV 一般不能直接复用分解。
- 弦线 s-u^2=(u-a)(b-u)>=0；d=||B[:,j]||^2>=0，故有限可行见证给 Q<=W<=S。零节点使未激活站点误差为零，单 EV 的 PWL 误差可界为 sum_t max_i d_ti Delta_ti^2/4。取达到 W 的有限见证可得 S-W 的 uniform 上界；该界不包含 W-Q 的有限见证误差。
- 添加 T<=qbar 后，对任意可行 u 仍可取 T=Q(u)，因此不破坏对 max Q 的上界方向。上界 master 是最大化；LinearMILP.solve 对 SciPy 最小化输出 fun/mip_dual_bound 同时反号，score_from_result 再乘正 scale，q_proxy_upper 来自 best bound。q_incumbent 不能替代全域上界。
- 浮点 RHS guard 和求解器 best bound 仅支持声明代理的数值上界；没有 interval arithmetic 证书，也不能转成完整 AC 设计全局界。固定动作 AC 可行回代只给相应 nuisance infimum 的可行见证上界。
- 主要证据：shared_power_square_upper.py:14-29,55-83,93-113,130-183；ev_operations_milp.py:110-126,247-295；ev_information_milp.py:177-199；report_shared_upper_fragment.tex:54-108,110-153。

## 2. 连续积分、一次历史更新与 ESS

- 每个成分的权重必须为 alpha_k/N_beta,k * pi_k/q_k * Lhat_H；证据估计为全部成分/参数样本权重之和。当前等样本数时，删共同 1/N_beta 不改变 normalized weights 或 ESS，却改变证据尺度。源码保留该因子。
- 固定 H、beta 和已冻结归一化 proposals，并要求 proposals 覆盖目标支持后，每时刻 importance mean 无偏；独立的时间样本云才使时间均值乘积 Lhat_H 对完整历史似然无偏。跨 beta/mode CRN 仅引入相关性，不破坏每个固定 beta 的该结论。该结论不表示 normalized posterior weights 无偏。
- guided latent 在白化坐标计算 baseline-meter Gaussian 与完整 Gaussian mixture 的密度比。相同 affine map 的 Jacobian 在比值中抵消，故等价于在共同 dP dQ dS0 测度恢复完整计量/proposal 比值；非线性 voltage、背景密度、源密度和 inverse Jacobian 仍进入目标。有限差分只塑造 proposal。
- beta 的 SequentialGaussianBoxProposal 逐坐标保留条件截断 normalizer、完整 mixture density 和 beta Jacobian；它是顺序条件截断 proposal，不是简单的整盒截断 multivariate Gaussian。
- 工程 Lhat_H^kappa 一般不以 L_H^kappa 为期望，因此不是 exact tempered posterior。最终只继承 normalized proposal 参数；fresh beta 与 fresh independent-time latent 云构造原 pi L_H 权重。prefix L32 仅复用 fresh beta/logq 和前32个核，重新替换 H likelihood，不乘旧 H weights。代码一致。
- 总体和成分 ESS 是定义明确的权重集中度诊断；它们不是误差界、概率或数值积分证书。六个退化成分均须保留，显示质量下溢和有限 MC 零证据不能成为精确模型拒绝。global ESS 提高不解除所有成分 gate。
- 主要证据：continuous_eig_sampling.py:52-58,450-530,582-655；continuous_eig_guided_latent.py:65-92,129-240；continuous_eig_proposals.py:126-158,160-205；continuous_eig_final_helpers.py:24-71；continuous_eig_final_run.py:79-91,103-121,133-180,188-200；continuous_eig_sampling_math.md:45-107,229-257。

## 3. 名义弱边与分类下限

- 基础完整 AC/KL 界覆盖全部声明独立 R/X 名义盒；辅助 star 的两叶各增加固定 2z_s 后，参数先验通常不同于真正名义 star，必须补概率桥。
- 条件于 z_s，四个独立 leaf R/X 均匀坐标被平移；等体积盒的 TV 等于1减交叠比例。每坐标平移不超过其盒宽，最大平移给 d_ij<.033。向真正 star 增加独立辅助 z_s、再经共同因果观测 kernel，是合法的数据处理步骤；积分 z_s 后两平移叶的相关性不影响此界。
- KL/Pinsker 和 prior-TV 三角得到 weak clade 与名义 star 的完整单条策略轨迹 TV<.267521。等先验二元 forced-classifier 风险下限、JS<=log(2) TV、该二元人口平均 future conditional information 上界成立；后者不是每个 realized H 的 conditional EIG，也不是四标签 EIG 上界。
- 四拓扑 forced classifier 映射到 star/weak 组，拓扑错误包含组错误，因此风险至少 integral min(.25 pstar,.375 pweakmix)>=.25(1-TV)>.18311975。剩余 strong 部分只贡献非负风险。此风险不等于允许弃判/多候选集合的误剔除率。
- 范围须保持：同一共同非前瞻控制器、共享线路只抽一次、相同逐时外生过程及噪声、精确功率/电量 caps、单条实际策略轨迹。不得拼接共享未来噪声的七方法 counterfactual；不得用理想 raw Gaussian 定理直接保证 success-conditioned 数据；数值可行容差不自动证明字面精确 caps。
- 主要证据：weak_branch_information_math.md:56-228；nominal_weak_information_math.md:13-107；nominal_weak_peer_review.md:13-29,59-69；nominal_weak_information.py:9-29,41-55。

## 4. 交付用语核查与小幅建议

报告已区分论文框架、自行推导、实现与有限实验，已明确源没有由当前 WZZT 默认接口估计，局部多初值拟合不称全局 MLE；共享上界不替换正式 C/D，continuous failed 不产 EIG 排序。这些边界应保留。

审查中通知整合者的三项文字补充均不推翻实际代码：

1. 在 r/g 和无偏公式处明确 q_k 覆盖 pi_k 的支持，g_k,beta,t 在目标核非零处为正；仅归一化不足以推出重要性恒等式。本实现的 uniform 参数混合和 baseline Gaussian latent 混合提供覆盖。
2. r 的分子宜称潜在状态与实际计量的联合密度核；它不是已对 x 积分的 marginal meter density。期望宜明确条件于固定 H、beta 和已冻结 proposals。
3. JS 后的 future conditional information 宜写该等先验二元实验的人口平均，避免与 .25/.375 的四拓扑总体 forced-risk 推导混淆。

非阻断命名建议：HistoryBank.summary 的 posterior_available 仅表示 log evidence 有限、数值粒子 bank 可构造。failed 交付须继续以 posterior_accepted=False 为准；以后可改名 numerical_particle_bank_available。本次未改源码。

附属 evidence JSON 记录本次读取对象的字节 hash。docs 正在由整合者更新，该 hash 仅代表保存时的读稿，不是最终交付冻结 hash。本审查不声称复核他人已报告的测试执行或逐粒子重算。

## 整合后复核

已重新只读检查整合稿：上述三项文字补充均已写入正文。提案覆盖支持、固定 H/beta/冻结提案下无偏性及联合密度核见当时正文1684-1729行；等先验二元人口范围见1558行。七个成分验收及连续结果失败的结论继续保留。没有新增阻断性发现。正文仍由整合者进行排版与冻结，本次 hash 不替代其最终交付 hash。
