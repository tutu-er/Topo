# 连续 proper 模型：H-only 数值诊断交付

五版均保留，proper prior、AC 物理模型和 Gaussian meter 目标未改。最终目标始终为原 π(β)L_H(β)，历史只更新一次。工程 pilot 的随机 Lhat^κ 权重不进入最终 estimator，也不是精确 tempered posterior。

|版本|每模式 β / 每时刻 L|三重复 global ESS|最大权重范围|logZ 重复范围|秒|
|---|---:|---|---:|---:|---:|
|v1|2048 / 32|1.000, 1.000, 1.000|1.0000–1.0000|25.561187|10.14|
|v2|2048 / 32|2.234, 3.709, 5.356|0.3692–0.6565|0.887723|13.71|
|v3|2048 / 32|33.821, 36.086, 12.314|0.1054–0.2736|0.647419|46.79|
|v4|2048 / 32|21.027, 9.350, 10.506|0.1502–0.3103|0.962367|85.84|
|v5 L32|4096 / 32|427.868, 232.348, 200.340|0.0215–0.0545|2.346539|502.74|
|v5 L128|4096 / 128|437.317, 259.058, 193.939|0.0206–0.0521|0.504244|502.74|

v5 用每重复相同 fresh β/logq 和同一 per-time L128 nuisance cloud 的前32项，分别重算 H 权重。配对 logZ128−logZ32：-0.183144, 1.659152, 0.076554。

最终三重复中，clade_02_strong 的 conditional ESS 为437.32、259.06、193.94；另六模式均约1，未达到20。第3重复 L128 最大权重0.0521也超过0.05。故即使 global ESS 全部超过100，最终仍明确 failed。

L128 的 clade_02_strong 条件加权 per-time latent ESS 平均90.58–92.76/128、relative SE 平均0.0570–0.0597、joint likelihood relative SE 样本矩0.2371–0.2474；其他模式的 joint 样本矩0.8311–2.4485。每模式原始 per-time 数组和独立重复证据跨度均保留。

原 gate 保持 global ESS≥100、每个模式 conditional ESS≥20、最大粒子权重≤0.05，并要求重复/积分稳定；显示质量很小不能免检。所有模式 conditional ESS 和 latent ESS/relSE 保存在 JSON。

θ 层 sample SE 条件于固定 latent cloud；latent 层 relSE 是样本矩诊断，由随机 H 权重选择，不能当误差界。跨 β 共用 latent CRN 使无条件 β/latent 方差分解无法从单个 bank 唯一识别。独立 repeats 和配对 doubling 提供额外诊断。

实际支持内 inverse / finite target 数值失败强制终止并留档。outside background/master 支持的 inverse-unusable 计数属于 proposal 零目标事件，不能当支持内 numerical failure，也不能等同模型概率零。

未接受 posterior，未运行 predictive EIG 或给动作排名。未改 formal C/D/stress 或 primary；无 Git。初始 trial schema 意外暴露已在数学 ledger 说明；正式 reader 只访问 sanitized H-only 六数组，NPZ future/latent/noise 未索引。

继续的可核查瓶颈：按每模式 ESS、paired log evidence 与 conditional latent variance 判断 β proposal 是否仍偏离；不能删 rare mode、复用 pilot weights、把 finite-MC 零当模型拒绝，或在未再次冻结配置前继续大规模 MC。

源代码与数学/测试交付清单见 continuous_eig_handoff_manifest.json；各版本 config/source_snapshot/rawbanks 保留 staging，raw NPZ 不推送。
