# 冻结 C 阶段结果与紧凑证据归档

C 已完成 cal128 / eval256。refit 相对 constant 的唯一正确保留增加 9/256，增益集中在 strong cherry；这批结果没有支持 star 或 weak cherry 的稳定唯一识别，也没有支持 refit 在七种方法中整体最好。七方法至少一次误拒真候选的历史为 14/256（5.46875%）。本结论仅针对冻结 nominal 模拟总体和固定数值策略。

本包从既有只读输出复制、压缩和独立重算，没有运行新 AC 优化、调整策略或门限、改动正式源码或覆盖旧输出。原始 JSON/NPZ 仍保留在来源目录，完整输入 manifest 和 15 个核心源码冻结 SHA256 见 source_manifests.json。已有独立审查原文及数字原样复制。

## 分母、失败与结果定义

评价单位为 256 条独立生成器历史。每条历史 7 方法、1 轮、4 候选，得到 1,792 条方法记录和 7,168 条候选拟合；方法/候选记录是历史内的配对测量，不是额外独立样本。所有计数保留完整 256 分母，事件可重叠。唯一正确保留要求真候选在库中、拟合完整、运行电压检查通过且联合保留集合仅含真候选。选址 fallback 另作失败标记，并另列 algorithm_unique_correct_retention。

| 方法 | 唯一正确保留 | 误拒真候选 | 未决 | 错误唯一保留 | 选址失败 | 拟合失败 |
|---|---:|---:|---:|---:|---:|---:|
| none | 19 | 7 | 230 | 0 | 0 | 0 |
| constant_fixed | 62 | 4 | 190 | 0 | 0 | 0 |
| fixed_pulse | 61 | 8 | 188 | 0 | 0 | 0 |
| random_action | 70 | 6 | 181 | 1 | 0 | 1 |
| legacy_library_proxy | 81 | 9 | 166 | 3 | 1 | 0 |
| fixed_mle_milp | 68 | 7 | 181 | 1 | 0 | 0 |
| refit_milp | 71 | 6 | 179 | 1 | 0 | 0 |

legacy_library_proxy 的选址失败 730085 与 random_action 的联合拟合失败 730180 均未删除；后者 clade_01、clade_02 的 optimizer success=false。错误唯一、空保留、真候选误拒不是同一事件；它们的逐历史列表和掩码可分别核查。全部方法的真候选遗漏、电压失败为 0。

## 冻结校准与拒绝规则

校准共有 128 历史，完整 127、计算失败 1。每条历史的 relative score 是七方法/所有声明轮次上“真 loss − 该方法最小候选 loss”的最大值，absolute score 是真 loss 的最大值；任何必要拟合失败在两组分数中记 +∞。calibration_scores.csv 保存全部 128 条（inf 文本表示正无穷），失败历史没有从校准分母中排除。

两门限分别使用 α=0.05，k=ceil((128+1)(1−0.05))=123。第 123 个顺序统计量为 relative=8.610311359932325、absolute=75.77255697515467。严格以大于门限拒绝；单门限秩尾界 (129−123)/129=6/129，合计 12/129=0.0930232558，冻结 nominal 总预算为 0.1。必要候选拟合失败时两拒绝掩码全 0、保留全部候选并记录失败。正常联合掩码为 relative OR absolute。

七方法联合误拒为 14/256；既有独立审查的 95% Clopper–Pearson 区间为 [3.02172%,9.00548%]。秩控制要求策略预先固定、同一声明模拟总体、历史独立/可交换，且对校准样本随机性与新历史共同取边际；它不是给定当前校准集后的条件保证、逐历史概率、各层风险保证或连续参数域统一保证。未用评价结果反调门限。

## 分层结果

以下各层分母由评价历史的真拓扑/案例确定，属于描述性分层。

| 层 | 分母 | 方法 | 唯一正确保留 | 误拒 | 未决 | 错误唯一 | 选址失败 | 拟合失败 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| star | 65 | none | 0 | 2 | 63 | 0 | 0 | 0 |
| star | 65 | constant_fixed | 0 | 1 | 64 | 0 | 0 | 0 |
| star | 65 | fixed_pulse | 0 | 2 | 64 | 0 | 0 | 0 |
| star | 65 | random_action | 0 | 4 | 62 | 1 | 0 | 0 |
| star | 65 | legacy_library_proxy | 0 | 2 | 63 | 1 | 0 | 0 |
| star | 65 | fixed_mle_milp | 0 | 3 | 62 | 1 | 0 | 0 |
| star | 65 | refit_milp | 0 | 3 | 62 | 1 | 0 | 0 |
| weak_cherry | 102 | none | 0 | 2 | 100 | 0 | 0 | 0 |
| weak_cherry | 102 | constant_fixed | 0 | 2 | 100 | 0 | 0 | 0 |
| weak_cherry | 102 | fixed_pulse | 0 | 2 | 100 | 0 | 0 | 0 |
| weak_cherry | 102 | random_action | 1 | 1 | 100 | 0 | 0 | 0 |
| weak_cherry | 102 | legacy_library_proxy | 0 | 4 | 98 | 2 | 0 | 0 |
| weak_cherry | 102 | fixed_mle_milp | 0 | 2 | 100 | 0 | 0 | 0 |
| weak_cherry | 102 | refit_milp | 0 | 2 | 100 | 0 | 0 | 0 |
| strong_cherry | 89 | none | 19 | 3 | 67 | 0 | 0 | 0 |
| strong_cherry | 89 | constant_fixed | 62 | 1 | 26 | 0 | 0 | 0 |
| strong_cherry | 89 | fixed_pulse | 61 | 4 | 24 | 0 | 0 | 0 |
| strong_cherry | 89 | random_action | 69 | 1 | 19 | 0 | 0 | 1 |
| strong_cherry | 89 | legacy_library_proxy | 81 | 3 | 5 | 0 | 1 | 0 |
| strong_cherry | 89 | fixed_mle_milp | 68 | 2 | 19 | 0 | 0 | 0 |
| strong_cherry | 89 | refit_milp | 71 | 1 | 17 | 0 | 0 | 0 |

weak 层 random 的一次唯一正确保留是 730121（真 clade_12）：真 loss=74.98534762927387，三个错候选 loss=78.18523061739937、78.18603612507276、78.19502542529348。relative gate 全保留，absolute 门限 75.77255697515467 恰好留下真候选；本例保存在 exception_cases.json，不能用它推断弱支路稳定分离。

## 配对比较的方向与结论边界

refit 对 constant 的唯一正确保留：两者都成功 62，refit 单独成功 9，constant 单独成功 0，两者都未成功 185；收益 +3.515625 个百分点。既有审查的保守配对 95% 区间为 [−0.26212,7.04877] 个百分点，双侧 exact McNemar p=0.00390625。多方法、多终点或事后分层比较另有比较族问题，不能把该 p 值当拓扑正确概率。

collector 的 generic paired_binary improved/worsened 仅表示事件 0→1 / 1→0。对误拒、未决、拟合失败，0→1 是不利变化。refit 对 constant 的误拒为新增 2、消除 0，频率差 +0.78125 个百分点，p=0.5；不能写成误拒改善。legacy 的唯一正确保留 81 高于 refit 的 71，故未证明 refit 七方法整体最优。

AC replay 全部 7,168 为 verified，验证的是保存目标值与方程重放；全局最优证书为 0。代理距离和实际 signed margin 的关联较弱（refit raw proxy 的 Spearman ρ=0.12801），不能把代理目标直接解释为新增未来辨别信息。阻抗统计的单位先在历史内合并边，再跨历史汇总；star 的 internal 成分不适用，不记零。

原 numerical_bounds.csv 为 7,920,099 字节，未复制进紧凑包，但其 SHA256 和来源路径保存在 evidence_manifest.json。评价负 raw gap 6 处、最大相对 crossing=6.503972900110877e−7；校准负 gap 3 处，原值与 crossing 记录保留。容差接受不等于严格上下界有序，也不提供非线性 AC 设计最优性证书。

## 复核入口

- history_method_masks.csv.gz 保存全部 1,792 方法历史的 relative/absolute/combined/retained 掩码与事件；bit 顺序固定为 star、clade_01、clade_02、clade_12，最低位为 star，拒绝掩码 1=拒绝，retained_mask 1=保留。
- originals/evaluation_v3_audit/fits.csv.gz 与 actions.csv.gz 解压后逐字节等同原 CSV，分别为 7,168 / 1,792 数据行。原始与压缩文件 SHA256 均在 evidence_manifest.json。
- compact_counts.json 保存独立重算方法/各层计数、联合误拒历史 ID。该重算与完整 audit_summary 的所有事件计数及既有独立审查一致；5,376 组门掩码、26,880 个事件值比较零差异。
- verify_C_compact.py 仅用 Python 标准库验证归档 SHA256、解压 SHA256、校准 rank、全历史掩码与计数，不调用项目源码或优化器。命令：python verify_C_compact.py。
- source_manifests.json 保存 cal257 / eval513 个输入文件哈希、15 个核心冻结哈希、3 个后处理哈希，本次均读原文件匹配。

证据强度止于冻结模拟总体、有限四候选类和保存的局部拟合；没有单次拓扑正确概率、现场泛化、连续全域识别或全局 MLE 结论。
