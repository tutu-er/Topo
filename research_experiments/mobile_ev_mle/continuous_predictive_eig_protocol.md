# 连续 proper 预测 EIG 的独立诊断尝试（不改冻结策略）

目标：检查当前 finite prior 适合度失败能否由更忠实的连续参数/状态表示改善。以已冻结 nominal 生成分布为一个明确的 proper 合成概率模型，不称现场 prior，也不以 MLE 为 prior。所有模型从 H 更新一次；WZZT同数据不再计独立似然；EV加入总P一次。

拓扑 prior 四类均匀；cherry强/弱为两分量各.5；逐边独立R/X按各分量canonical参数的[.8,1.2]连续均匀先验。源逐时P/Q及末端共同背景、独立偏移严格使用 nominal 的已声明有界iid过程。bias/root固定校准、追踪1。这些先验范围来自运行前 nominal 声明；不看 evaluation 真值/未来。更宽参数 prior及质量权重敏感性单列。

先保存完整变化变量推导。用潜在末端总P/Q、主表实际P/Q代替源坐标；观测的Gaussian P/Q密度可用相同Gaussian proposal抵消，留下电压密度、背景/源prior和变化变量Jacobian。高电压AC逆映射、唯一性与Jacobian需独立核查，不得凭点值认为积分正确。共用背景变量c可通过所有终端的区间交长度精确消去。

先小型解析AC engine/导数/坐标变换与项目完整AC比较；再少量H-only importance积分。proposal若由H拟合得到，必须计proposal密度，不能把其当prior；使用独立重复流、ESS/权重集中度、积分层数增加及源码/配置哈希。ESS不足、积分不稳定保留失败，不报可信排序。

动作只取已保存pilot H和该H的少量冻结优化/恒充方案。proper predictive Monte Carlo仅抽声明模型下的假想未来，不读实际未来noise/真拓扑。用共同随机数作动作配对，再独立重复验证。报告后验熵、EIG、积分误差/偏差诊断、prior敏感与排序变化；概率仅属于声明proper模型，不能叫现场正确率。

验收：数学变化变量与Jacobian成立；engine独立AC验证；proper模型/一次H更新/importance密度正确；少量积分稳定才报告数值，否则交付不可用原因与直接续接点。本尝试不改C/D/stress策略、阈值或正式评价；不用于调策略。
