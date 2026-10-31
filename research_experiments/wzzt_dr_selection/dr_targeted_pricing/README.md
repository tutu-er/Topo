# 目标块DR：实际影响预计算、报价优化和独立核算

本轮已全部完成；先读[ACTUAL_TEST_REPORT.md](ACTUAL_TEST_REPORT.md)和统一[MATHEMATICAL_MODEL.md](MATHEMATICAL_MODEL.md)。MODEL_AND_PLAN.md保留事前基础协议及四价补充。

真实结果：无奖励自然新增与简单轮换各3/4完整恢复，优化报价2/4，未通过独立验证。planning32请求/31unique，test20请求/15unique，4871固定族LP，失败未决0。有限初值准入和3%最简回选是关键作用环节；不归因为末端收缩。

已有主识别器fit_observed未重写。完整历史288/288保留至480/480，原25秒/3%/.16、.0005/.005、预算1.6与零弱权重保持。当前主参照为已知错误块/已知生成模型，一个规划自然背景；合成参与律未校准。

文件：
- prepare_baseline.py：paper15资格与flynn主例当前基线，已结束勿重跑。
- run_precomputed.py：基础状态预计算、复用fit/finish/aggregate。
- decision_layer.py / event_optimizer.py：四独立报价主模型，100可行列；全部16状态积分后优化并冻结，再生成独立test。
- optimizer.py：训验重复两价的受限对照。
- export_results.py：终态白名单导出；拒绝未冻结/运行中状态。
- audit_results.py：独立评分、概率、报价、费用、接受和选模核验。--root PRIVATE --helper ../dr_day_modules/audit_day_trial.py --out NEW_FILE；输出文件须新路径，0新fit/AC。
- review_terminal_mechanism.py：已存准备点与真实选择的只读机制核验，--root PRIVATE --phase planning或test --out NEW_FILE --allow-terminal-read；反事实不是新算法实验。

全部private artifact位于visualizations任务根的wzzt_targeted_pricing_v1/flynn_run_v1，base/decision协议、源zip、history、planning/test、oracle、summary、audit和机制报告齐全。evidence/targeted_pricing_v1仅含汇总/离散支持/检查/hash，不上传家庭、PQV/AC轨迹或连续拟合参数。

不要重新运行32状态或20测试，不调用run_precomputed旧两价optimize/observe test入口。没有新可检验规则前，不扩大本例种子、噪声、预算或网络。
