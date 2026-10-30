# 公开恢复证据：WZZT响应结果表与随机价格

本阶段实现并有界测试，主DR恢复收益尚未取得。64条实际配对记录、44种不同拟合均恢复；无奖励同样恢复。只有一条共享四日自然轨迹，不是64个独立样本。v1的1项未决及31项未启动完整保留。

入口是 ../../dr_competition_bridge/REPORT.md、METHOD.md 和 NEXT_CODED_DR.md。生产默认未改，原自动任务保持关闭。本阶段没有待继续的求解进程；不要以当前源码直接重跑已冻结的旧protocol。

文件范围：

- summary.json：完整分母、费用及本阶段结论。
- actual_records.json.gz：64条价格、结果与运行来源元数据，无观测轨迹。
- actual_fits.zip、fit_index.json：44份筛选后的拟合、路径、LP及选模元数据。index分别记录原私有文件hash和公开成员hash；公开文件不是原文件的逐字节副本。
- stage_records.zip：各版本协议、响应mask结果、冻结价格、噪声诊断、独立审计和160条件的候选层级盘点。只保留既定元数据；完整P/Q/V、AC物理时序和家庭负荷不上传。
- source_snapshot.zip：本轮源码及相关core、冻结物理生成器和独立审计。包含旧v1源码zip，避免v2提案变化覆盖失败版本。文档于交付时校正了零效应评价与P/Q抵消条件；实际运行代码以各protocol记录的源hash为准。
- manifest.json：公开文件、源码成员及压缩记录hash。

私有恢复根均位于 C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/：

- wzzt_dr_competition_bridge_v1：16完成、1未决，原源码及封存回执。
- wzzt_dr_competition_bridge_v2：48/48局部规划、720 LP。
- wzzt_dr_competition_site_v3：第二站对48/48、720 LP。
- wzzt_dr_competition_validation_noise_v4：仅预测验证噪声重评分，0新LP/AC。
- wzzt_dr_competition_actual_v2：64实际记录、44不同拟合、3246 LP，含独立电气/冻结核验。

公开证据足以核对汇总、拟合及选择来源；完整物理重放仍需上述私有观测与冻结生成器环境。下一编码机制仅有模板、给定设备边界检查和块响应恒等式测试，尚无新价格或物理恢复结果。
