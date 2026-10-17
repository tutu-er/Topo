# 既有贝叶斯响应误差定位

本检查只重算已存证据；新增 AC、拟合、MILP、观测和计划调用均为 0。

预定历史 128，完整可评价 88，缺失 40。两个未来状态先在每份历史内平均，32 次噪声不增加独立样本量。

贝叶斯移动充放电：预计风险 0.1828%，实际误判 27.7876%。
零误判/全部误判/混合历史：53/16/19。
最差 20/25 份历史占全部错误的 81.02%/96.68%。
全部误判历史平均置信 98.643%，预计风险 0.0046%。

## 无噪评分定位

下面是历史×未来状态的诊断次数，并非独立历史数或平均风险。

|方法|可得状态|联合评分错|去历史权重后错|历史权重救回|历史权重翻错|欧氏最近但联合评分错|
|---|---:|---:|---:|---:|---:|---:|
|joint_bc__mobile_bidir|176|48|53|5|0|11|
|square__mobile_bidir|176|47|49|2|0|8|
|constant|176|82|87|9|4|16|

贝叶斯方案无噪评分为负的 48 个状态，实际平均误判 96.7448%；其余状态为 1.9287%。前者承载全部实际错误的 94.9521%。这些是对保存数据的描述性拆分。

欧氏最近而联合评分错反映均值偏差、协方差加权及历史权重的耦合，不能仅归因于协方差大小。

## 描述性关联

- posterior_R_relative_error 与误判的 Spearman：0.1672，历史 n=88。
- posterior_X_relative_error 与误判的 Spearman：-0.2251，历史 n=88。
- historical_truth_log_weight 与误判的 Spearman：-0.3160，历史 n=88。
- historical_truth_weight 与误判的 Spearman：-0.3160，历史 n=88。

## 预定历史 0 的独立重放

|方法|状态|原错误/32|统一权重|真均值替换|真均值+统一权重|去P/Q边际|
|---|---|---:|---:|---:|---:|---:|
|joint_bc__mobile_bidir|regular|28|31|0|0|28|
|joint_bc__mobile_bidir|shift|29|32|0|0|29|
|square__mobile_bidir|regular|28|31|0|0|28|
|square__mobile_bidir|shift|29|32|0|0|29|
|constant|regular|4|8|0|0|4|
|constant|shift|0|0|0|0|0|

历史 0 的已存完整 AC 重拟合（只读取，未新增拟合）：

- joint_bc__mobile_bidir / regular：AC 选 star，M=0.784852；贝叶斯同首个噪声选 clade_12；全局证明=False。
- joint_bc__mobile_bidir / shift：AC 选 star，M=1.030151；贝叶斯同首个噪声选 clade_12；全局证明=False。
- square__mobile_bidir / regular：AC 选 star，M=0.784852；贝叶斯同首个噪声选 clade_12；全局证明=False。
- square__mobile_bidir / shift：AC 选 star，M=1.030151；贝叶斯同首个噪声选 clade_12；全局证明=False。
- constant / regular：AC 选 clade_12，M=-0.652763；贝叶斯同首个噪声选 star；全局证明=False。
- constant / shift：AC 选 star，M=0.287219；贝叶斯同首个噪声选 star；全局证明=False。

真均值替换使用真值，仅定位综合均值误差；不可部署，也不构成性能上界。一个历史的敏感性不能推广为总体改善。

与既有汇总逐项核对 24 项；完整 JSON 保存缺失、分层、配对结果、各竞争者三项分数及所有输入哈希。
