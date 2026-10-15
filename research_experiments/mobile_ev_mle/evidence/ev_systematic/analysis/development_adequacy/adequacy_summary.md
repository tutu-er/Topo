# 附加绝对误差门诊断

原primary规则保持不变。附加规则分别以0.05校准相对SSE差和每观测SSE；空集表示全部候选未通过绝对误差门。
空集保留在误拒分母；计算不完整保留全部。压力场景的触发率是经验结果。

|方法|推断|总数|误拒|唯一正确|唯一错误|未决|空集|失败保留全部|漏真|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|none|fixed_total|64|6|23|2|31|4|0|4|
|none|fixed_holdout|64|11|12|2|39|7|0|4|
|none|joint_total|64|9|30|2|22|6|0|4|
|constant|fixed_total|64|7|11|1|44|4|0|4|
|constant|fixed_holdout|64|8|9|1|45|5|0|4|
|constant|joint_total|64|10|35|1|16|8|0|4|
|highlow|fixed_total|64|7|14|1|42|3|0|4|
|highlow|fixed_holdout|64|7|14|1|42|3|0|4|
|highlow|joint_total|64|16|37|3|8|12|0|4|
|square_fixed|fixed_total|64|8|15|2|40|3|0|4|
|square_fixed|fixed_holdout|64|7|15|1|41|3|0|4|
|square_fixed|joint_total|64|15|40|4|6|10|0|4|
|square_mobile|fixed_total|64|8|15|2|40|3|0|4|
|square_mobile|fixed_holdout|64|7|15|1|41|3|0|4|
|square_mobile|joint_total|64|15|39|4|7|10|0|4|
|nearest_fixed|fixed_total|64|6|13|1|43|3|0|4|
|nearest_fixed|fixed_holdout|64|5|15|1|42|2|0|4|
|nearest_fixed|joint_total|64|16|39|4|6|11|0|4|
|gradac_fixed|fixed_total|64|6|13|1|43|3|0|4|
|gradac_fixed|fixed_holdout|64|6|14|1|42|3|0|4|
|gradac_fixed|joint_total|64|16|39|4|6|11|0|4|
|scalar_fixed|fixed_total|64|6|13|1|43|3|0|4|
|scalar_fixed|fixed_holdout|64|5|14|1|43|2|0|4|
|scalar_fixed|joint_total|64|16|38|4|7|11|0|4|
|scalar_hist_fixed|fixed_total|64|6|12|1|44|3|0|4|
|scalar_hist_fixed|fixed_holdout|64|5|14|1|43|2|0|4|
|scalar_hist_fixed|joint_total|64|16|38|4|7|11|0|4|
|scalar_mobile|fixed_total|64|4|2|0|56|2|0|4|
|scalar_mobile|fixed_holdout|64|4|4|1|54|1|0|4|
|scalar_mobile|joint_total|64|15|38|3|8|11|0|4|
