# 附加绝对误差门诊断

原primary规则保持不变。附加规则分别以0.05校准相对SSE差和每观测SSE；空集表示全部候选未通过绝对误差门。
空集保留在误拒分母；计算不完整保留全部。压力场景的触发率是经验结果。

|方法|推断|总数|误拒|唯一正确|唯一错误|未决|空集|失败保留全部|漏真|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|none|fixed_total|72|8|36|3|30|3|0|0|
|none|fixed_holdout|72|9|15|3|51|3|0|0|
|none|joint_total|72|13|39|4|22|7|0|0|
|constant|fixed_total|72|10|15|2|52|3|0|0|
|constant|fixed_holdout|72|10|14|2|54|2|0|0|
|constant|joint_total|72|8|49|1|15|7|0|0|
|highlow|fixed_total|72|10|16|1|52|3|0|0|
|highlow|fixed_holdout|72|9|16|2|53|1|0|0|
|highlow|joint_total|72|14|54|4|4|10|0|0|
|square_fixed|fixed_total|72|10|17|0|51|4|0|0|
|square_fixed|fixed_holdout|72|9|18|2|51|1|0|0|
|square_fixed|joint_total|72|13|55|3|4|10|0|0|
|square_mobile|fixed_total|72|10|17|0|51|4|0|0|
|square_mobile|fixed_holdout|72|9|18|2|51|1|0|0|
|square_mobile|joint_total|72|13|55|3|4|10|0|0|
|nearest_fixed|fixed_total|72|11|18|1|49|4|0|0|
|nearest_fixed|fixed_holdout|72|10|19|3|49|1|0|0|
|nearest_fixed|joint_total|72|13|55|3|4|10|0|0|
|gradac_fixed|fixed_total|72|9|15|0|53|4|0|0|
|gradac_fixed|fixed_holdout|72|9|16|2|53|1|0|0|
|gradac_fixed|joint_total|72|12|56|3|4|9|0|0|
|scalar_fixed|fixed_total|72|11|18|1|49|4|0|0|
|scalar_fixed|fixed_holdout|72|10|19|3|49|1|0|0|
|scalar_fixed|joint_total|72|13|55|3|4|10|0|0|
|scalar_hist_fixed|fixed_total|72|11|17|1|50|4|0|0|
|scalar_hist_fixed|fixed_holdout|72|10|17|3|51|1|0|0|
|scalar_hist_fixed|joint_total|72|13|55|3|4|10|0|0|
|scalar_mobile|fixed_total|72|9|2|1|67|2|0|0|
|scalar_mobile|fixed_holdout|72|7|1|1|70|0|0|0|
|scalar_mobile|joint_total|72|12|54|2|6|10|0|0|
