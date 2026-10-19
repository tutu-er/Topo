# 优化策略内部的排序诊断

每条历史只对四种优化策略计算预测平均分数与实际M的Spearman相关，使用实际执行的动作；回退恒充保留其相同分数和并列秩。全部九策略的相关作为对照。M为最近错误候选联合SSE减真结构候选SSE。

|响应|四优化策略相关的中位数|范围|有效历史|九策略对照中位数|
|---|---:|---:|---:|---:|
|actual_M|0.8|[0.2,1]|12/12|0.8979|
|noiseless_joint_M|0.8|[-0.4,1]|12/12|0.8983|

|历史|家族|四策略有噪声ρ|四策略无噪声ρ|九策略有噪声ρ|九策略无噪声ρ|
|---|---|---:|---:|---:|---:|
|988401|star|0.8|0|0.7|0.2833|
|988402|cherry_01|0.8|1|0.9322|0.9831|
|988403|cherry_02|0.8|0.8|0.8655|0.8992|
|988404|cherry_12|0.2|0.4|0.9328|0.8992|
|988405|star|0.8|1|0.759|0.8554|
|988406|cherry_01|0.4|0.4|0.7479|0.7479|
|988407|cherry_02|0.8|0.8|0.8983|0.8983|
|988408|cherry_12|1|1|0.9492|0.8983|
|988409|star|0.4|-0.4|0.5966|0.07563|
|988410|cherry_01|0.8|0.8|0.9496|0.9328|
|988411|cherry_02|0.8|0.4|0.8974|0.7436|
|988412|cherry_12|1|1|1|0.9333|

## 每个预声明配对跨历史的增量相关

分别对每个配对计算Δ预测平均分数与Δ实际M的相关。每个配对最多12条历史；配对之间不合并样本。

|配对|响应|有效/计划历史|Spearman ρ|缺失历史|
|---|---|---:|---:|---|
|charge_time|delta_actual_M|12/12|0.4895|[]|
|charge_time|delta_noiseless_joint_M|12/12|0.4615|[]|
|fixed_V2G|delta_actual_M|12/12|0.8252|[]|
|fixed_V2G|delta_noiseless_joint_M|12/12|0.7762|[]|
|charge_space|delta_actual_M|12/12|0.4406|[]|
|charge_space|delta_noiseless_joint_M|12/12|0.3916|[]|
|V2G_space|delta_actual_M|12/12|0.7552|[]|
|V2G_space|delta_noiseless_joint_M|12/12|0.7762|[]|
|mobile_V2G|delta_actual_M|12/12|0.4685|[]|
|mobile_V2G|delta_noiseless_joint_M|12/12|0.4056|[]|
|full_vs_constant|delta_actual_M|12/12|0.2797|[]|
|full_vs_constant|delta_noiseless_joint_M|12/12|0.3706|[]|
|full_vs_quality|delta_actual_M|12/12|0.7029|[]|
|full_vs_quality|delta_noiseless_joint_M|12/12|0.7174|[]|
|full_vs_random|delta_actual_M|11/12|0.8091|[988405]|
|full_vs_random|delta_noiseless_joint_M|11/12|0.8|[988405]|

Spearman使用平均秩处理相同分数；某列全相同时相关未定义，保存为空。固定家族配额下仅作描述，未计算p值、置信区间或拓扑正确概率。原失败、回退和配对缺失均保留；方法数与候选方向数不增加独立历史数。
