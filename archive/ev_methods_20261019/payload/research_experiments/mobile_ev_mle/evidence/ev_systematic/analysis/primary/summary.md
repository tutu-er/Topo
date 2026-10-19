# EV多场景统一对照

历史槽数：72；不同seed：72。

M=min(竞争者SSE)−真候选SSE；较大表示该次观测的排错裕量较大。不同推断路线分别校准。

|方法|推断|完整/总数|平均M|中位M|对同域平方W/T/L/缺失|平均规划秒|平均总秒|
|---|---|---:|---:|---:|---|---:|---:|
|none|fixed_total|72/72|6.9538|6.79261|21/0/51/0|0|1.73815|
|none|fixed_holdout|72/72|1.58476|0.209113|21/0/51/0|0|1.73815|
|none|joint_total|72/72|7.26995|6.44781|2/0/70/0|0|4.38178|
|constant|fixed_total|72/72|20.3884|36.1513|22/0/50/0|0|1.73854|
|constant|fixed_holdout|72/72|14.7873|28.3167|22/0/50/0|0|1.73854|
|constant|joint_total|72/72|29.814|28.6353|5/0/67/0|0|4.37793|
|highlow|fixed_total|72/72|45.067|59.2212|22/0/50/0|0|1.73825|
|highlow|fixed_holdout|72/72|39.531|51.2096|22/0/50/0|0|1.73825|
|highlow|joint_total|72/72|76.2122|81.1528|7/0/65/0|0|4.30005|
|square_fixed|fixed_total|72/72|130.773|215.028|0/72/0/0|0.165202|1.90312|
|square_fixed|fixed_holdout|72/72|125.172|212.909|0/72/0/0|0.165202|1.90312|
|square_fixed|joint_total|72/72|189.131|212.366|0/72/0/0|0.165202|4.36285|
|square_mobile|fixed_total|72/72|130.773|215.028|0/72/0/0|0.837015|2.57493|
|square_mobile|fixed_holdout|72/72|125.172|212.909|0/72/0/0|0.837015|2.57493|
|square_mobile|joint_total|72/72|189.131|212.366|0/72/0/0|0.837015|5.03466|
|nearest_fixed|fixed_total|72/72|130.19|224.951|21/28/23/0|1.14717|2.8855|
|nearest_fixed|fixed_holdout|72/72|124.552|219.513|21/28/23/0|1.14717|2.8855|
|nearest_fixed|joint_total|72/72|191.923|222.456|24/28/20/0|1.14717|5.3725|
|gradac_fixed|fixed_total|72/72|133.487|248.578|19/41/12/0|0.605786|2.34402|
|gradac_fixed|fixed_holdout|72/72|127.894|236.668|19/41/12/0|0.605786|2.34402|
|gradac_fixed|joint_total|72/72|189.059|204.115|15/41/16/0|0.605786|4.82876|
|scalar_fixed|fixed_total|72/72|131.054|228.762|24/25/23/0|1.09016|2.82839|
|scalar_fixed|fixed_holdout|72/72|125.415|223.324|24/25/23/0|1.09016|2.82839|
|scalar_fixed|joint_total|72/72|191.959|222.456|25/25/22/0|1.09016|5.3116|
|scalar_hist_fixed|fixed_total|72/72|131.054|228.762|24/25/23/0|1.10503|2.84344|
|scalar_hist_fixed|fixed_holdout|72/72|125.415|223.324|24/25/23/0|1.10503|2.84344|
|scalar_hist_fixed|joint_total|72/72|191.959|222.456|25/25/22/0|1.10503|5.33014|
|scalar_mobile|fixed_total|72/72|68.9084|169.665|28/16/28/0|3.14155|4.87991|
|scalar_mobile|fixed_holdout|72/72|63.347|159.402|28/16/28/0|3.14155|4.87991|
|scalar_mobile|joint_total|72/72|173.014|177.08|28/16/28/0|3.14155|7.36139|

固定/联合补偿归因：δ_k=S_F,k−S_J,k；竞争者b固定为联合拟合后最近的错误候选。
同一b的差距满足 g_J=g_F+δ_true−δ_b；固定与联合的最近竞争者不同时，直接比较两边M不满足这个同b表达。

|方法|完整配对/总数|正确→错误|错误→正确|状态未变|涉及并列|平均δ_true|平均δ_b|恒等式最大误差|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|none|72/72|3|1|68|0|6.60941|7.06823|0|
|constant|72/72|3|16|53|0|75.9765|87.8413|1.13687e-13|
|highlow|72/72|1|19|52|0|184.621|216.827|2.27374e-13|
|square_fixed|72/72|1|19|52|0|446.182|517.977|4.54747e-13|
|square_mobile|72/72|1|19|52|0|446.182|517.977|4.54747e-13|
|nearest_fixed|72/72|1|20|51|0|449.618|526.171|4.54747e-13|
|gradac_fixed|72/72|1|19|52|0|440.984|546.128|4.54747e-13|
|scalar_fixed|72/72|1|20|51|0|448.404|525.474|4.54747e-13|
|scalar_hist_fixed|72/72|1|20|51|0|448.404|525.474|4.54747e-13|
|scalar_mobile|72/72|0|25|47|0|434.679|408.818|4.54747e-13|

|方法|推断|误拒/总数|唯一正确|唯一错误|未决|失败保留全部|漏真|
|---|---|---:|---:|---:|---:|---:|---:|
|none|fixed_total|12/72|43|8|21|0|0|
|none|fixed_holdout|10/72|20|5|47|0|0|
|none|joint_total|12/72|49|9|14|0|0|
|constant|fixed_total|10/72|26|3|43|0|0|
|constant|fixed_holdout|11/72|23|3|46|0|0|
|constant|joint_total|9/72|61|9|2|0|0|
|highlow|fixed_total|8/72|26|2|44|0|0|
|highlow|fixed_holdout|8/72|25|2|45|0|0|
|highlow|joint_total|4/72|68|4|0|0|0|
|square_fixed|fixed_total|10/72|35|4|33|0|0|
|square_fixed|fixed_holdout|10/72|35|4|33|0|0|
|square_fixed|joint_total|3/72|69|3|0|0|0|
|square_mobile|fixed_total|10/72|35|4|33|0|0|
|square_mobile|fixed_holdout|10/72|35|4|33|0|0|
|square_mobile|joint_total|3/72|69|3|0|0|0|
|nearest_fixed|fixed_total|10/72|36|4|32|0|0|
|nearest_fixed|fixed_holdout|10/72|35|4|33|0|0|
|nearest_fixed|joint_total|3/72|69|3|0|0|0|
|gradac_fixed|fixed_total|9/72|37|3|32|0|0|
|gradac_fixed|fixed_holdout|9/72|36|3|33|0|0|
|gradac_fixed|joint_total|3/72|68|3|1|0|0|
|scalar_fixed|fixed_total|10/72|36|4|32|0|0|
|scalar_fixed|fixed_holdout|10/72|35|4|33|0|0|
|scalar_fixed|joint_total|3/72|69|3|0|0|0|
|scalar_hist_fixed|fixed_total|10/72|36|4|32|0|0|
|scalar_hist_fixed|fixed_holdout|10/72|35|4|33|0|0|
|scalar_hist_fixed|joint_total|3/72|69|3|0|0|0|
|scalar_mobile|fixed_total|11/72|24|3|45|0|0|
|scalar_mobile|fixed_holdout|11/72|24|3|45|0|0|
|scalar_mobile|joint_total|4/72|68|4|0|0|0|

未决列只含已完整计算、仍保留多个候选的历史；计算失败另列并保留全部候选。
阈值针对声明的IID混合分布，分场景计数不承诺同样覆盖；分数和似然不表示拓扑正确概率。
