# 参数域与EV增量的配对对照

同一历史、真线路、未来噪声、计量及设备条件分别在asset/numeric域运行。4条pilot为开发复用，2条短边为参数域压力；原asset独立评价仍为12条。

M为最近错误候选联合SSE减真结构家族联合SSE；G=M方法−M恒充。G先在各参数域内计算，最后比较两域G，分开参数先验共同作用与各域下的EV增量。

|组别|方法|历史数|asset平均M|numeric平均M|asset平均G|numeric平均G|两域G差(asset−numeric)|
|---|---|---:|---:|---:|---:|---:|---:|
|pilot_domain|none|4|-0.00888|-0.4809|-5.526|-3.534|-1.993|
|pilot_domain|constant|4|5.517|3.053|0|0|0|
|pilot_domain|quality_fixed|4|60.55|51.9|55.03|48.85|6.179|
|pilot_domain|quality_mobile|4|79.43|67.22|73.91|64.17|9.746|
|pilot_domain|random_mobile|4|117.8|87.25|112.3|84.2|28.09|
|pilot_domain|charge_fixed|4|76.67|73.21|71.16|70.16|0.9972|
|pilot_domain|v2g_fixed|4|212.9|138.5|207.4|135.5|71.88|
|pilot_domain|charge_mobile|4|154.9|52.71|149.4|49.65|99.74|
|pilot_domain|v2g_mobile|4|130.6|36.52|125.1|33.46|91.59|
|short_edge_domain|none|2|-0.2649|0.123|3.364|0.3459|3.018|
|short_edge_domain|constant|2|-3.629|-0.2229|0|0|0|
|short_edge_domain|quality_fixed|2|-14.33|0.02663|-10.71|0.2496|-10.95|
|short_edge_domain|quality_mobile|2|-20.72|-0.07389|-17.09|0.1491|-17.24|
|short_edge_domain|random_mobile|2|-10.06|0.06472|-6.435|0.2877|-6.723|
|short_edge_domain|charge_fixed|2|-14.33|-0.2229|-10.71|0|-10.71|
|short_edge_domain|v2g_fixed|2|-33.68|-0.2229|-30.05|0|-30.05|
|short_edge_domain|charge_mobile|2|-18.41|-0.2229|-14.78|0|-14.78|
|short_edge_domain|v2g_mobile|2|-9.968|-0.2229|-6.339|0|-6.339|

## 真实参数覆盖

|组别/历史|域|结构覆盖|真实R/X入域|域外边数|解释|
|---|---|---|---|---:|---|
|pilot_domain/988301|asset|True|True|0|structure_and_RX_covered|
|pilot_domain/988301|numeric|True|True|0|structure_and_RX_covered|
|pilot_domain/988302|asset|True|True|0|structure_and_RX_covered|
|pilot_domain/988302|numeric|True|True|0|structure_and_RX_covered|
|pilot_domain/988303|asset|True|True|0|structure_and_RX_covered|
|pilot_domain/988303|numeric|True|True|0|structure_and_RX_covered|
|pilot_domain/988304|asset|True|True|0|structure_and_RX_covered|
|pilot_domain/988304|numeric|True|True|0|structure_and_RX_covered|
|short_edge_domain/988501|asset|True|False|1|outside_RX_domain|
|short_edge_domain/988501|numeric|True|True|0|structure_and_RX_covered|
|short_edge_domain/988502|asset|True|False|1|outside_RX_domain|
|short_edge_domain/988502|numeric|True|True|0|structure_and_RX_covered|

2m内部支路落在asset声明长度族之外。结构仍在候选表中，生成该结构的真实阻抗已在允许域外；对应M用于观察受限家族的损失排序，参数误差标为域外距离。

每域每方法的失败、回退、未闭合优化和数值未决计数保存在per_domain_method.csv；逐边R/X误差保存在line_errors.csv。缺失成本保持为空，九方法分母完整。

对相同曲线和计量，精确最小成本满足J_numeric≤J_asset。nested_cost_audit.csv保留局部取得成本的反向现象，用于定位初值或数值问题；M作为两个成本之差本身没有单调性。所有结果为配对描述，未新增统计拒绝规则。
