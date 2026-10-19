# 四域嵌套规划与动作排序审计

独立统计单位是一条已冻结历史；同历史的 regular/shift 未来为配对观测。规划只计一次，缺失历史保留12方法及每方法4候选槽。Phi 是共同完整AC预测的平均最近半SSE差；平方基线的主规划目标是 Q，不要求其 Phi 随域单调。

## evaluation

历史：8/8 个文件；域边单调已知 96/96，违反 0。

|场景|方法|历史/有效|M均值|M中位|规划问题|物理失败/缺失|拟合失败/缺失|raw正确/完整集|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|n3/regular|square__fixed_charge|4/4|48.559|53.238|0|0/0|0/0|3/4|
|n3/regular|square__fixed_bidir|4/4|116.552|141.632|0|0/0|0/0|3/4|
|n3/regular|square__mobile_charge|4/4|62.972|79.913|0|0/0|0/0|3/4|
|n3/regular|square__mobile_bidir|4/4|127.694|153.868|0|0/0|0/0|4/4|
|n3/regular|aligned__fixed_charge|4/4|54.863|63.194|0|0/0|0/0|3/4|
|n3/regular|aligned__fixed_bidir|4/4|113.318|141.548|0|0/0|0/0|3/4|
|n3/regular|aligned__mobile_charge|4/4|68.794|87.933|0|0/0|0/0|4/4|
|n3/regular|aligned__mobile_bidir|4/4|148.652|191.343|0|0/0|0/0|4/4|
|n3/regular|guarded__fixed_charge|4/4|53.537|63.194|0|0/0|0/0|3/4|
|n3/regular|guarded__fixed_bidir|4/4|117.075|142.678|0|0/0|0/0|3/4|
|n3/regular|guarded__mobile_charge|4/4|67.270|88.510|0|0/0|0/0|3/4|
|n3/regular|guarded__mobile_bidir|4/4|117.075|142.678|0|0/0|0/0|3/4|
|n3/shift|square__fixed_charge|4/4|47.793|60.152|0|0/0|0/0|3/4|
|n3/shift|square__fixed_bidir|4/4|114.273|139.762|0|0/0|0/0|3/4|
|n3/shift|square__mobile_charge|4/4|60.766|77.936|0|0/0|0/0|3/4|
|n3/shift|square__mobile_bidir|4/4|121.480|149.082|0|0/0|0/0|4/4|
|n3/shift|aligned__fixed_charge|4/4|54.964|70.405|0|0/0|0/0|3/4|
|n3/shift|aligned__fixed_bidir|4/4|111.961|135.576|0|0/0|0/0|3/4|
|n3/shift|aligned__mobile_charge|4/4|65.195|80.856|0|0/0|0/0|4/4|
|n3/shift|aligned__mobile_bidir|4/4|142.847|182.702|0|0/0|0/0|4/4|
|n3/shift|guarded__fixed_charge|4/4|52.920|70.405|0|0/0|0/0|3/4|
|n3/shift|guarded__fixed_bidir|4/4|114.801|140.818|0|0/0|0/0|3/4|
|n3/shift|guarded__mobile_charge|4/4|64.133|83.073|0|0/0|0/0|3/4|
|n3/shift|guarded__mobile_bidir|4/4|114.801|140.818|0|0/0|0/0|3/4|
|n4/regular|square__fixed_charge|4/4|15.214|4.977|0|0/0|0/0|3/4|
|n4/regular|square__fixed_bidir|4/4|29.665|14.602|0|0/0|0/0|3/4|
|n4/regular|square__mobile_charge|4/4|12.776|3.985|0|0/0|0/0|3/4|
|n4/regular|square__mobile_bidir|4/4|29.665|14.602|0|0/0|0/0|3/4|
|n4/regular|aligned__fixed_charge|4/4|14.997|5.243|0|0/0|0/0|3/4|
|n4/regular|aligned__fixed_bidir|4/4|27.940|11.140|0|0/0|0/0|3/4|
|n4/regular|aligned__mobile_charge|4/4|19.033|8.084|0|0/0|0/0|3/4|
|n4/regular|aligned__mobile_bidir|4/4|39.700|13.719|0|0/0|0/0|3/4|
|n4/regular|guarded__fixed_charge|4/4|14.864|4.977|0|0/0|0/0|3/4|
|n4/regular|guarded__fixed_bidir|4/4|29.665|14.602|0|0/0|0/0|3/4|
|n4/regular|guarded__mobile_charge|4/4|19.127|6.682|0|0/0|0/0|4/4|
|n4/regular|guarded__mobile_bidir|4/4|31.854|17.181|0|0/0|0/0|4/4|
|n4/shift|square__fixed_charge|4/4|14.547|5.308|0|0/0|0/0|3/4|
|n4/shift|square__fixed_bidir|4/4|28.061|14.606|0|0/0|0/0|3/4|
|n4/shift|square__mobile_charge|4/4|12.389|4.745|0|0/0|0/0|3/4|
|n4/shift|square__mobile_bidir|4/4|28.061|14.606|0|0/0|0/0|3/4|
|n4/shift|aligned__fixed_charge|4/4|14.450|6.084|0|0/0|0/0|3/4|
|n4/shift|aligned__fixed_bidir|4/4|26.769|11.995|0|0/0|0/0|3/4|
|n4/shift|aligned__mobile_charge|4/4|18.758|9.052|0|0/0|0/0|3/4|
|n4/shift|aligned__mobile_bidir|4/4|39.125|14.905|0|0/0|0/0|3/4|
|n4/shift|guarded__fixed_charge|4/4|14.062|5.308|0|0/0|0/0|3/4|
|n4/shift|guarded__fixed_bidir|4/4|28.061|14.606|0|0/0|0/0|3/4|
|n4/shift|guarded__mobile_charge|4/4|19.209|7.511|0|0/0|0/0|4/4|
|n4/shift|guarded__mobile_bidir|4/4|30.610|17.516|0|0/0|0/0|4/4|

|场景|家族最高Phi父比较|选中父/预期|平均指标牺牲真ref|真ref预测升而实际降|平均预测升而实际降|guard违反|
|---|---|---:|---:|---:|---:|---:|
|n3/regular|square|12/12|0|2|1|0|
|n3/regular|aligned|12/12|0|2|2|0|
|n3/regular|guarded|12/12|0|1|1|0|
|n3/shift|square|12/12|0|2|1|0|
|n3/shift|aligned|12/12|0|2|2|0|
|n3/shift|guarded|12/12|0|1|1|0|
|n4/regular|square|12/12|0|2|2|0|
|n4/regular|aligned|12/12|1|0|1|0|
|n4/regular|guarded|12/12|0|1|1|0|
|n4/shift|square|12/12|0|2|2|0|
|n4/shift|aligned|12/12|1|0|1|0|
|n4/shift|guarded|12/12|0|1|1|0|

QP 2677/2856 达到间隙要求；原参数域可核查 2856，线路越界 0、源越界 0；master 0/64 达到代理间隙。

动作对一致率、并列和缺失分母、逐域及同域平方对照、完整求解界、压缩证书、被拒提案与逐历史误差分解保存于 JSON。它们是描述性诊断，未进行新的误拒率校准。

