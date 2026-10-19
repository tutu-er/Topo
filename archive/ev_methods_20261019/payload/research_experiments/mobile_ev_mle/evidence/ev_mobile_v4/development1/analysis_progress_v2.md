# 四域嵌套规划与动作排序审计

独立统计单位是一条已冻结历史；同历史的 regular/shift 未来为配对观测。规划只计一次，缺失历史保留12方法及每方法4候选槽。Phi 是共同完整AC预测的平均最近半SSE差；平方基线的主规划目标是 Q，不要求其 Phi 随域单调。

## development

历史：1/4 个文件；域边单调已知 12/48，违反 0。

|场景|方法|历史/有效|M均值|M中位|规划问题|物理失败/缺失|拟合失败/缺失|raw正确/完整集|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|n3/regular|square__fixed_charge|1/0|—|—|0|0/0|1/0|0/0|
|n3/regular|square__fixed_bidir|1/0|—|—|0|0/0|1/0|0/0|
|n3/regular|square__mobile_charge|1/0|—|—|0|0/0|1/0|0/0|
|n3/regular|square__mobile_bidir|1/0|—|—|0|0/0|1/0|0/0|
|n3/regular|aligned__fixed_charge|1/0|—|—|0|0/0|1/0|0/0|
|n3/regular|aligned__fixed_bidir|1/0|—|—|0|0/0|1/0|0/0|
|n3/regular|aligned__mobile_charge|1/0|—|—|0|0/0|1/0|0/0|
|n3/regular|aligned__mobile_bidir|1/0|—|—|1|0/0|1/0|0/0|
|n3/regular|guarded__fixed_charge|1/0|—|—|0|0/0|1/0|0/0|
|n3/regular|guarded__fixed_bidir|1/0|—|—|0|0/0|1/0|0/0|
|n3/regular|guarded__mobile_charge|1/0|—|—|0|0/0|1/0|0/0|
|n3/regular|guarded__mobile_bidir|1/0|—|—|1|0/0|1/0|0/0|
|n3/shift|square__fixed_charge|1/0|—|—|0|0/1|0/1|0/0|
|n3/shift|square__fixed_bidir|1/0|—|—|0|0/1|0/1|0/0|
|n3/shift|square__mobile_charge|1/0|—|—|0|0/1|0/1|0/0|
|n3/shift|square__mobile_bidir|1/0|—|—|0|0/1|0/1|0/0|
|n3/shift|aligned__fixed_charge|1/0|—|—|0|0/1|0/1|0/0|
|n3/shift|aligned__fixed_bidir|1/0|—|—|0|0/1|0/1|0/0|
|n3/shift|aligned__mobile_charge|1/0|—|—|0|0/1|0/1|0/0|
|n3/shift|aligned__mobile_bidir|1/0|—|—|0|0/1|0/1|0/0|
|n3/shift|guarded__fixed_charge|1/0|—|—|0|0/1|0/1|0/0|
|n3/shift|guarded__fixed_bidir|1/0|—|—|0|0/1|0/1|0/0|
|n3/shift|guarded__mobile_charge|1/0|—|—|0|0/1|0/1|0/0|
|n3/shift|guarded__mobile_bidir|1/0|—|—|0|0/1|0/1|0/0|
|n4/regular|square__fixed_charge|1/0|—|—|0|0/1|0/1|0/0|
|n4/regular|square__fixed_bidir|1/0|—|—|0|0/1|0/1|0/0|
|n4/regular|square__mobile_charge|1/0|—|—|0|0/1|0/1|0/0|
|n4/regular|square__mobile_bidir|1/0|—|—|0|0/1|0/1|0/0|
|n4/regular|aligned__fixed_charge|1/0|—|—|0|0/1|0/1|0/0|
|n4/regular|aligned__fixed_bidir|1/0|—|—|0|0/1|0/1|0/0|
|n4/regular|aligned__mobile_charge|1/0|—|—|0|0/1|0/1|0/0|
|n4/regular|aligned__mobile_bidir|1/0|—|—|0|0/1|0/1|0/0|
|n4/regular|guarded__fixed_charge|1/0|—|—|0|0/1|0/1|0/0|
|n4/regular|guarded__fixed_bidir|1/0|—|—|0|0/1|0/1|0/0|
|n4/regular|guarded__mobile_charge|1/0|—|—|0|0/1|0/1|0/0|
|n4/regular|guarded__mobile_bidir|1/0|—|—|0|0/1|0/1|0/0|
|n4/shift|square__fixed_charge|1/0|—|—|0|0/1|0/1|0/0|
|n4/shift|square__fixed_bidir|1/0|—|—|0|0/1|0/1|0/0|
|n4/shift|square__mobile_charge|1/0|—|—|0|0/1|0/1|0/0|
|n4/shift|square__mobile_bidir|1/0|—|—|0|0/1|0/1|0/0|
|n4/shift|aligned__fixed_charge|1/0|—|—|0|0/1|0/1|0/0|
|n4/shift|aligned__fixed_bidir|1/0|—|—|0|0/1|0/1|0/0|
|n4/shift|aligned__mobile_charge|1/0|—|—|0|0/1|0/1|0/0|
|n4/shift|aligned__mobile_bidir|1/0|—|—|0|0/1|0/1|0/0|
|n4/shift|guarded__fixed_charge|1/0|—|—|0|0/1|0/1|0/0|
|n4/shift|guarded__fixed_bidir|1/0|—|—|0|0/1|0/1|0/0|
|n4/shift|guarded__mobile_charge|1/0|—|—|0|0/1|0/1|0/0|
|n4/shift|guarded__mobile_bidir|1/0|—|—|0|0/1|0/1|0/0|

|场景|家族最高Phi父比较|选中父/预期|平均指标牺牲真ref|真ref预测升而实际降|平均预测升而实际降|guard违反|
|---|---|---:|---:|---:|---:|---:|
|n3/regular|square|3/3|0|0|0|0|
|n3/regular|aligned|3/3|1|0|0|0|
|n3/regular|guarded|3/3|0|0|0|0|
|n3/shift|square|0/3|0|0|0|0|
|n3/shift|aligned|0/3|0|0|0|0|
|n3/shift|guarded|0/3|0|0|0|0|
|n4/regular|square|0/3|0|0|0|0|
|n4/regular|aligned|0/3|0|0|0|0|
|n4/regular|guarded|0/3|0|0|0|0|
|n4/shift|square|0/3|0|0|0|0|
|n4/shift|aligned|0/3|0|0|0|0|
|n4/shift|guarded|0/3|0|0|0|0|

QP 239/264 达到间隙要求；原参数域可核查 264，线路越界 0、源越界 0；master 0/6 达到代理间隙。

动作对一致率、并列和缺失分母、逐域及同域平方对照、完整求解界、压缩证书、被拒提案与逐历史误差分解保存于 JSON。它们是描述性诊断，未进行新的误拒率校准。
