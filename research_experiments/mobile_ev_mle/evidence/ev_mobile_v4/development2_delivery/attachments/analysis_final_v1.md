# 四域嵌套规划与动作排序审计

独立统计单位是一条已冻结历史；同历史的 regular/shift 未来为配对观测。规划只计一次，缺失历史保留12方法及每方法4候选槽。Phi 是共同完整AC预测的平均最近半SSE差；平方基线的主规划目标是 Q，不要求其 Phi 随域单调。

## development

历史：4/4 个文件；域边单调已知 48/48，违反 0。

|场景|方法|历史/有效|M均值|M中位|规划问题|物理失败/缺失|拟合失败/缺失|raw正确/完整集|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|n3/regular|square__fixed_charge|1/1|30.824|30.824|0|0/0|0/0|1/1|
|n3/regular|square__fixed_bidir|1/1|117.039|117.039|0|0/0|0/0|1/1|
|n3/regular|square__mobile_charge|1/1|38.350|38.350|0|0/0|0/0|1/1|
|n3/regular|square__mobile_bidir|1/1|195.760|195.760|0|0/0|0/0|1/1|
|n3/regular|aligned__fixed_charge|1/1|53.460|53.460|0|0/0|0/0|1/1|
|n3/regular|aligned__fixed_bidir|1/1|156.446|156.446|0|0/0|0/0|1/1|
|n3/regular|aligned__mobile_charge|1/1|53.460|53.460|0|0/0|0/0|1/1|
|n3/regular|aligned__mobile_bidir|1/1|195.760|195.760|0|0/0|0/0|1/1|
|n3/regular|guarded__fixed_charge|1/1|30.824|30.824|0|0/0|0/0|1/1|
|n3/regular|guarded__fixed_bidir|1/1|117.039|117.039|0|0/0|0/0|1/1|
|n3/regular|guarded__mobile_charge|1/1|38.350|38.350|0|0/0|0/0|1/1|
|n3/regular|guarded__mobile_bidir|1/1|117.039|117.039|0|0/0|0/0|1/1|
|n3/shift|square__fixed_charge|1/1|121.184|121.184|0|0/0|0/0|1/1|
|n3/shift|square__fixed_bidir|1/1|215.070|215.070|0|0/0|0/0|1/1|
|n3/shift|square__mobile_charge|1/1|96.497|96.497|0|0/0|0/0|1/1|
|n3/shift|square__mobile_bidir|1/1|181.997|181.997|0|0/0|0/0|1/1|
|n3/shift|aligned__fixed_charge|1/1|121.184|121.184|0|0/0|0/0|1/1|
|n3/shift|aligned__fixed_bidir|1/1|174.179|174.179|0|0/0|0/0|1/1|
|n3/shift|aligned__mobile_charge|1/1|96.497|96.497|0|0/0|0/0|1/1|
|n3/shift|aligned__mobile_bidir|1/1|192.323|192.323|0|0/0|0/0|1/1|
|n3/shift|guarded__fixed_charge|1/1|121.184|121.184|0|0/0|0/0|1/1|
|n3/shift|guarded__fixed_bidir|1/1|215.070|215.070|0|0/0|0/0|1/1|
|n3/shift|guarded__mobile_charge|1/1|96.497|96.497|0|0/0|0/0|1/1|
|n3/shift|guarded__mobile_bidir|1/1|215.070|215.070|0|0/0|0/0|1/1|
|n4/regular|square__fixed_charge|1/1|3.256|3.256|0|0/0|0/0|1/1|
|n4/regular|square__fixed_bidir|1/1|4.074|4.074|0|0/0|0/0|1/1|
|n4/regular|square__mobile_charge|1/1|3.285|3.285|0|0/0|0/0|1/1|
|n4/regular|square__mobile_bidir|1/1|4.074|4.074|0|0/0|0/0|1/1|
|n4/regular|aligned__fixed_charge|1/1|3.531|3.531|0|0/0|0/0|1/1|
|n4/regular|aligned__fixed_bidir|1/1|4.073|4.073|0|0/0|0/0|1/1|
|n4/regular|aligned__mobile_charge|1/1|10.869|10.869|0|0/0|0/0|1/1|
|n4/regular|aligned__mobile_bidir|1/1|7.932|7.932|0|0/0|0/0|1/1|
|n4/regular|guarded__fixed_charge|1/1|3.256|3.256|0|0/0|0/0|1/1|
|n4/regular|guarded__fixed_bidir|1/1|4.074|4.074|0|0/0|0/0|1/1|
|n4/regular|guarded__mobile_charge|1/1|3.285|3.285|0|0/0|0/0|1/1|
|n4/regular|guarded__mobile_bidir|1/1|4.074|4.074|0|0/0|0/0|1/1|
|n4/shift|square__fixed_charge|1/1|-2.952|-2.952|0|0/0|0/0|0/1|
|n4/shift|square__fixed_bidir|1/1|18.988|18.988|0|0/0|0/0|1/1|
|n4/shift|square__mobile_charge|1/1|-3.148|-3.148|0|0/0|0/0|0/1|
|n4/shift|square__mobile_bidir|1/1|18.988|18.988|0|0/0|0/0|1/1|
|n4/shift|aligned__fixed_charge|1/1|11.897|11.897|0|0/0|0/0|1/1|
|n4/shift|aligned__fixed_bidir|1/1|22.909|22.909|0|0/0|0/0|1/1|
|n4/shift|aligned__mobile_charge|1/1|1.019|1.019|0|0/0|0/0|1/1|
|n4/shift|aligned__mobile_bidir|1/1|22.909|22.909|0|0/0|0/0|1/1|
|n4/shift|guarded__fixed_charge|1/1|-2.952|-2.952|0|0/0|0/0|0/1|
|n4/shift|guarded__fixed_bidir|1/1|16.744|16.744|0|0/0|0/0|1/1|
|n4/shift|guarded__mobile_charge|1/1|1.019|1.019|0|0/0|0/0|1/1|
|n4/shift|guarded__mobile_bidir|1/1|16.744|16.744|0|0/0|0/0|1/1|

|场景|家族最高Phi父比较|选中父/预期|平均指标牺牲真ref|真ref预测升而实际降|平均预测升而实际降|guard违反|
|---|---|---:|---:|---:|---:|---:|
|n3/regular|square|3/3|0|0|0|0|
|n3/regular|aligned|3/3|1|0|0|0|
|n3/regular|guarded|3/3|0|0|0|0|
|n3/shift|square|3/3|0|1|1|0|
|n3/shift|aligned|3/3|1|1|1|0|
|n3/shift|guarded|3/3|0|1|1|0|
|n4/regular|square|3/3|0|0|0|0|
|n4/regular|aligned|3/3|0|0|0|0|
|n4/regular|guarded|3/3|0|0|0|0|
|n4/shift|square|3/3|0|1|1|0|
|n4/shift|aligned|3/3|0|1|1|0|
|n4/shift|guarded|3/3|0|0|0|0|

QP 1373/1488 达到间隙要求；原参数域可核查 1488，线路越界 0、源越界 0；master 0/32 达到代理间隙。

动作对一致率、并列和缺失分母、逐域及同域平方对照、完整求解界、压缩证书、被拒提案与逐历史误差分解保存于 JSON。它们是描述性诊断，未进行新的误拒率校准。

