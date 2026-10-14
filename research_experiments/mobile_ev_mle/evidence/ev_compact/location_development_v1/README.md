# 两主方法的固定功率曲线位置消融：开发接口

仅4条开发历史989101--989104，两主方法各4个配对，共8个方法条件。正式独立12H尚未读取或执行；不套用原主策略的校准阈值。

运行冻结清单包含父72源及新增/复用3份代码或说明，共75份；`source_snapshot.json.gz`保存字节。4条父slim输入及其原raw哈希与独立父AC/操作审查一致。

## 已执行与复算

8/8配对完整。5条实际改接曲线新增40次候选拟合（每曲线×noisy/noiseless×4候选），fit内AC1114次，物理forward5次，合计1119次。3条未移动曲线直接复用原拟合，ΔM=0。

`ev_compact_location_review.py`复用第16阶段独立branch-current AC和SSE算术，已复算144个保存拟合状态（16份H状态及128份联合状态，包含缓存复制）、16个物理状态及全部40个新fit。5个新增forward均覆盖。最大反算白噪声差约5.03e-12；未调用新的拟合或选择器。

|历史|方法|原−地点0实际ΔM|无噪ΔM|取消驾驶后多出电量kWh|
|---|---|---:|---:|---:|
|989101|refined|2.953|4.151|0.2|
|989101|cap|15.981|14.871|0.2|
|989102|refined|0|0|0|
|989102|cap|34.359|57.212|0.2|
|989103|refined|23.603|16.116|0.2|
|989103|cap|110.792|108.191|0.2|
|989104|refined|0|0|0|
|989104|cap|0|0|0|

这些结果验证开发接口，并描述既有动作的电气位置作用。取消驾驶后的搬回曲线不再满足原终端服务任务，不能把它称为同服务比较。原动作和全部4候选的失败处理及正式分母已在协议中固定。

## 拟合初值与局部优化范围

原位置复用冻结主研究结果；搬回地点0重新拟合。两者使用同一候选历史拟合的R/X及历史P/Q、源状态作为warm start；未来状态由对应未来计量初始化，随位置数据而变化。两者参数盒相同，`multistart=2`、`max_nfev=250`、`tol=1e-10`、`linear_solver=exact`；noisy种子均为父policy seed+802+候选序号，无噪为+1202+候选序号。第二起点用同seed对R/X作相同规则的乘法扰动和盒内裁剪。

因此随机起点规则相同，未来状态初值并非相同数组。128个联合状态均报告success；最大报告optimality约0.00770，新40状态最大约0.00677。它们按ftol/xtol等软件停止条件得到局部取得值；本对照不声称全局MLE，也不将ΔM全部解释为已证明的全局因果增益。

直接物理响应另存`physical_response_differences.csv`及review中的完整差数组。5条实际搬移的电压最大差约0.00234--0.00279 pu；根表P最大差约0.000695--0.00958 kW，Q约0.000701--0.00307 kvar。EV网侧总功率不变，根表功率和损耗由AC自然改变。这些直接物理变化与局部拟合分数分别报告。

## 正式12H入口与审查门

在完整独立评价及其AC/操作审查完成后，运行：

```powershell
python -m research_experiments.mobile_ev_mle.ev_compact_location_audit `
  --input <完整12H评价raw或slim目录> `
  --output research_experiments/mobile_ev_mle/evidence/ev_compact/location_evaluation_v1 `
  --parent-review <同12H的ev_compact_review完整AC/操作报告> --execute

python -m research_experiments.mobile_ev_mle.ev_compact_location_review `
  --directory research_experiments/mobile_ev_mle/evidence/ev_compact/location_evaluation_v1 `
  --output research_experiments/mobile_ev_mle/evidence/ev_compact/location_evaluation_v1/review_full.json
```

输入必须是phase=evaluation的完整预声明12H，种子须等于父config.evaluation_seeds，4候选及两主方法保留；父72源须与冻结字节一致。父审查须是完整AC/操作报告，不能用仅QP检查报告替代；执行器核对success、policy_id、12个seed及全部原压缩raw哈希。原review中的audited_histories/available_independent_histories均应12，missing_histories为空，source_audit成功且当前源匹配。

正式统计单位仍是12条H，每方法各12个配对，24个方法条件不作为24条独立历史。位置过程未单独校准，只报告实际/无噪裕量、通道、R/X、物理响应、资源差及失败。
