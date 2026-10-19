# 当期源拟合的多初值诊断

这是事后计算诊断，未调整冻结策略、原始记录或拒绝阈值。按36个完整校准历史中 `design_v2g_mobile` 的实际M从小到大排序（并列按seed），取最差两例991302、991305。输入来源已逐个与冻结配置、源码清单和策略身份核对，36/36一致。

保留每候选原训练R/X，仅对验证段及未来实际含噪声观测重新估计源，原P/Q输入、计量精度、源盒、180次评估预算和容差保持不变。每组8初值为原保存源加7个标准差0.01pu的扰动，裁剪到原源盒。

| 历史 | 原M | 8初值M | 差值 |
|---|---:|---:|---:|
| 991302 | −372.2581119332 | −372.2581119332 | −1.24e−11 |
| 991305 | −106.1744280729 | −106.1744280728 | +6.59e−12 |

16/16组源拟合、128/128次重启成功，32个原/新状态通过独立AC回放。最大单组SSE改善2.50e−11，最大重启间SSE跨度2.56e−11；负M没有改善。新增源拟合688个AC批次/3440快照，独立回放32/160，诊断34.13秒；后置只读来源核验28.28秒、零新增拟合。

保存的最大SciPy `optimality` 为0.00250，部分按xtol/ftol终止。因此结论是多初值的取得值稳定，这两例应优先检查训练参数与预测失配；不提供源子问题全局最优证明。

脚本在原执行位置 `<仓库>/outputs/` 下寻找仓库。以下先复制保存的原脚本，再对恢复校准目录重跑；输出用新路径，已有JSON不覆盖：

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
Copy-Item research_experiments/mobile_ev_mle/evidence/ev_dynamic/posthoc/dynamic_source_restart_check.py outputs/dynamic_source_restart_check.py
python outputs/dynamic_source_restart_check.py --directory <恢复的calibration目录> --output outputs/new_source_restart_check.json
```

来源核查和计数名称是在16组拟合完成后补充，原执行脚本的base64及SHA保留在JSON；后置 `--audit-existing` 未重新拟合。保存脚本及JSON原字节没有修改。
