# 最小输出guard（隔离原型，未推广）

比较已经拟合的raw点与原contract输出，按相同验证L1和3%复杂度规则选择，不再强制contract。生产默认不变，不拟合、不LP、不AC。完整结果见REPORT.md。

2项确定性输出检查通过；15条已见开发记录（7个不同输入）0结构改善、2退化、13不变，完整恢复不变。拒绝作为当前结构改进或DR方法推广。当前自动路线已结束，DR主目标未达。

入口：

```powershell
# 使用D:/apps/miniconda3/envs/Topo/python.exe，PYTHONPATH包含仓库根及rnj_wzzt_core。
python -B -m research_experiments.wzzt_dr_selection.dr_output_guard.study --root <私有输出根> --prepare-only
python -B -m unittest research_experiments.wzzt_dr_selection.dr_output_guard.test_guard -v
python -B -m research_experiments.wzzt_dr_selection.dr_output_guard.study --root <同一根>
```

上述固定批次已完成，不重跑。protocol锁定源/输入hash；partial/failed记录保留，不隐式重试；已完成summary可只读复用。测试及study机制回执是同2例，不重复计独立分母。

私有根为visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_dr_output_guard_v1；公开evidence/dr_output_guard_v1保存源码、协议、结果和独立审计。原始损失复现需要受控私有既有观测，家庭和聚合轨迹不上传。原子支持、零/弱权与当前1e−9有效支持口径均保留。
