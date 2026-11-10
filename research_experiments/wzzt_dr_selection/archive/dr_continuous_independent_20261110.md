# 旧DR原型归档，2026-11-10

当前可独立运行的核心和说明迁至[dr_pricing](../dr_pricing/README.md)。原`dr_continuous_independent`以及两个已被替代的paper15说明PDF在逐文件哈希核验后从活动树清理，原目录保留跳转说明。

清理例外：`paper15_dr_pricing_forecast_day.pdf`及旧`main.pdf`被Windows中的打开程序占用，已核验归档并从版本控制移出，本机文件暂留，未关闭用户应用。旧main对应的Git0a0603f1 LaTeX源也已归档。其余旧目录及缓存已清理；当前说明以新包MODEL/REVIEW为准，精确状态见清单的`cleanup`字段。

- 源码、报告及公开汇总可由Git提交`0a0603f1`恢复，详细四阶段原批另见`78eb5b17`。
- 私有完整ZIP为`C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_dr_portable_v1/archives/dr_continuous_independent_20261110.zip`。
- [逐文件清单与SHA-256](dr_continuous_independent_20261110.json)记录ZIP指纹、原路径、大小和每个文件指纹；解压前后可逐条复核。
- 最近一次历史调度及结果保留在`dr_pricing/results/`，明确属于权重2的已完成批次，不转写为新配置成绩。

私人自然负荷、实际P/Q/V、AC观测、所有失败/未决及原源码快照没有删除。其他独立研究支线本轮未判定为无用，未批量清除。

恢复单个历史源文件示例

```text
git show 0a0603f1:research_experiments/wzzt_dr_selection/dr_continuous_independent/full_price.py
```

不把旧目录重新加回新核心依赖。paper15已导出为不含pickle依赖的私有`wzzt_dr_portable_v1/paper15_ready_v2`，只读回放通过，权重等设置待用户决定。
