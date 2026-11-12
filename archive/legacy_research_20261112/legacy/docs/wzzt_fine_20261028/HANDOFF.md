# RNJ+WZZT 当前策略与细化测试交接

工作区 `D:/0-github_workspace/Topo`；正式运行 `outputs/wzzt_fine_20261028_full`；当前说明与索引 `docs/wzzt_fine_20261028`。

- 流程、参数、接口以 [FLOW.md](FLOW.md) 为准；本轮结果以 [README.md](README.md)、`per_job.csv` 为准。
- 推荐策略为 `ridge_union`：原 raw 联合 L1 模型库和三个初值路径，追加 lambda=0.001/0.01 候选并集修正。保留原 3% 验证简洁规则，再做 0.16 最终收缩。AC 只作训练后处理和独立预测评价，不参与拓扑选择。
- 本轮新数据是四个固定合成网络、三个独立训练/验证/测试 bundle、电压噪声 0.02%/0.05%/0.1%、N=24/36/48/72/96，共 180 输入。三个物理日；N 为训练总点数，验证/测试各 288 点。噪声/N 条件配对，独立重复单位为 bundle。
- 主面板 1620 条、预定高噪声 N24/N48 子面板 144 条，共 1764 条。180/180 工作进程返回；扩池搜索一条触及路径预算，详情保留在 `ridge_wall_limit.csv`。
- 平均结构 F1：NJ 0.64158，固定 RNJ 0.65920，联合 L1 库 0.74971，单初值 0.76656，多初值 0.79943，扩池 0.80688。扩池相对多初值/同预算追加原池控制均 6 胜、174 平、0 负；完整簇恢复 93→95/180。
- 单 lambda=0.01 与并集的逐输入 F1 一致；本轮未单独测其部署总耗时。保留并集作为已连续使用的推荐策略，精简版作为后续成本对照。
- 扩池剩余 85 项未完整恢复，其中中噪声 27、高噪声 58。`unidentified.csv` 保留所有方法的未恢复项；`failure_modes.csv` 及 `diagnostic_json` 区分池漏真簇和最终交叉错误簇，不能据最终结构直接推断早期阻挡因果。
- 子面板严格验证保护删簇 `delete_guard0` 单轮得到 7 胜、17 平、0 负；容差 1%/3% 有退化；两步宽度 3/5 均未改善。这些组件未与扩池联合验证，禁止累加增益。
- 仅 NJ、多初值、扩池三方法预定 AC 评价，共 540 行，去重 370 个输入/参数组合；局部求解器收敛、物理可行和结构正确分别解释。其他方法 AC 未评价。

## 精确调用

```python
from research_experiments.wzzt_fine.records import find, load, load_trace
catalog = r"D:\0-github_workspace\Topo\docs\wzzt_fine_20261028\catalog.sqlite"
rows = find(catalog, case="flynn16", voltage_sigma=0.001, N=48)
model = load(catalog, case="flynn16", voltage_sigma=0.001, bundle=1,
             N=48, variant="ridge_union", with_observations=True)
trace = load_trace(catalog, case="flynn16", voltage_sigma=0.001, bundle=1, N=48)
```

不要使用上一轮缺少 bundle 维度的记录读取器加载本轮目录。新观测估计用 `research_experiments.wzzt_fine.pipeline.infer`，不接收真拓扑。其默认同时生成全部主对照，运行总耗时不等于部署单方法耗时。

## 恢复与核验

运行目录 `source/` 是本轮计算冻结代码；`data/` 保存实测/无噪数据和网络；`jobs/` 保存输入、checkpoint、最终模型库、搜索轨迹及 LP 原生证据；`ac/` 保存训练重估与冻结测试评价。

文档目录 `producer_source.zip` 绑定运行时源，`analysis_source.zip` 绑定汇总/审计/读取代码。`manifest.json`、`result_manifest.json`、`analysis_source_manifest.json`、`delivery_manifest.json` 保留字节哈希。`audit_summary.json` 为数值与机制检查，`delivery_verification.json` 为最终记录/交付核验。重建命令见 README。

旧 80 项与本轮输入面板、随机实现、样本量组成不同，不直接比较两轮总体 F1。当前只有三个独立随机 bundle 和四个固定网络，不把本轮零退化写为全局保证。
