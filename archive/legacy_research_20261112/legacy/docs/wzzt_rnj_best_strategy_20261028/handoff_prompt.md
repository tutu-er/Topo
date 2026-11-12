# 可直接复制给其他对话

请先读取本地策略交接说明：

`D:\0-github_workspace\Topo\docs\wzzt_rnj_best_strategy_20261028\README.md`

并读取同目录 `best_strategy.json`。工作区是 `D:\0-github_workspace\Topo`，Python 是 `D:\apps\miniconda3\envs\Topo\python.exe`。

当前在既有 80 组算例上平均拓扑 F1 最高的已测 RNJ+WZZT 版本是 **`multistart_original_rule`**，F1=0.9252397562。原完整 RNJ 组合对照是 `baseline25`，F1=0.9194371947。增强版相对基线 5 项提升、72 项持平、3 项退化；退化均为 high_voltage_noise/48 点，网络为 paper15、flynn16、pengwah18。

准确策略是：两预处理分支×NJ/RNJ×三几何×五阈值形成有限候选池；raw 的 15 个完整 RNJ 家族联合 L1 拟合；主初值 raw/RX75/0.16 修正 25 秒；从经过收缩、联合重估并按验证 MAE 排序的原 RNJ 家族中，再取最多两个不同初值各修正 25 秒；最终合并原模型库、原路径和新增路径，保持原选模规则，再作统一的最终输出收缩。

关键参数为 raw、r/x 界 [0,2]、绝对残差 L1、交叉支持可替换、增益门槛 1e-10+0.001×当前损失。最终选模用最终输出收缩前的独立验证 MAE：最低值 3% 范围内，按支持数、MAE、列表索引选最小；之后以固定 raw RX75 校准的 0.16 分辨率收缩，保留叶，权重不再重估。为挑额外初值所做的收缩后重估应保留；不要把它与另一个全候选收缩后重估对照混淆。

请使用精确版本名。`multistart75` 平均 F1=0.9032532051，与最佳版本不同。单初值 25 秒增至 75 秒没有改变 80 项最终 F1；全候选收缩后评分/重估与本轮 AC 门控均不并入最佳策略。此有限池方法采用固定族 LP，`mip_rel_gap` 不适用。

已有数据可直接从以下记录库读取：

`D:\0-github_workspace\Topo\docs\wzzt_rnj_budget_20261028\catalog.sqlite`

接口为 `research_experiments.wzzt_budget.records.find/load/load_trace`；指定 case、condition、N 和 variant=`multistart_original_rule`。共有 80 组输入、560 条版本记录，支持读取 R/X、簇、r/x、原 P/Q/V/root、选模来源、失败/停止状态及完整搜索轨迹。catalog 中实际最终模型的 `validation_mae` 与选模使用的收缩前 MAE 分别解释。真簇诊断字段仅作事后分析标签。

最佳版本组装代码是 `research_experiments/wzzt_budget/variants.py::multistart_original_rule`；额外初值和轨迹在 `worker.py::run` 生成。最佳版本属于已保存轨迹的分析派生结果，原始 result.json 的 models 字典只有五个初始对照；优先从 catalog 或 selected_models.json 读取。当前尚无只运行最佳策略、直接接收任意新观测的独立推断包装器。

请先按我在本对话提出的任务使用这些资料。优先复用已有结果；重跑实验需使用新目录并保持相同观测、划分和计分口径。保留原基线、三项退化和未决状态。现有结论针对四个仓库构建网络、五种条件及 24/48/96/288 点；新随机重复和未知网络的验证另行处理。
