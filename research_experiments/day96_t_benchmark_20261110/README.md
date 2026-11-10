# 一日96点末端方法对照

`run.py` 冻结优化源码和数据，`analyze.py` 在训练后独立核验144个计划配置；`candidates.py`导出历史拟合后冻结的完整响应模型供后续DR接入。

## 后分析的种子依赖修正

物理数据审计确认：生成器在四个网络之间复用相同的物理曲线种子。48个观测输入有16个网络×种子历史，但只有4个共享随机种子组，三档噪声也采用配对随机数。

因此，后分析bootstrap以4个 `bundle` 共享种子组为抽样单位，每次整体携带四个网络和三档噪声；不再把16个网络×种子历史当成独立抽样。只报告这一种主分析，四个抽样单位下的区间作为描述性证据。

本项是冻结后的后分析代码修正，不改变冻结优化器、输入、方法、预算或原始结果。实际执行的 `analyze.py` 及哈希应另存 `analysis_source` 与核验回执；不要覆盖冻结的 `outputs/.../source`。

## 复现

使用新的运行目录，冻结入口拒绝覆盖已有协议。`pilot`为四网络首种子中噪声的12配置；`all`自动保留这些结果，再运行其余配置。

```powershell
D:/apps/miniconda3/envs/Topo/python.exe -B research_experiments/day96_t_benchmark_20261110/run.py freeze outputs/day96_t_benchmark_REPLICATION --workers 8
D:/apps/miniconda3/envs/Topo/python.exe -B research_experiments/day96_t_benchmark_20261110/run.py pilot outputs/day96_t_benchmark_REPLICATION --workers 8
D:/apps/miniconda3/envs/Topo/python.exe -B research_experiments/day96_t_benchmark_20261110/run.py all outputs/day96_t_benchmark_REPLICATION --workers 8
D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.day96_t_benchmark_20261110.analyze outputs/day96_t_benchmark_REPLICATION --workers 2
```

每配置14400秒总预算，单步MILP3600秒、固定族LP300秒，gap=.0001、单线程，最多8配置并行。达到精度即提前结束。失败和未完成保留144计划分母。单次方法成本含公共前端一次记账，与批次等待时间分别报告。

`data.py`生成训练连续日96、独立验证288、另一独立连续日96的观测。优化器只读取训练/验证；真值仅供后分析。

`candidates.py`保留自身原始路径、E输出与单点保护模型的完整层级/R/X、来源、节点顺序、单位、数据/代码哈希。`.16`收缩为独立拓扑映射，不改变预测参数。完整RNJ整树备用不参与导出。

`verify_bridge.py`用合成数据调用隔壁实际冻结预测与3%选择函数，核验预测零差、选择一致、哈希和终端顺序保护，不启动对方报价、AC或参数拟合。

## 最终后分析与归档顺序

`finish.py`是一次性等待器：原生144条全部返回或记录失败后，运行独立`analyze.py`，只有144条均核验通过才绘制三组图。它的完成状态仅覆盖数值核验和绘图；最终归档还需要完整候选桥接、选模规则对照、恢复耗时上界核验与人工视觉检查。

原生批次完成并通过核验后，依次运行：

```powershell
D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.day96_t_benchmark_20261110.bridge_candidates outputs/day96_t_benchmark_20261110
D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.day96_t_benchmark_20261110.compare_rules outputs/day96_t_benchmark_20261110
D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.day96_t_benchmark_20261110.timing_uncertainty outputs/day96_t_benchmark_20261110
```

确认PNG图形后，记录`docs/day96_t_benchmark_20261110/visual_review.json`，再运行`deliver.py`。归档入口会核对144条数值结果、144份桥接导出、对照与时间回执，以及图形引用的是同一版原生核验结果。重新生成原生核验回执后，须同步更新依赖它的桥接、对照、时间与图形回执。

`deliver.py`将原生分析表复制到文档目录，将选模对照和时间核验表分别加`compatibility_`和`timing_`前缀复制，并全部纳入ZIP与哈希清单。时间区分成功尝试算法耗时、含基础设施中断代理扣账的操作耗时，以及宿主确认的保守上界；未知的中断终止时刻不写成精确值。保守预算未决或超限不会被改成通过，相关分母与归档回执字段继续保留。
