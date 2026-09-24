# theft_wzzt：RNJ-wzzᵀ 主线 + 偷电（未计量负荷）识别

独立可运行包。`theft_wzzt/{models,estimation,graph,scenario,data,pipeline.py}`
是从 `../rnj_wzzt_core` 逐文件拷贝（仅改写 import 前缀）的 RX75-RNJ +
有限候选域 L1-MILP 主线；`theft_wzzt/theft/` 是新增的偷电识别模块
（仿真注入、联合 MILP、可信度评价、对照基线）。

## 环境

~~~powershell
conda activate Topo
python -m pip install -e .
~~~

## 复现主线锚点（M1 验收）

~~~powershell
python -m experiments.run_mainline --cases paper15
~~~

## 偷电实验入口（M5/M6）

~~~powershell
python -m experiments.run_theft null --replicates 20   # 零分布校准（无偷电全流程重跑）
python -m experiments.run_theft exp2                   # 单组实验：exp1..exp6
python -m experiments.run_theft report                 # 汇总 outputs/theft/summary.json
~~~

两种树模式（`--tree`）：`identified`（默认，真实链路——树与逐边权界来自主线在
清洁数据上的 RX75-RNJ + wzzᵀ-MILP 拟合，缓存于 outputs/theft/identified_tree.json，
零分布复制用 replicate≥10 避免与训练数据重叠）；`true`（oracle 对照——真树 +
真值 ±25% 界）。

identified 模式首批结果（paper15，T=8 窗口，8 kW 偷电，20 复制零分布 q95≈5.5）：
exp1 正常与 exp6 阻抗漂移 +10% 均**不告警**；exp2 内部节点、exp3 末端、
exp4 位置切换、exp5 无观测支路均**告警且定位覆盖真值区域**（检测集合允许含
真值区域的祖先区域——"区域正确但未能逐户"是合法输出；兄弟/子节点冗余反映
重估自由度下的真实歧义，以集合覆盖/宽度报告而非单点对错）。

## 边界声明

- 拓扑辨识通道维持 daily_demean；偷电通道使用 raw 绝对量与受约束截距，
  否则恒定偷电会被预处理/自由截距系统性吸收。
- 偷电识别需要首端总表提供的独立功率锚点；无锚点时存在阻抗-漏计尺度
  等价类，模块应输出相容集合而非单点结论。
- 详细设计与可行性见 ../docs/ 下 2026-09-17 的四份文档。
