# 配电网拓扑辨识复现实验

本项目用于复现和扩展配电网拓扑辨识实验，当前包含两条主线：

1. Zhang 等 IEEE TSG 2020 方法：基于 Smart Meter 的拓扑和线路参数辨识；
2. 末端 Smart Meter + 微型 PMU：通过 max-min 拓扑分离度优化微型 PMU 布点，并评估有限候选拓扑识别准确率。

代码目标是提供可运行、可检查的研究原型，而不是声明完全复现论文的私有数据结果。已有脚本会把结果写入 `results/`。

## 快速运行

```powershell
$env:PYTHONPATH='src'
python scripts/reproduce_zhang2020.py
python scripts/reproduce_terminal_pmu.py
python -m pytest -q
```

主要输出：

- `results/zhang2020.json`
- `results/terminal_pmu.json`

## 当前复现状态

Zhang 等论文的公开材料存在两个限制：作者仓库当前不可直接获取，论文所用爱尔兰住户负荷轨迹和精确开关状态也未随论文发布。因此本项目使用 IEEE 33 节点算例和可复现的合成负荷数据进行机制复现。

当前实现包含：

- Step 1：使用 `p/v = G v`、`q/v = -B v` 的角度无关矩阵回归，筛选具有物理符号的候选边；
- Step 2：在候选边上联合估计支路 `g,b` 和未量测相角，采用带边界的 Gauss-Newton / least-squares 形式；
- 评估指标：候选边数量、真实边召回率、精度、`g/b` 参数 MAPE、相角误差和优化目标值。

当前 `results/zhang2020.json` 显示 IEEE 33 合成实验中真实边召回率为 1.0，但参数 MAPE 仍明显高于论文报告值。因此它应被视为“算法路径复现 + 可运行基线”，不是精确数值复现。

## 末端 Smart Meter + 微型 PMU

`src/topoident/terminal_pmu.py` 构造了一个 12 节点径向网络，末端 9--12 节点具有 Smart Meter。候选拓扑由一条 tie-line 闭合并打开环路上的一条非终端支路生成。

实验流程：

1. 对每个候选拓扑生成多负荷场景下的电压幅值和相角签名；
2. 仅用末端 Smart Meter 电压幅值作为基线；
3. 枚举微型 PMU 安装节点，最大化最差拓扑对的 Mahalanobis 分离度；
4. 用加噪有限假设 WLS 分类器评估识别准确率。

当前 `results/terminal_pmu.json` 中，optimized 2-PMU 方案在该小网络上达到 1.0 的识别准确率，优于 degree-based 2-PMU 和 Smart Meter only 基线。

## 代码结构

- `src/topoident/powerflow.py`：Ybus 构造、AC 潮流和图工具；
- `src/topoident/ieee33.py`：IEEE 33 节点数据和合成量测；
- `src/topoident/zhang2020.py`：Zhang 2020 的 Step 1 / Step 2 原型；
- `src/topoident/pseudo_refinement.py`：伪量测角度精化；
- `src/topoident/terminal_pmu.py`：末端 Smart Meter + 微型 PMU 布点实验；
- `src/topoident/innovation_*.py`：若干创新想法的最小可运行原型；
- `scripts/`：复现和创新实验入口；
- `docs/`：方法说明、文献梳理和创新点评估；
- `tests/`：潮流、辨识和创新原型的回归测试。

## 创新实验

创新实验说明见 `README_INNOVATIONS.md` 和 `docs/transactions_innovations.md`。

```powershell
$env:PYTHONPATH='src'
python scripts/run_all_innovations.py
```

该命令会生成 `results/innovation_suite.json`，其中汇总了 hidden-load Bayesian、active probing、HMM、potential sparse 和 conformal adaptive 等实验结果。`innovation_5_dro_placement.py` 和 `innovation_6_topology_free_latent_tree.py` 为额外单项实验。

## 注意事项

- 当前多数结果来自小规模仿真网络，适合验证机制，不足以直接支持大规模工程结论；
- 若要支撑论文投稿，需要在 IEEE 33/123、OpenDSS 或真实事件数据上补充严格消融、参数扰动、量测缺失、拓扑标签校准和运行时间评估；
- 对于 conformal 或 DRO 相关结论，必须明确校准集、测试集、策略冻结方式和扰动分布，避免把启发式准确率误写成覆盖保证或鲁棒保证。
