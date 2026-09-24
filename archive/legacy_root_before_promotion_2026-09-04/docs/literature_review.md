# Smart Meter 与微型 PMU 拓扑辨识文献梳理

## 1. Zhang 等 IEEE TSG 2020

Zhang、Wang、Weng、Zhang 2020 研究了只使用 Smart Meter 的配电网拓扑和线路参数辨识。其核心输入是多时刻的有功、无功和电压幅值，即 `p,q,|V|`。方法分为两步：

1. 在忽略相角的条件下，利用近似矩阵关系 `p/v = G v`、`q/v = -B v` 估计导纳矩阵的 `G,B`；
2. 根据 `gamma(i,j)=|G_ij|/|G_ii|` 等相对贡献筛选候选边，再在候选边上使用 Newton 型方法联合估计线路参数和未量测相角。

论文报告 IEEE 33 节点系统上 `g/b` MAPE 可达到较低水平。由于作者仓库和原始用户负荷轨迹当前无法直接复用，本项目只复现算法路径和合成数据上的基线表现，不声明精确数值复现。

参考：IEEE DOI https://doi.org/10.1109/TSG.2020.2979368

## 2. 相关方法脉络

| 文献 | 量测条件 | 主要思想 | 与本项目关系 |
| --- | --- | --- | --- |
| Deka, Backhaus, Chertkov 2016 | 末端电压或统计量 | 从电压统计和径向结构恢复拓扑 | 提供无候选拓扑库结构学习基础 |
| Park, Deka, Chertkov 2018/2020 | 电压相关统计 | 基于电压协方差 / 距离的径向拓扑恢复 | 与 latent tree / Green matrix 方法相关 |
| Exact Topology and Parameter Estimation 2017 | 线性模型量测 | 同时估计拓扑和参数 | 支持“结构 + 参数联合估计”方向 |
| Srinivas & Wu 2022 | Smart Meter + 微型 PMU | UKF、NR 和 PMU 放置评估 | 与微型 PMU 辨识和信息设计相关 |
| Biswas et al. 2020 | PMU / Smart Meter | 传感器布点和拓扑可观性 | 与 PMU placement 相关 |
| Arghandeh et al. 2015 | PMU / SCADA | 配网拓扑检测和数据驱动方法 | 工程背景参考 |
| Bariya et al. 2020 | 扰动和在线数据 | 拓扑辨识、鲁棒性和 OpenDSS 实验 | 可作为后续仿真平台参考 |
| Deka, Kekatos, Cavraro 等 | 开关/量测混合 | 基于优化或 MIP 的拓扑恢复 | 与候选边稀疏恢复相关 |

## 3. 本项目与现有工作的差异

当前项目从“有限候选拓扑辨识”出发，但逐步扩展到三个方向：

1. **不确定性建模**：隐藏负荷、线路参数误差和量测噪声不再被硬设为零，而是通过 Bayesian marginal likelihood 或 conformal calibration 进入辨识过程；
2. **信息获取**：微型 PMU 布点和主动 probing 不只用于提高量测精度，而是直接针对拓扑可分性或预测集合大小进行设计；
3. **弱拓扑先验**：从依赖完整候选拓扑库，推进到候选边稀疏恢复和 topology-free latent tree，明确只恢复末端量测可辨识的约化结构。

## 4. 现有创新点的文献定位

### Hidden-load Bayesian likelihood

它不是单纯换分类器，而是把隐藏节点注入作为 nuisance variable 边缘化。与传统 WLS 残差相比，它更接近真实配网缺量测场景。需要补充负荷先验学习、协方差稳健估计和分布漂移测试。

### Active probing + PMU

它可定位为“拓扑辨识的主动实验设计”。区别于普通 PMU placement 的地方是：设计对象包括可控注入扰动，目标是增加候选拓扑之间的信息距离。后续需要加入 AC 安全约束和 DER 可执行性。

### HMM

它使用开关操作图作为时序先验，适合缺失异步量测下的动态拓扑追踪。与静态拓扑分类不同，它可以评估切换检测延迟和 switch F1。

### Potential sparse recovery

它更接近候选边上的 sparse topology estimation。当前标量 Laplacian 是简化模型，论文级版本应升级到 AC 复导纳、严格径向约束和参数重估。

### Conformal topology set

它把输出从 Top-1 改为有限样本覆盖意义下的候选集合。该方向的关键不是“准确率更高”，而是“在可交换条件下少排除真实拓扑”。自适应 PMU 查询需要端到端重新校准。

### Topology-free latent tree

它与 Deka / Park 等无候选拓扑库方法最接近，但当前项目强调只用末端注入和电势估计 Green 矩阵，再由加性距离自动生成潜在节点。论文中必须明确 degree-2 隐藏节点不可辨识，只恢复约化树和 terminal splits。

## 5. 后续文献补充清单

后续应重点补充以下方向：

1. 仅末端量测的径向拓扑恢复和 additive tree learning；
2. 配网 topology and line parameter joint estimation；
3. 微型 PMU placement、Fisher information、Bayesian experimental design；
4. 主动配网 probing 和安全 DER 调制；
5. conformal prediction 在时序、分布漂移和结构化输出上的扩展；
6. OpenDSS / IEEE 123 / 8500-node feeder 上的 benchmark 工作。

## 6. 推荐写作边界

不建议把“无需候选拓扑”泛泛作为唯一创新点，因为相关文献已经存在。更稳妥的表述是：

**在仅末端量测和隐藏注入不确定性下，先恢复可辨识约化树，再进行 AC 复导纳精化，并通过可校准集合输出和主动信息获取降低拓扑不确定性。**
