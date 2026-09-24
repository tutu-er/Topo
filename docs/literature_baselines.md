# 文献对照算法：实现范围、数学模型与公平性

本项目将“完整算法”定义为：从论文规定的量测量开始，包含矩阵估计、距离构造、隐藏树生成、候选选择和结果诊断；不能获得的私有训练参数必须显式替换并记录。所有比较数据均由逐时刻径向 AC 潮流生成，再加入 `P/Q=0.5%`、`V=0.02%` 的相对标准差噪声。不存在无噪声结果混入汇总。

## 1. 已实现方法

### 1.1 Soumalas 2017：整数线长 + Prüfer 场景搜索

论文从末端智能表估计公共路径灵敏度，再构造叶节点间距离

\[
d_{ij}=R_{ii}+R_{jj}-2R_{ij}.
\]

其关键额外信息不是普通 NJ 所需的信息，而是候选架空线/服务线类型，以及线路长度为最小长度整数倍的假设。实现 `soumalas_prufer_reconstruction` 对每个最小阻抗跨度 \(\ell\) 执行：

1. \(D^{(\ell)}=\operatorname{round}(D/\ell)\)；
2. 恢复加性加权树，并把整数权边展开为单位边；
3. 生成并重新解码 Prüfer 序列，验证标签树；
4. 以终端距离残差筛选场景；同残差时选择隐藏 degree-2 节点更少的树。

该实现完整保留论文的整数跨度和多场景验证思想，但用标准加性树求解器替代了论文流程图中的专用逐叶提取过程，因此登记为 **paper-assumption-faithful**，不是逐行原代码复刻。

### 1.2 Park/Deka 2020：minimal-observability recursive grouping

从终端量测估计加性阻抗距离后，对活动节点 \(i,j\) 计算

\[
\Phi_{ijk}=d(i,k)-d(j,k).
\]

若 \(\Phi_{ijk}\) 对所有其他活动节点 \(k\) 为常数，则 \(i,j\) 为兄弟或父子；兄弟关系会引入新的隐藏父节点。实现为 `recursive_grouping`。在精确加性距离、隐藏节点零注入且隐藏节点度数至少为 3 时，可恢复最小隐藏树及边阻抗。degree-2 隐藏链只能作为合并阻抗恢复，不能唯一恢复其节点个数。

### 1.3 Pengwah 2022：受约束灵敏度 + 回溯 RG

电流灵敏度模型为

\[
\Delta V=S_r\Re(\Delta I)+S_x\Im(\Delta I).
\]

实现施加：对称性、线路阻抗非负、对角线公共路径上界、树距离条件负定外近似。对每对活动节点计算

\[
T_{\rm noise}=\max_k\Phi_{ijk}-\min_k\Phi_{ijk},\qquad
T_{\rm par}=\left|d_{ij}-|\operatorname{mean}_k\Phi_{ijk}|\right|.
\]

当父子和兄弟代价比位于 \([1/\kappa,\kappa]\) 时同时保留两条回溯分支。候选树按论文的四类目标排序：灵敏度方程误差、重构电压误差、电压相对次序误差和电压相关矩阵误差。

论文的 logistic 权重由 11,243 个未公开合成拓扑训练，无法原样获得。本项目对候选池内四项目标分别归一化并默认等权；接口允许传入复现得到的权重。这是唯一明确的不可完全复刻项。

### 1.4 Flynn 2023：transformer common mode + 物理可行 RG

Flynn 把变压器二次侧公共波动作为未知时序：

\[
\Delta V=\mathbf 1\Delta V_{tr}+S\Delta I,
\qquad
\min_{S,\Delta V_{tr}}
\|\Delta V-\mathbf1\Delta V_{tr}-S\Delta I\|_F^2
+\alpha\|\Delta V_{tr}\|_2^2.
\]

`fit_current_sensitivity(..., transformer_mode="free_regularized")` 用交替更新和投影梯度求解。`enhanced_recursive_grouping` 进一步实施论文物理约束：

- 两个智能表客户不能直接构成父子；
- 新生成的兄弟边长度必须严格为正；
- 更新后的距离不得为负；
- 已知根节点不能再获得父节点。

这解释了 regulated-root 或 common-mode 版本明显优于固定根假设的原因：它使用“所有末端节点共享同一上游电压扰动”这一结构信息，但不要求根电压表。当前算例实际提供根电压；公平比较仍让 Flynn 估计该模态，而不是直接读取真值。

### 1.5 Pengwah 2024：部分智能表覆盖

客户分为智能表集合 \(S\) 和间隔电能表集合 \(I\)。只使用 \(V^S,P^S,Q^S,P^I,V^{tr}\)，拟合

\[
V^{tr}\mathbf1-V^S\simeq
[R^{SS}\ R^{SI}\ X^{SS}\ X^{SI}]
[I_r^S\ \bar I_r^I\ I_x^S\ 0]^\top.
\]

`fit_partial_meter_impedance` 不读取 interval 节点的 `V/Q`，并按论文令 \(I_x^I=0\)。先由 \(R^{SS}\) 和 RG 恢复智能表约化树，再利用

\[
\hat d_i=\max_{s\in S}R^{SI}_{si}
\]

确定 interval 节点与最近智能表的公共路径深度：若该深度对应已有节点则直接挂接，否则分裂路径上的一条边并插入隐藏节点。实现位于 `partial_meter.py`。

### 1.6 Choi 2011 CLGrouping 与 Ni 2011 RNJ

CLGrouping 先用最小距离树建立全局邻域，再仅在每个局部父子星形中执行 RG；初始距离树只是计算加速和局部化工具，**不是最终 terminal-equivalent MST 输出**。实现位于 `cl_grouping.py`。

RNJ 使用已知根到终端的距离：

\[
\rho(i,j)=\frac{d(0,i)+d(0,j)-d(i,j)}{2},
\]

即根到最近公共祖先的共享路径长度。它优先合并 \(\rho\) 最大的节点，并以最小边长给出的阈值处理噪声，输出已定根的最小隐藏树。

## 2. 尚不能作为同条件隐藏树对照的方法

- 电压差方差 MST / Chow-Liu 在全节点可观时可恢复物理树；仅有 terminal 量测时通常只能得到 observed-node tree，不能与 full hidden tree 的边 F1 直接比较。
- PaToPa/PaToPaEM、Split-EM 面向给定候选线路或多个离散运行拓扑的联合参数/状态识别，不自动生成未知隐藏节点。
- GridTopo-GAN 等监督或生成式方法需要大量“量测-拓扑标签”训练对。直接在本算例训练并测试会引入训练拓扑库这一额外信息，不能与零训练的 NJ/RG/RNJ 公平比较。
- 2025 subset-sum 方法只使用功率量测，适合另建候选拓扑/下游功率守恒基准；其公开预印本与当前基于阻抗距离的隐藏树任务输入并不相同。

这些方法已记录在文献表，但不伪装成同条件结果。后续若加入，报告必须单列“候选图已知”“监督训练”“额外相角/PMU”三类信息预算。

## 3. 运行接口

快速单条件检查：

```powershell
python -m experiments.run_complete_literature_baselines `
  --cases paper15 --scenario-counts 3 --T-counts 96 --replicates 1 `
  --output outputs/literature_quick
```

完整 case bank：

```powershell
python -m experiments.run_complete_literature_baselines `
  --cases paper15 soumalas11 flynn16 pengwah18 `
  --scenario-counts 1 3 5 --T-counts 48 96 288 --replicates 2 `
  --pq-noise-rel 0.005 --v-noise-rel 0.0002 `
  --output outputs/complete_literature_baselines
```

若要严格执行 Pengwah/Flynn 的穷举回溯而不使用 128 个候选的运行上限，请添加 --max-backtracking-candidates 0。该模式可能指数增长；CSV 中的 candidate_generation_truncated 会明确标记限幅运行。

输出包括 `results.csv`、`summary.csv`、`metrics.json`、`method_metadata.json` 和 `report.md`。主指标为隐藏节点标签无关的 terminal split F1；`exact_unrooted_recovery=true` 表示所有非平凡 terminal splits 完全一致。rooted clade F1 另外评价根方向。

## 4. 主要来源

- [Soumalas et al., 2017](https://doi.org/10.1109/PTC.2017.7981061)
- [Park et al., 2020](https://doi.org/10.1109/TCNS.2020.2979882)
- [Pengwah et al., 2022](https://doi.org/10.1109/JSYST.2021.3128175)
- [Flynn et al., 2023](https://doi.org/10.1109/TSG.2023.3239650)
- [Pengwah et al., 2024](https://doi.org/10.1109/TPWRD.2024.3354292)
- [Choi et al., 2011](https://www.jmlr.org/papers/v12/choi11b.html)
- [Ni and Tatikonda, 2011](https://doi.org/10.1109/TIT.2011.2168901)
- [Deka, Kekatos, and Cavraro tutorial, 2024](https://doi.org/10.1109/TSG.2023.3271902)
- [Yu, Weng, and Rajagopal, PaToPaEM](https://arxiv.org/abs/1812.06619)
- [Wu et al., GridTopo-GAN](https://doi.org/10.1109/TII.2022.3158614)
- [Xu and Chen, subset-sum preprint, 2025](https://arxiv.org/abs/2507.16924)

BibTeX 记录见 `docs/literature_references.bib`。
