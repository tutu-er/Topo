# 创新实验说明

本目录包含若干面向配电网拓扑辨识的研究原型。它们的定位是“可运行的证据样机”：用于检验机制是否有信号，而不是直接给出最终论文结论。

## 一键运行

```powershell
$env:PYTHONPATH='src'
python scripts/run_all_innovations.py
```

单项运行：

```powershell
python scripts/innovation_1_bayesian_hidden_loads.py
python scripts/innovation_2_active_probing.py
python scripts/innovation_3_dynamic_hmm.py
python scripts/innovation_4_potential_sparse.py
python scripts/innovation_5_conformal_adaptive.py
python scripts/innovation_5_dro_placement.py
python scripts/innovation_6_topology_free_latent_tree.py
```

主要汇总结果写入 `results/innovation_suite.json`。DRO 和 topology-free latent tree 的单项结果分别写入 `results/innovation_5_dro.json` 和 `results/innovation_6_topology_free.json`。

## 当前创新点概览

1. **隐藏负荷边缘化 Bayesian likelihood**  
   不再把隐藏节点强行设为零注入，而是对隐藏负荷扰动进行 Monte Carlo 边缘化。当前小网络中准确率从 zero-injection WLS 的 46.25% 提升到 97.92%。可行，但需要更真实的隐藏负荷先验和 AC 协方差建模。

2. **主动 probing + 微型 PMU 联合设计**  
   在有限拓扑库内联合选择安全注入扰动和 PMU 节点，最大化最差拓扑对分离度。当前 joint active PMU 准确率为 99.87%。工程上有潜力，但必须加入 DER 容量、电压约束、用户扰动成本和安全约束。

3. **开关图约束 HMM**  
   将拓扑切换看成稀疏的 branch-exchange 状态转移，用 Viterbi 平滑缺失异步量测。当前 35% 缺失率下准确率从 69% 提升到 97%。可行性较强，但依赖有限拓扑状态库和合理转移模型。

4. **potential-edge 稀疏恢复**  
   在候选边集合上用标量 Laplacian surrogate、group sparsity 和树投影恢复拓扑。当前可恢复 11/11 真实边，但参数 MAPE 约 35.53%。它是一个有用的凸核心原型，还需要升级到 AC 复导纳和严格径向约束。

5. **class-conditional conformal 拓扑集合 + 自适应 PMU 查询**  
   将拓扑辨识输出从 Top-1 改成集合，当前 5% 目标误覆盖下经验覆盖率为 97.22%，平均集合大小 2.07。方向可行，但当前自适应 PMU 查询后的 Top-1 还没有端到端 conformal 校准。

6. **DRO-CVaR PMU 布点**  
   尝试在线路参数扰动集合上优化 worst-tail 分离度。当前实验中 robust 方案在 shifted test 上反而低于 nominal 方案，说明原始指标和真实识别损失不一致。暂不建议作为主贡献，除非重做鲁棒目标。

7. **不依赖候选拓扑库的 latent tree 恢复**  
   只用 slack、末端标签和终端注入/电势估计 Green 矩阵，再用加性距离恢复潜在树。当前 split precision / recall 均为 1.0，且不使用候选拓扑、候选边或隐藏节点数。它是最值得发展成主线的创新点之一，但需明确只恢复可辨识约化树。

## 推荐路线

优先把创新点组织成一条连贯主线，而不是堆叠彼此独立的小实验：

1. 用 topology-free latent tree 给出无候选库的结构初始化和可辨识性边界；
2. 在初始化附近做 AC 复导纳精化和隐藏注入边缘化；
3. 用 conformal 输出结构集合或编辑半径，而不是只报 Top-1；
4. 用主动 probing / PMU 设计减少集合大小或结构风险；
5. 对 HMM 作为时变拓扑扩展，放在动态场景章节。

详细可行性分析见 `docs/transactions_innovations.md`；不依赖候选拓扑库的理论和方法边界见 `docs/topology_free_methods.md`。
