# 新近机器学习与结构学习方向补充

本文档补充 `random_error_precision_guide.tex` 中没有充分展开的近年机器学习、深度学习和可微结构学习方向。当前结论是：对本项目的 terminal-only hidden-node topology identification，最稳的主线仍应是物理约束统计估计；机器学习更适合作为误差修正、滤波/窗口选择、边置信度集成和主动实验设计的外层模块，而不是直接黑箱替代 reduced sensitivity。

## 1. GNN / topology-aware state estimation

近年 GNN 在电力系统中更多用于 state estimation、voltage estimation 和 topology-aware forecasting，而不是直接从 terminal-only smart-meter 数据恢复隐藏树。可借鉴点包括：

- 用图结构 inductive bias 处理 sparse measurements；
- 在 topology changes、PMU failure、bad data、non-Gaussian noise 下增强鲁棒性；
- 用 hierarchical GNN 或 graph attention 建模多层 feeder/substation 结构。

对本项目的启发：

- 不建议直接训练一个 GNN 输出边，因为 case 数量太少、拓扑监督有限，容易过拟合。
- 更合理的是训练一个 residual model：

```text
AC_drop_residual = y_AC - (R P + X Q + gamma)
```

然后把 GNN/ML 只用于修正 LinDistFlow 与 AC 潮流之间的系统误差。

## 2. Physics-informed graphical learning

Physics-informed graphical learning 的核心思想是用可微物理模型替代纯黑箱网络，在 loss 中保留 Ohm/DistFlow/AC residual。对 terminal-only 算例，可形成：

```text
loss = data_fit(V, P, Q)
     + lambda_phys * power_flow_residual
     + lambda_sym * ||R - R.T||^2
     + lambda_pos * negative_part(R, X)
     + lambda_tree * tree_metric_violation
```

这比普通 neural network 更符合本问题，因为拓扑辨识的可辨识性来自物理约束和 additive distance，而不是单纯预测精度。

## 3. Neural relational inference / differentiable latent graph

NRI 类方法把 interaction graph 作为 latent variable，同时学习 dynamics decoder。对应到本问题：

```text
encoder: time-series P/Q/V -> edge logits
latent graph: terminal-equivalent tree or candidate hidden graph
decoder: graph-conditioned LinDistFlow/AC surrogate -> V
```

关键限制：

- 必须强制 radial tree 约束，否则会学到统计相关图而不是物理树；
- 必须把 root/common mode 单独建模，否则 latent graph 会吸收公共模态；
- 需要大量 synthetic feeders 和 domain randomization，单个 case33 不足以训练。

## 4. Differentiable spanning tree / matrix-tree learning

可以把 MST 或生成树后验做成可微模块。给每条候选边一个分数：

```text
s_e = f_theta(features_e)
```

用 matrix-tree theorem 得到边际概率：

```text
p_e = w_e b_e^T L^{-1} b_e
```

再用 edge marginal entropy、held-out residual、bootstrap stability 训练或选择参数。该方向适合作为“自适应滤波/距离融合”的实现方式。

## 5. Bayesian / ensemble uncertainty

新场景中最大的痛点不是输出一棵树，而是不知道哪些边可靠。推荐：

- bootstrap over days/scenarios；
- Bayesian linear regression over R/X；
- matrix-tree posterior；
- conformal-style topology set。

输出应从：

```text
one MAP tree
```

升级为：

```text
MAP tree + edge marginal + high-confidence edge set + ambiguous local region
```

## 6. Foundation/surrogate model 方向

大规模 feeder 数据集，例如 SMART-DS，可用于预训练：

- feeder voltage surrogate；
- residual correction model；
- profile/noise augmentation model；
- topology-aware embedding model。

但对当前科研代码，优先级不应高于统计可解释方法。更务实的路线是：

1. 生成大量 terminal-only synthetic feeders；
2. 对每个 feeder 用 AC 潮流生成多场景数据；
3. 训练 ML 模型预测每条候选边的 correction score；
4. 最终仍由 constrained MST / matrix-tree posterior 保证树结构。

## 7. 推荐优先级

当前项目最值得实现的 ML/新方法顺序：

1. 无监督 bootstrap + matrix-tree edge posterior；
2. 自适应滤波窗口选择器；
3. R/X 多视角距离集成器；
4. AC residual correction model，先用 ridge/random forest/gradient boosting，再考虑 GNN；
5. differentiable spanning tree scorer；
6. synthetic feeder pretraining；
7. full neural relational inference。

