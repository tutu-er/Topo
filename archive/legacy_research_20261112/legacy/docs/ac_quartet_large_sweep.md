# AC 重排与 Quartet 四点条件大规模测试

## 1. 测试目的

本轮不引入 GTLS、机器学习或新的时延估计器，只检验以下主线：

1. 有序约束的 R/X 回归生成 RNJ 候选；
2. 每个候选树内用非负边阻抗重新拟合树可加 R/X；
3. 使用留出场景的非线性 AC 潮流电压似然和 BIC 重排候选；
4. 用 block bootstrap 四点条件和边阻抗效应收缩不可靠隐藏边。

所有终端电压均由逐时径向 AC 潮流生成。测试不包含无噪声数据。

## 2. 测试网格

共完成 72 个条件：

- 算例：paper15、soumalas11、flynn16、pengwah18；
- 数据量：sparse=3场景x96点、medium=5场景x288点、rich=9场景x960点；
- 量测噪声：
  - low：P/Q 0.25%，V 0.01%；
  - nominal：P/Q 0.5%，V 0.02%；
  - high：P/Q 1.0%，V 0.05%；
- 每个组合使用两个独立 trial；
- Quartet block bootstrap 使用 8 次重复；
- 约 20% 场景仅用于 AC 留出验证，其余场景用于 R/X 和边阻抗拟合。

噪声标准差相对于对应时刻量测绝对值定义。根电压均值为 1.02 p.u.，并保留场景内随机波动。

## 3. AC profile likelihood 修正

原固定噪声似然采用

[
rac{1}{2N}sum_trac{e_t^2}{sigma_V^2}
+rac{klog N}{N}.
]

AC 模型误差和线性阻抗初值误差通常明显大于 0.02% 电压表噪声，使第一项过大而 BIC 几乎失效，容易选择包含多余隐藏边的候选。

现在对每个候选估计有效残差方差

[
hatsigma_{mathrm{eff}}^2
=maxleft{rac{1}{N}sum_t e_t^2, arsigma_V^2ight},
]

并使用 profile Gaussian score

[
mathcal L_{mathrm{AC}}
=
rac{1}{2}left[
loghatsigma_{mathrm{eff}}^2+
rac{operatorname{MSE}}{hatsigma_{mathrm{eff}}^2}
ight]
+rac{klog N}{2N}.
]

这不是 GTLS，只是承认 AC 候选重排存在未建模误差，使复杂度惩罚与残差处于合理尺度。

## 4. Quartet 的保守使用

对候选隐藏 clade 的四点 gap 做 block bootstrap。正 gap 的显著性继续作为支持证据，但不再使用“不显著即删除”的规则，因为统计功效不足不代表隐藏边不存在。

最终仅在以下任一条件成立时删除边：

1. 拟合边阻抗效应低于正内部边中位数的 2%；
2. Quartet gap 的上置信界仍小于 -0.05，即四点条件对该 split 给出显著负证据。

这种非对称决策区分了“没有足够证据支持”和“有足够证据反对”。

## 5. 总体结果

| 方法 | 平均 F1 | 最低 F1 | 完全恢复率 |
|---|---:|---:|---:|
| fixed RNJ | 0.9547 | 0.5000 | 79.17% |
| profile AC+BIC | 0.9813 | 0.7059 | 87.50% |
| 正显著 Quartet 硬筛选，仅作诊断 | 0.9780 | 0.7059 | 83.33% |
| profile AC+BIC + 保守 Quartet/边效应 | **0.9895** | **0.7500** | **94.44%** |

与旧的固定噪声 AC 评分相比：

- AC 重排平均 F1：0.9330 -> 0.9813；
- 最终平均 F1：0.9583 -> 0.9895；
- 最终完全恢复率：65.28% -> 94.44%。

## 6. 分算例结果

| 算例 | fixed RNJ F1 | profile AC F1 | 最终 F1 | 最终完全恢复率 |
|---|---:|---:|---:|---:|
| paper15 | 0.9679 | 0.9872 | 1.0000 | 100.00% |
| soumalas11 | 0.9938 | 1.0000 | 1.0000 | 100.00% |
| flynn16 | 0.9007 | 0.9872 | 0.9907 | 94.44% |
| pengwah18 | 0.9565 | 0.9508 | 0.9671 | 83.33% |

最终方法在 72 个条件中有 68 个完全恢复。8 个条件经过 Quartet/边效应收缩后 F1 提升，没有条件因保守收缩而下降。

## 7. 剩余失败

四个非完全恢复条件为：

| 条件 | 最终 F1 | 候选集中存在真树 | FP | FN |
|---|---:|---:|---:|---:|
| flynn16 sparse/high trial0 | 0.8333 | 否 | 1 | 1 |
| pengwah18 sparse/high trial0 | 0.8000 | 否 | 2 | 1 |
| pengwah18 medium/low trial0 | 0.8571 | 是 | 1 | 1 |
| pengwah18 rich/high trial0 | 0.7500 | 是 | 3 | 1 |

因此剩余误差分为两类：

- RNJ 候选召回不足：低数据、高噪声下真拓扑未进入候选集；
- AC 排序不足：pengwah18 的短隐藏边和相近替代结构仍会得到接近的 AC 电压。

## 8. 运行和输出

运行命令：

    D:\apps\miniconda3\envs\Topo\python.exe -m experiments.run_ac_quartet_large_sweep --trials 2 --quartet-replicates 8 --output outputs\ac_quartet_profile_conservative_sweep

主要输出：

- metrics.csv：每个条件的候选召回、AC、Quartet 和最终指标；
- edge_evidence.csv：每条候选隐藏边的四点 gap、置信区间和阻抗效应；
- summary_by_method.csv、summary_by_case.csv、summary_by_regime.csv；
- mean_f1_by_case.png；
- report.md。

## 9. 限制

本轮只有两个 trial，Quartet bootstrap 只有 8 次，适合算法筛选但还不足以给出论文级置信区间。测试也未加入异步上传、15 分钟 P/Q 区间平均或电表时延。下一步若继续，应优先增加随机 trial 和 bootstrap 次数，而不是再增加新的优化算法。
