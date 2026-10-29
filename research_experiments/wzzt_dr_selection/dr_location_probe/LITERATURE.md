# WZZT逐候选决策与需求响应设计：定向文献核对

核对日期：2026-10-29。复用既有LITERATURE_CLOSURE.md，并读取以下开放全文的相关章节。没有重新下载论文或使用订阅认证。此表为定向最近邻，不是穷尽性检索。

| 论文、来源及已读位置 | 已有内容 | 对本研究的限制 |
|---|---|---|
| Dette & Titoff, 2009, Optimal discrimination designs; DOI 10.1214/08-AOS635；[电子重印本](https://arxiv.org/pdf/0908.1912)，§2式(2.4)，§3及§4相关讨论 | 输入设计最大化竞争模型重新优化参数后仍不能解释的差异；讨论未知参数依赖以及模型判别与参数估计的差别 | 共同重调权是模型判别设计已有基础，改用L1本身不构成贡献。其固定参考模型与一般WZZT早期多遗漏候选不同。 |
| Hainy, Price, Restif & Drovandi, 2022, Optimal Bayesian design for model discrimination via classification; DOI 10.1007/s11222-022-10078-2；[正文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8924111/)，损失函数§式(3)/(5)、交叉验证式(6)/(7)及独立评价段落 | 直接优化决策器的模型误选损失；讨论模拟拟合、验证与计算成本 | 将WZZT放进期望结构错误函数属于该思想的具体化，不能仅凭目标写法主张方法新颖。需要证明WZZT结构带来的实际计算或恢复收益。 |
| Bhela, Kekatos & Veeramachaneni, 2019, Smart Inverter Grid Probing for Learning Loads: Part II; DOI 10.1109/TPWRS.2019.2906306；[作者稿](https://engineering.purdue.edu/~kekatos/papers/TPWRS2018b.pdf)，§II-B/C式(8)–(12) | 已知网络下筛除不可行注入，并优化系统状态差异以改善未计量负荷估计 | 可辨识性分析加激励设计已有；没有WZZT结构更新或价格介导的随机居民响应。 |
| Brégère & Bessa, 2020, Simulating Tariff Impact in Electrical Energy Consumption Profiles with Conditional Variational Autoencoders; DOI 10.1109/ACCESS.2020.3009060；[开放稿](https://arxiv.org/pdf/2006.07115)，§5.3、§7.3–8 | 价格、天气和日历条件下生成随机整日负荷；保留提前响应、反弹、时段关联，并报告训练外价格曲线的外推问题 | 当前整日经验响应是简化实现，尚无新的行为模型贡献。 |

另在上一轮已核对Cavraro、Bernstein与Dall'Anese的2026预印本[Demand Response Under Stochastic, Price-Dependent User Behavior](https://arxiv.org/html/2603.15983v1)，§II–III：响应分布依赖节点价格，均值线性化仅用于小调整，研究目标为运行控制。这也限制“概率价格响应”作为独立创新的表述。

## 本轮保留的数学对象

以代码实际重拟合、剪零后的当前块族B为起点。令B_C删除所有与候选C交叉的当前块。在全部保留块和新增块的连续权重上共同最小化平均L1损失，得到J_B、J_(B_C)、J_(B_C+C)。恒等式为：

`J_B - J_(B_C+C) = [J_(B_C) - J_(B_C+C)] - [J_(B_C) - J_B]`。

这是加入收益减去交叉删除代价。候选还必须在实际扫描/并列规则下获胜，并超过`1e-10 + 0.001*J_B`的接受门槛。该量可核对实际更新；单独最大化它可能强化错误候选。

价格改变P/Q特征，也改变未知网络下的未来Y。完整历史解释、连续权重、零/弱效应和非唯一最优参数仍需保留。仅有少数拟合模型时，可计算的是模型条件下的规划结果；没有完整覆盖或校准模型概率。外层全H最坏结构错误公式尚未成为可执行方法。

一个只在被选决策上正确的局部设计，还必须经过后续替换、验证选模和最终收缩。当前程序的最终结构错误才是效果评价，候选损失、接受边际、池变化及输入波动只用于解释。

## 能被实验否定的贡献要求

1. 在同样候选和连续权重域内，利用WZZT更新结构，以更低代价准确评估接受、拒绝或并列，而非靠删去困难解释节省计算。
2. 这种评估所选择的价格，在相同预算与可行动作下，改善独立实际响应后的最终恢复，并超过无奖励、随机位置、普通输入设计和必要的模型判别基线。

两个要求均未因公式成立或文献有差异而自动满足。本批固定P15全部位置检查只回答动作空间是否有收益；它不是新价格算法验收。
