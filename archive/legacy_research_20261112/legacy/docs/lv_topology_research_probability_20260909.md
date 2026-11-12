# 低压配电网拓扑辨识的研究价值与概率化路径

最值得延续的主线是：在末端量测和隐藏节点条件下，识别数据能够支持的结构，保留不能排除的结构歧义，并根据运行任务决定是否需要补充信息。现有 RNJ、层状树原子、wzzT 优化、根信息模型和候选树实验可以形成这条主线的基础。概率化的关键是明确随机性来自何处、结论对什么对象成立，以及搜索遗漏如何影响结论。

本文以截至 2026 年 9 月 9 日能够核实的公开文献，以及当前 Topo 仓库的代码、理论文档和归档实验为依据。公开来源包括原始论文、作者机构页面和预印本；只能获取摘要或元数据的条目在文末注明。研究建议属于基于这些证据的判断，不等同于已经证明的文献空白。研究对象优先取当前项目的已知根参考、末端 P/Q/V、隐藏零注入节点、最简有根径向树；三相、隐藏注入和现场设备图另外讨论。

**现有研究已经形成有价值的基础，但贡献需要重新聚焦。** 当前独立核心依次完成多场景约束回归、R/X 共享路径几何、RNJ 候选生成、bootstrap、层状结构拟合及验证选择。有序约束的回归是凸 QP，有限域 wzzT 是另一层结构优化；两者的最优性和统计含义不能混用。[当前核心说明](D:/0-github_workspace/Topo/rnj_wzzt_core/README.md)

最新归档实验提供了比单一平均准确率更有意义的研究线索：

| 已有观察 | 证据与范围 | 对下一项研究的意义 |
|---|---|---|
| RNJ 候选加 wzzT 可以改善结构 | 36 个主条件中，整体 clade F1 从 0.8389 到 0.9158，末端 F1 从 0.5370 到 0.8151；根部均值相同 | 候选生成与结构选择的分工有效，收益主要在末端 |
| 电压验证分数未保留多数结构收益 | 再与固定 RNJ 树按验证 MAE 比较后，F1 为 0.8511，对 RNJ 为 3 胜、33 平 | 预测损失和结构损失之间存在实质差异 |
| 稳定组不一定正确 | 本批 N8 条件下，选中冻结块的真值精度为 0/8；主消融仅使用 12 次 bootstrap | 稳定度不能直接升级为硬结构事实 |
| 根误差已经有明确统计落点 | 528 个根模型结果均记录 success；已研究公共方向协方差和表计锚定 | 后续应研究完整噪声传播和校准，不能把“加公共项”当作新贡献 |
| 已经存在候选树广义后验 | 历史 GTLS 实验用留出 EIV 分数、复杂度项和温度归一化 | 下一步应解决权重的解释、覆盖和校准，而非重复添加 softmax |
| 已经存在人工查询排序原型 | 完整树集上的结构分歧下降与查询成本；现场回答后的闭环尚未执行 | 可以扩展到带回答误差和实际决策价值的信息获取 |

上述主要均值已从当前归档结果重新核算；本次没有重新运行整套算法。36 条件来自四个小型合成网络、三个重复及嵌套样本量，不能视为 36 条独立真实馈线。历史 GTLS 结果使用不同场景与算法设置，不能与新 reference 消融的数值直接排行。[研究报告与限定](D:/0-github_workspace/Topo/rnj_wzzt_core/docs/research_study_20260908.md:123)、[原始消融结果](D:/0-github_workspace/Topo/artifacts/root_information_study_20260908/ablation/metrics.csv)、[历史广义后验](D:/0-github_workspace/Topo/docs/gtls_topology_posterior.md)

**文献已经覆盖了不少看似自然的扩展。** 必须先区分用户—变压器归属、相别归属、已知线路上的开关状态、未知隐藏节点的最简树重建，以及具名物理设备图恢复。不同任务的 accuracy、线路 F1 和 clade F1 不具有直接可比性。已有系统性教程也明确区分主动/被动数据、观测配置和可辨识条件。[S1](https://arxiv.org/abs/2206.10837)

| 文献 | 已有内容 | 对当前研究的约束 |
|---|---|---|
| Zhang 等，2026 年 9 月卷期 | 末端电压幅值和 P/Q；物理约束回归、full-branch iterative grouping、非线性潮流修正 | “末端量测＋约束矩阵＋分组＋物理修正”已经有直接竞争者，应纳入对照 |
| Fang 等，2024 | 联合估计三相灵敏度与变压器电压，增强 RG、backtracking 和候选选择 | 三相化和根电压估计本身不足以构成新颖性 |
| Low-Voltage Distribution Grid Topology Identification With Latent Tree Model，2022 | 潜树概率模型、候选搜索、BIC 和 EM | 概率潜树不是空白领域 |
| Liu 等，2026，Ca-TI | 弱监督和证据融合，输出表计级置信度以辅助核查；案例以相别归属为例 | “给置信度＋指导人工核查”已有先例；其完整隐藏树覆盖能力不能由此推断 |
| ResNet 与共形预测，2025 会议论文 | 已有拓扑识别结合 conformal prediction 的题目与出版记录 | 不能宣称首次把共形预测用于拓扑；本文未获得全文，不评价其具体保证 |
| Talkington 等，2025 预印本 | 量化量测下的参数误差界，设定包括全节点量测、固定功率因数和均匀随机抖动量化 | 值得迁移传感器建模方法，不能直接照搬其界到末端隐藏树 |
| Théate 等，2025；Castin 等，2026 | 真实 DSO 数据上的相别/拓扑核验，以及利用状态估计残差空间模式定位错误 | 局部核验和现场可解释性是明确的工程应用方向 |
| Kumarawadu 等，2025 | 用智能表灵敏度和阻抗估计计算三相四线 DOE | “把识别出的 R/X 接到 DOE”本身已经有研究 |

对应原始来源：[S2](https://www.sciencedirect.com/science/article/pii/S2352467726002675)、[S3](https://research.monash.edu/en/publications/three-phase-voltage-sensitivity-estimation-and-its-application-to/)、[S4](https://ieeexplore.ieee.org/document/9696306/)、[S5](https://www.mdpi.com/1996-1073/19/6/1503)、[S6](https://doi.org/10.1109/CEEPE64987.2025.11033908)、[S7](https://arxiv.org/html/2508.05620v1)、[S8](https://orbi.uliege.be/handle/2268/327542)、[S9](https://orbi.uliege.be/handle/2268/342287)、[S10](https://research.monash.edu/en/publications/smart-meter-data-driven-dynamic-operating-envelopes-for-ders/)。

Ca-TI 作者还明确把同源方法之间的依赖、物理可解释性和置信校准列为后续问题。这与本项目“R-RNJ、X-RNJ、RX75 和 MILP 都来自同一批 P/Q/V”的情况直接相关。但从该论文的表计标签结果，不能推导未知隐藏树上的同时覆盖保证。[S5](https://www.mdpi.com/1996-1073/19/6/1503)

**优先方向一：带覆盖保证的部分拓扑辨识。** 建议把问题表述为：给定有限末端量测、根观测模式和声明的模型误差范围，哪些 clade 可以确认，哪些可以排除，哪些必须保留为未决？输出可以是一棵局部未细分的树及一组兼容模型，无须每次强制给出唯一完全细化的树。

这与当前资产最接近。层状原子提供清晰的结构空间，RNJ 产生可行候选，wzzT 可以拟合或搜索相反假设。真正需要新增的是有效的同时误差界、完整兼容域或保守外包，以及短边和搜索超时下的拒判规则。

一个可独立评审的问题是：能否构造一组随数据变化的已确认 clade，使“其中至少一条错误”的概率不超过预设水平？进一步研究置信集合的直径如何随最短内部边、激励条件、根信息和样本量变化。覆盖结论本身的集合推导很简单；难点在低样本 EIV、模型偏差和组合计算，不能把一个集合定义当作全部贡献。

该方向的失败判据也应明确：若半径有效但置信集合长期包含几乎所有树，则方法诚实却没有足够分辨力。应报告集合宽度、拒判率和所需新信息，而不是把保守覆盖率单独作为成功。

**优先方向二：结构歧义下的 DOE 或电压控制。** 问题可以进一步转为：完整拓扑尚未确定时，现有信息是否足以支持某个运行决定？哪些未确定连接最值得核查，才能增加可用容量或减少越限风险？

这比简单的识别—优化串联更有研究价值。已有数据驱动 DOE 和鲁棒 DOE 工作分别表明灵敏度模型、负荷不确定性与客户在包络内的自由动作都需要处理；新增贡献应聚焦于从识别数据产生的结构/参数不确定性及其计算传播。[S10](https://research.monash.edu/en/publications/smart-meter-data-driven-dynamic-operating-envelopes-for-ders/)、[S11](https://arxiv.org/abs/2212.03976)

可以定义模型集合在动作域 \(\mathcal U\) 上的电压分歧：

\[
\Delta_{\mathcal U}(\mathcal M)
=\sup_{m,m'\in\mathcal M,\ u\in\mathcal U}
\|f_m(u)-f_{m'}(u)\|_\infty.
\]

若这个分歧及模型误差上界都小于预留电压裕度，则部分结构未知可能不妨碍该项电压任务。若某一处拓扑分歧显著改变约束，则它才是高价值核查对象。这是本文提出的任务化评价方式，不是现有实验已经证明的能力。

必须限制迁移范围：端口电压响应相同的串联线路细分，可能具有不同载流量、设备归属或保护配置。仅靠末端 R/X 无法识别热额定值。电压约束下的可用模型，不自动适用于热约束、故障定位或保护整定。

**优先方向三：针对不可区分结构的主动信息获取。** 对每对仍兼容且影响决策的模型，比较额外根表、局部总表、独立 P/Q 扰动或现场接线核查的成本和价值。主动逆变器 probing 及其可恢复条件、MILP 拓扑处理已有研究；新增点应放在隐藏树歧义、错误回答、有限预算和下游任务，而非“首次主动辨识”。[S12](https://arxiv.org/abs/1803.04506)、[S13](https://arxiv.org/abs/2004.02370)

本项目的查询排序工具可以作为起点，但它目前计算候选树之间的经验分歧下降，并假定答案准确；并不预测现场闭环恢复收益。外部核查若推翻所有候选，应扩展模型集合并重新拟合。为发现所有候选的共同错误，仍需分配一定探索预算；集合内分歧为零不能证明该处无需检查。[当前查询设计](D:/0-github_workspace/Topo/rnj_wzzt_core/docs/prior_acquisition_design.md:119)

**三相、缺测、异步和隐藏注入也值得做，但宜选一个机制作为主问题。** 三相方法需要相别映射、中性线和互耦模型；不能将单相正权路径度量复制三份。隐藏非零注入会改变当前“末端注入解释全部压降”的模型。异步和量化可能制造固定偏差，重复采样不一定消除它们。这些都是工程上重要的问题，但把它们同时放入一个概率模型会使假设难以辨识、实验难以归因。

若希望形成独立理论问题，较好的表述是“漏测注入与拓扑错误何时观测等价，最少增加什么量测能够区分”，或“量化精度和时间错位如何决定可辨内部边的尺度”。它们比同时增加多个噪声模块更容易得到可反驳的命题。

| 路线 | 与当前基础的贴合程度 | 最需要新增的内容 | 主要风险 |
|---|---|---|---|
| 部分拓扑与覆盖保证 | 最高 | 同时置信集合、完整域/外包、拒判机制 | 有效集合可能过宽 |
| 拓扑歧义下的 DOE | 高 | 结构与参数不确定性传播、全包络保证 | 热约束还需要设备信息 |
| 主动核查与 probing | 高 | 回答/干预似然、成本、顺序有效性 | 候选外共同错误和执行条件 |
| 三相测量缺陷专项 | 中 | 新物理模型、对应可辨识性和真实数据 | 容易演变为难归因的大系统 |
| 通用树空间统计迁移 | 中 | 新领域数据生成模型与比较基线 | 表示相似不等于统计模型相同 |

**已有工作的迁移价值主要在结构表示和推断分工。** 对一棵有根树，令 \(w_e\) 表示支路 \(e\) 下游终端的指示向量。按当前平方电压约定，理想单相线性模型为

\[
R(T,r)=2\sum_{e\in E(T)}r_e w_e w_e^\top,\qquad
X(T,x)=2\sum_{e\in E(T)}x_e w_e w_e^\top.
\]

支路下游集合构成层状族；任意两个集合嵌套或不相交。对固定结构，参数拟合容易处理；跨结构搜索困难。这种“共享层次＋多个非负权重通道”的表示可以迁移到端口等效网络、层次潜变量模型、树结构协方差估计，以及具有共享层次的多任务回归。

树空间上的概率推断已有独立统计研究。例如 Yao 等建立超度量协方差与树空间的联系，并设计贝叶斯采样方法，研究对象允许二叉或多分叉树。这提供了跨领域方法参考，但不能据此把当前电压数据直接当作该论文的高斯潜树样本。[S16](https://thyao.github.io/publication/ultramat/)

关键区别是：这里的 R/X 是物理灵敏度，通常不是观测电压协方差。若 \(v=M s+\epsilon\)，则在适当独立性条件下

\[
\operatorname{Cov}(v)=M\operatorname{Cov}(s)M^\top+\operatorname{Cov}(\epsilon).
\]

负荷相关性会改变观测协方差；相似的矩阵形状不能替代生成模型。另一项可迁移资产是可信 clade 的约化与展开，但当前精确关系依赖零注入和线性无损模型；估计的伪边界量还存在共同误差，不能当作独立的新表计。

**概率应先放在观测机制，再决定是否放在拓扑上。** 静态辨识窗口内，物理拓扑可以是固定但未知的参数；频率学方法只让数据随机。贝叶斯方法则用拓扑分布表达认知不确定性。只有研究实际开关变更时，才另外考虑随时间变化的 \(T_t\) 及转移模型。

| 概率陈述 | 精确含义 | 需要的依据 |
|---|---|---|
| 表计误差方差 | 重复量测误差的分布特征 | 表计校准、误差规格或独立重复量测 |
| bootstrap 中某 clade 出现 90% | 给定当前样本，重采样和算法重复输出的稳定度 | 重采样方案；不是直接的真值概率 |
| 贝叶斯 \(p(T\mid D)\) | 指定生成模型和先验下的条件信念 | 联合似然、先验、参数积分和推断诊断 |
| 广义后验权重 | 以某个损失和学习率更新的信念/决策权重 | 损失选择、尺度、候选域和校准说明 |
| 95% 拓扑置信集合 | 重复获取数据时，集合包含固定真模型的概率至少为 95% | 有效统计构造及覆盖它的计算域 |
| 运行越限概率 | 指定动作规则、结构/参数模型和未来扰动下的事件概率 | 识别不确定性与运行不确定性的联合传播 |

R-RNJ、X-RNJ、RX75 和 MILP 共享数据，不能把其支持率作为独立似然相乘。MILP gap 衡量声明优化问题中的上下界差距，既不是后验概率，也不是置信水平。

**一个合理的概率模型可以直接沿用当前物理结构。** 令真实功率为 \(p_t^\star,q_t^\star\)，有噪观测为 \(\widetilde p_t,\widetilde q_t,\widetilde v_t\)，并记

\[
z_t=U_{\rm ref}^2\mathbf1-\widetilde v_t^{\,2},\qquad
c_t=U_{\rm ref}^2-U_0(t)^2.
\]

候选模型可以写成

\[
\begin{aligned}
z_t&=R(T,r)p_t^\star+X(T,x)q_t^\star
 c_t\mathbf1+b_s+\delta_t+\epsilon_{v,t},\\
\begin{bmatrix}\widetilde p_t\\\widetilde q_t\end{bmatrix}
&=\begin{bmatrix}p_t^\star\\q_t^\star\end{bmatrix}+\epsilon_{pq,t},\\
a_t&=c_t+\nu_{0,t}\quad\text{（根表存在时）}.
\end{aligned}
\]

\(b_s\) 是场景常量项，\(\delta_t\) 表示线性化及未建模物理误差。这里仍须对平方电压噪声的均值进行处理；幅值零均值噪声经过平方后通常不再零均值。若追求精确生成似然，可以直接在原始电压幅值和非线性潮流层建模，代价是计算明显增加。

对于时间相关的根变化，可试验 \(c_t=\phi c_{t-1}+w_t\) 或更合适的状态模型；这属于新增统计假设。不能因为它使曲线更平滑，就宣称它恢复了原本缺少量测的共同干线参数。模型偏差 \(\delta_t\) 也不宜任意设置为足够大的高斯噪声来吸收错误拓扑；需要独立估计、物理界或模型敏感性分析。

若使用减去根量测后的目标，并暂假定量测误差通道相互独立，则线性化的同刻残差协方差应包含

\[
\Sigma_{e,t}\approx
\Sigma_{v,t}
\sigma_{0,t}^2\mathbf1\mathbf1^\top
M_T\Sigma_{pq,t}M_T^\top
\Sigma_{\delta,t},
\qquad M_T=[R(T,r)\ \ X(T,x)].
\]

同一根表误差同时影响全部终端；同一个 P/Q 表计误差也通过网络影响多个输出，因而完整协方差通常不是对角矩阵。若误差通道相关，还要加入交叉协方差。时间相关性则需要块协方差、条件创新模型或有依据的分块推断。

当前根信息研究已经处理了 \(\sigma_v^2 I+\sigma_0^2\mathbf1\mathbf1^\top\) 的常方差近似；历史 GTLS 分数主要使用逐输出的传播方差。后续可以统一这两者。但误差传播方差本身还不是精确 EIV 似然：观测 P/Q 与有效残差可能相关，严格贝叶斯建模需要积分潜在真实功率，或使用另外可论证的校正方法。[根模型](D:/0-github_workspace/Topo/rnj_wzzt_core/docs/root_information_model.md:66)、[候选评分实现](D:/0-github_workspace/Topo/terminal_case33/pipeline/gtls_topology_posterior.py:274)

高斯近似下的分数应包含

\[
\ell_T=\frac12\sum_t
\left(e_t^\top\Sigma_{e,t}^{-1}e_t+\log\det\Sigma_{e,t}\right).
\]

当协方差依赖候选参数时，不能遗漏 \(\log\det\Sigma\)，否则增大方差会获得不合理优势。固定权重下的凸拟合，也不意味着交替更新协方差后的整体问题凸。将残差按场景再去均值，则相当于继续估计 nuisance intercept；其自由度和样本依赖需要在统计解释中明确。

对于有取整/量化的电压读数，已知量化区间 \([l,u)\) 时，可以用相应传感器连续分布的区间概率 \(F(u)-F(l)\) 建立似然，而不是默认每个误差都独立高斯。上述形式是建模建议；量化误差界文献中的抖动和全节点观测假设并未因此自动满足。[S7](https://arxiv.org/html/2508.05620v1)

**先验应描述真实信息，而不是替代可辨识性。** 已核查事实可以限制允许的层状结构；旧 GIS 则适合使用可修正的编辑距离惩罚，例如

\[
p_0(T)\propto
\mathbf1\{T\in\mathcal T_{\rm allowed}\}
\exp[-\lambda K(T)-\rho d(T,T_{\rm GIS})].
\]

这个先验需要在所有允许结构上归一化，且 \(\lambda,\rho\) 应做敏感性分析。对称二元 cherry、完整多元末端组、同根支路和具名实物边是不同事实，必须使用对应约束。现有 candidate_supports 只是允许候选出现，并不等于概率软先验；initial_supports 保留结构，也不保证内部原子有正权重。[先验语义与接口](D:/0-github_workspace/Topo/rnj_wzzt_core/docs/prior_acquisition_design.md:29)

对可能误查的答案 \(Y\)，应使用回答模型

\[
p(T\mid D,Y,q)\propto p(Y\mid A_q(T),q)\,p(T\mid D).
\]

例如一个结构事件原概率为 0.6，核查的灵敏度和特异度都为 0.9，收到“是”后，该事件概率为 \(0.54/(0.54+0.04)\approx0.931\)。这里的 0.9 必须有外部依据；与原始 P/Q/V 共享来源的算法判断，不能伪装成独立现场证据重复使用。

无根量测且公共项完全自由时，可行条件下的

\[
R'=R+u\mathbf1\mathbf1^\top,\quad
X'=X+v\mathbf1\mathbf1^\top,\quad
c'_t=c_t-u\mathbf1^\top p_t^\star-v\mathbf1^\top q_t^\star
\]

保持预测不变。先验可以偏好其中一组参数，但数据本身没有消除该自由度。串联二度隐藏节点的细分、零权内部边的保留/收缩也需要规范化。应尽量对可辨识的等价类、端口响应或具有声明电气分辨率的结构事件报告结论。

**最贴近现有理论的概率化方法，是把随机误差界接到确定性树几何上。** 不必先为全部树指定后验。假设能构造一个关于物理响应及 nuisance 参数的集合 \(\mathcal U_\alpha(D)\)，使

\[
\Pr_{m_\star}\{m_\star\in\mathcal U_\alpha(D)\}\ge1-\alpha.
\]

这里 \(m_\star\) 包含真结构、参数和必要的公共项/模型偏差；集合须按真实观测机制构造。将物理允许模型与它相交，再投影到规范化树：

\[
\mathcal C_\alpha(D)
=\{T:\exists\theta,\nu,\ (T,\theta,\nu)\in
\mathcal M_{\rm phys}\cap\mathcal U_\alpha(D)\}.
\]

只要物理模型类确实包含真系统，则直接得到

\[
\Pr\{T_\star\in\mathcal C_\alpha(D)\}\ge1-\alpha.
\]

这是在前提成立时的严格集合推论，而不是当前代码已经具备的覆盖定理。定义

\[
\mathcal C_{\rm certain}
=\bigcap_{T\in\mathcal C_\alpha}\mathcal C(T),\qquad
\mathcal C_{\rm possible}
=\bigcup_{T\in\mathcal C_\alpha}\mathcal C(T).
\]

置信集合非空时，可以确认交集中的 clade，排除并集外的 clade，其余保持未决。在同一个覆盖事件上，这些结构声明同时正确，因此不必把一系列未经校正的逐对检验拼接为整树结论。集合为空应触发模型/数据/先验冲突检查，不能利用空集逻辑宣布所有结论成立。

一个基准半径来自固定设计回归。若 \(Y=Z\Theta+B+E\)、\(Z^\top Z/N\succeq\kappa I\)、条件误差为合适的次高斯向量且每列 \(\|B_{\cdot j}\|_2\le\beta\sqrt N\)，则可由伪逆行范数及 union bound 得到形如

\[
\|\widehat\Theta-\Theta\|_{\max}
\le
\sigma\sqrt{\frac{2\log(2dn/\alpha)}{N\kappa}}
+\frac{\beta}{\sqrt\kappa}
\]

的同时界，其中 \(d\) 为设计列数。它展示了随机误差、激励条件和模型偏差的不同作用：增加样本只能减小第一项。这里的满秩、噪声条件和 EIV 偏差控制不能省略；普通交叉拟合本身不会消除 P/Q 测量误差偏差。

已有理论文档给出 RNJ 的条件安全区间：共享路径/根深误差不超过 \(\varepsilon\)，且最短相关骨架边为 \(\lambda_\star\) 时，需要类似

\[
\varepsilon<\lambda_\star/4,\qquad
2\varepsilon\le\tau<\lambda_\star-2\varepsilon
\]

的分离条件。因此统计界与确定性恢复命题可以组成高概率结论。但 \(\lambda_\star\) 未知时，这只是条件定理，不能作为部署时已经验证的证书；可操作办法是维护兼容集合，并只宣布其中一致的结构。[基准回归界](D:/0-github_workspace/Topo/docs/current_mainline_complete_proofs.tex:340)、[RNJ 条件命题](D:/0-github_workspace/Topo/docs/current_mainline_complete_proofs.tex:557)

还存在三个代码边界。第一，旧文档对某些投影步骤的 max-norm 结论，不自动覆盖当前带 ridge 的约束 QP。第二，低样本时设计可能不满秩，不能用正则化后的数值唯一解替代原问题的可辨识性。第三，RX75 的系数含估计距离的归一化尺度；若用 R/X 误差推导 RX75 误差，需要把这些随机尺度纳入分析，或用独立数据固定尺度。可先在未投影参数上构造有效集合，再与物理约束相交，避免把点估计投影误写成置信操作。[RX75 实现](D:/0-github_workspace/Topo/rnj_wzzt_core/rnj_wzzt/graph/sensitivity_geometry.py:52)

**计算覆盖是这一方向最容易被忽略的部分。** 从 RNJ 有限候选池中保留与数据兼容的树，得到的通常只是全部兼容树的内近似；从中找到反例能否定唯一性，但在里面找不到反例不能证明全域没有反例。用于确认结构的计算集合应完整，或者是包含全部兼容模型的保守外包。

实际可以对一个待确认 clade 搜索“数据仍可解释，但该 clade 不存在”的替代模型。找到可行替代者就保留歧义；证明全域不可行才确认；超时且没有充分界时记为未决。当前有限候选、单步扩展的 MILP 证书不等于这个全域反例搜索的证书。

若另有真树进入候选集的保证 \(1-\alpha_{\rm gen}\)，且给定进入后推断失败概率至多 \(\alpha_{\rm inf}\)，总失败概率可用二者之和控制。没有候选覆盖依据，就只能如实报告候选域内结论。在候选分布中随意添加 5% 的“其他”质量，并不能创造有依据的域外概率；域外模型还需要可计算的预测或约束。

**一个值得探索的统计—优化接口是 split likelihood ratio。** Universal inference 提供了无需经典光滑渐近条件的有限样本检验思想，并允许用最大似然的上界构造保守检验；它仍然需要成立的概率模型和相应数据分离条件。[S15](https://pmc.ncbi.nlm.nih.gov/articles/PMC7382245/)

应用到本问题，可以用独立训练数据构造任意归一化预测密度 \(q_A(D_B)\)，再对“某 clade 不存在”的复合原假设 \(H_0\) 定义

\[
E_{H_0}
=\frac{q_A(D_B)}{\sup_{m\in H_0}p_m(D_B)}.
\]

若真模型属于 \(H_0\)，分母至少为真模型似然，因此条件于训练数据有 \(\mathbb E[E_{H_0}]\le1\)。按 \(E_{H_0}\ge1/\alpha\) 拒绝即可控制一次检验错误率。这里的推导说明其适配方向；真实时间相关量测不能只通过任意切块就宣称独立。

对计算尤其有用的是界的方向：最大似然的有效上界会使比值更小，从而更保守；只在少量候选中找到的最大似然一般是下界，会使比值虚高。对应最小化负对数似然，需要全原假设域上的有效下界，而不是一条候选路径的可行解值。

当前 L1 损失只有在明确的 Laplace 观测模型及尺度下才能直接转换为似然；当前逐点 EIV 分数和温度化 softmax 也不能原样当作上述归一化联合密度。多条结构声明、反复查询和反复看数据，还需同时检验或顺序有效性设计。这条路线的优势是可以把组合优化界变为统计保守性，困难是全域建模、有效密度和检验功效。

**如果确实需要拓扑概率，应区分完整贝叶斯与广义后验。** 完整贝叶斯需要

\[
p(T\mid D)\propto p_0(T)
\int p(D\mid T,\theta,\nu)\,p(\theta,\nu\mid T)\,d\theta\,d\nu,
\]

其中连续阻抗、真实 P/Q、公共根项和噪声参数属于需要处理的未知量。用最优拟合参数代替积分，再对分数取 softmax，通常得到的是 plug-in 权重，不是这个边际后验。零内部边造成结构模型重叠，也使常规 BIC 或卡方似然比近似的适用性需要另外论证。

如果暂不承诺完整生成模型，可以定义

\[
\pi_\eta(T\mid D,\mathcal K)
\propto p_0(T)\exp[-\eta L_T(D)],\qquad T\in\mathcal K.
\]

广义贝叶斯理论为基于损失的信念更新提供了依据，但不自动给予频率学覆盖或真实拓扑概率校准。[S14](https://pmc.ncbi.nlm.nih.gov/articles/PMC5082587/)

历史代码正接近这一形式：有效样本数乘候选分数差，再除以温度归一化。可以保留它作为排序和决策权重，但要报告候选域、损失是求和还是平均、温度的选择以及独立校准结果。改变损失单位或平均维度会改变权重集中程度；固定温度 1 不是普适统计结论。[权重实现](D:/0-github_workspace/Topo/terminal_case33/pipeline/gtls_topology_posterior.py:451)

对完整候选树分布，clade 边际为 \(\pi(C)=\sum_T\pi(T)\mathbf1\{C\in\mathcal C(T)\}\)。这些边际不是独立 Bernoulli。若使用严格大于 0.5 的多数 clade，它们必相容：交叉的两个 clade 不会共存于同一树，故两者边际之和不超过 1；恰好等于 0.5 的并列需另外处理。但相容的多数树仍未必提高 F1，更不保证包含候选域外的真结构。

后验均值 \(\mathbb E[R\mid D]\) 和 \(\mathbb E[X\mid D]\) 可以用于线性预测，却未必对应单棵树。非线性潮流或安全约束下，先平均模型再优化也不等于对模型不确定性进行正确积分或最坏情况控制。

**校准应回答一个清楚的问题。** 若声称“结构正确概率为 90%”，应在具有可信真值、与使用场景相符的独立案例中检查这类预测是否约有 90% 正确，并报告误差条。需要区分整树概率、clade 边际和未来电压预测；后三者不能互相替代。

建议至少评价候选真树覆盖、结构置信集合覆盖与大小、整树/分区正确率、Brier 或对数评分、可靠性曲线、接受率—错误率关系，以及换馈线、季节和负荷相关模式后的退化。四个合成馈线上的高 F1 不足以校准 99% 级别的结构概率；模拟校准只能支持模拟分布下的结论。

标准共形预测的覆盖结论需要交换性等相应条件，时间依赖和分布迁移需另外处理。[S17](https://arxiv.org/abs/2107.07511) 对本问题的推论是：已有带拓扑真值标签、校准与测试条件相符的多案例时，可以研究结构标签集合；只有某一条未知馈线的电压时序时，对未来电压构造共形区间，并不能自动覆盖隐藏真树。普通随机拆分也无法自动建立所需条件。

另外，“错误且被确认”的概率不超过 \(\alpha\)，并不自动意味着“在所有被接受的结果中错误比例”不超过 \(\alpha\)。后者是条件/选择风险，需要独立的风险控制分析。这一点对“只核查低置信结果”的部署流程尤其重要。

**概率不能解决观测机制本身的不辨识。** 以各时刻独立、同协方差的高斯噪声为一个说明性模型，两个候选的均值分别为 \(\mu_m(t)\) 和 \(\mu_{m'}(t)\)，则联合数据的 KL 差异是

\[
\mathrm{KL}(P_m^{(N)},P_{m'}^{(N)})
=\frac12\sum_t\|\mu_m(t)-\mu_{m'}(t)\|_{\Sigma^{-1}}^2.
\]

若所有已有激励下均值相同且其他分布特征相同，数据不能区分两者；等先验二选一的最小错误率为 1/2。均值仅有很小差异时，也需要足够的有效信息量才能可靠区分。因此电压验证未选中结构更好的树，可能来自损失不对齐、有限样本、参数拟合或本身弱可辨识；不能仅凭实验把其中一种原因认定为已证明。

一个特别清楚的情形是 \(q_t=\kappa p_t\)。此时被动数据主要识别 \(R+\kappa X\)，而不保证分别识别 R 和 X。独立无功干预可能增加信息，但只采更多具有同样固定功率因数的样本未必有效。若未来动作改变 P/Q 关系，历史预测准确也未必支持外推控制。

对两个固定参数模型，在同协方差近似下，一次 P/Q 干预的分离度可写成

\[
\frac12\left\|
\Sigma^{-1/2}\big[(R_m-R_{m'})\Delta p+
(X_m-X_{m'})\Delta q\big]\right\|_2^2.
\]

这提示针对候选歧义设计激励；但实际应同时考虑参数可重拟合、公共项和安全动作集合。用两个点估计计算的分离度可能高估真实可区分程度。对于天然观测等价的结构，则必须改变量测位置或核查物理事实。

**获取信息的目标宜使用决策损失。** 对确有概率解释的模型分布 \(p\)，定义

\[
\mathcal R(p)=\min_a\mathbb E_{m\sim p}L(a,m),\qquad
\operatorname{EVSI}(q)=
\mathcal R(p)-\mathbb E_Y\mathcal R(p(\cdot\mid Y,q)).
\]

其中动作 \(a\) 可以是输出部分树、发布 DOE、安排量测或决定是否继续核查。成本应与风险/收益使用一致单位，或作为明确预算约束。熵大的问题未必影响容量，概率较低但后果严重的接线错误也可能更值得查。

当前查询工具使用两棵随机候选之间的期望结构分歧，能够描述集合内部的信息压缩；它与上述最优决策的 Bayes risk 是不同函数。若权重仍是经验频率，相关 EVSI 只能称为经验代理。若没有可信分布，可以先用兼容集合上的最坏损失或最坏后悔值。

**迁移到 DOE 时，可以给出一个无需拓扑后验的概率接口。** 假设从识别数据获得联合结构—参数置信集合 \(\mathcal M_\alpha(D)\)，覆盖真模型的概率至少为 \(1-\alpha\)；未来不可控扰动集合 \(\mathcal W_\beta(D)\) 覆盖实际扰动的概率至少为 \(1-\beta\)。模型误差、根电压和必要设备约束必须包含在声明范围内。

若发布的包络 \(\mathcal E(D)\) 满足

\[
\forall m\in\mathcal M_\alpha(D),\
\forall\xi\in\mathcal W_\beta(D),\
\forall u\in\mathcal E(D):\quad g(u,\xi;m)\le0,
\]

则由两个覆盖事件和 union bound，直接有

\[
\Pr_{D,\xi}\{\exists u\in\mathcal E(D):
g(u,\xi;m_\star)>0\}\le\alpha+\beta.
\]

这个推论不要求两个覆盖事件独立。它同时区分了识别误差、未来扰动和客户在包络内自主选择的动作：若 DOE 承诺客户可以独立行使额度，不能仅对某个假设随机动作分布取平均后称整个包络安全。

这里真正的研究工作是如何得到不太宽的联合模型集合、保守而可算的潮流约束，以及容量损失和信息成本之间的关系。若优化使用的只是线性近似，则保证也只对该近似成立，除非另外纳入 AC 误差界或可验证的内近似。已有鲁棒 DOE 方法提供运行层基线，但不替代从拓扑识别获得有效置信集合这一步。[S11](https://arxiv.org/abs/2212.03976)

**建议把下一轮研究组织成三个可独立判断成败的问题。**

1. **短内部边与根信息不足时，能够以多大置信度恢复到哪一种结构分辨率？** 从现有共享路径误差条件出发，研究完整/外包置信集合、同时 clade 声明及拒判。核心比较是覆盖—集合大小—计算成本，而非仅平均 F1。
2. **在部分辨识的网络上，哪些运行包络仍可获得声明范围内的可靠性保证？** 研究结构/参数歧义对电压与容量的影响，并比较单一 MAP 模型、无结构矩阵置信集和树结构置信集。
3. **为了增加运行价值，下一项信息应该从哪里来？** 比较被动补样、独立 P/Q 激励、根/支路表计和人工事实；纳入误查、过期先验、候选共同错误及成本。

这三项可以共享数据与模型，但各自具有独立的科学目标。第一项最容易延续当前理论，第二项的应用迁移价值最高，第三项把前两项连接到实际信息获取。通用潜树统计可以作为方法输入；三相和现场模型则适合在目标清楚后逐项扩展。

**有说服力的验证应覆盖成功边界和失败边界。** 保留当前 reference 配对协议，增加以下受控轴：内部边长度、P/Q 激励退化、根表偏差、隐藏注入、量化/异步、错误先验和候选遗漏。最好分别设计“存在可分辨替代树”和“观测等价替代树”，以检查算法是否能在应该拒判时拒判。

候选构造、权重/超参数选择、概率或阈值校准、最终评价应在数据用途上清楚分离；模拟中可以生成独立重复，现场需要对时间依赖和跨馈线分布变化作明确安排。特别应进行真树从候选池删除的压力试验，检查方法会否错误输出极高置信度。精确真树不在有限池中时，其负对数评分理论上为无穷，报告截断数值时需保留覆盖失败的事实。

基线应包括同根信息的 NJ/RNJ、同候选池的结构拟合、与当前任务匹配的 RG/backtracking，以及尽可能忠实的 FBIG。只能获得摘要的文献应先补齐方法细节，再声称复现。不同方法的量测位置、真实参数可用性、候选线路、搜索预算和训练/调参数据量须一致或分别披露。

实验无需承诺每个方向都提高精度。若置信集合较宽而运行包络仍稳定，说明完整拓扑恢复对该任务并非必要；若结构概率很集中但留出工况越限严重，则要检查模型偏差和动作外推；若补根表只消除公共方向歧义，也应把收益准确限定在该方向。它们都是能够支撑研究判断的结果。

**来源与访问范围。** 下列条目为文中实际使用的主要来源；本文未把未能获取全文的论文当作已经完成方法审计。

1. **S1 — Deka, D.; Kekatos, V.; Cavraro, G.** Learning Distribution Grid Topologies: A Tutorial. IEEE Transactions on Smart Grid 15(1), 999–1013, 2024；预印本始于 2022。[论文](https://arxiv.org/abs/2206.10837)，[出版元数据](https://research-hub.nlr.gov/en/publications/learning-distribution-grid-topologies-a-tutorial-2/)。用于任务分类与可辨识性背景。
2. **S2 — Zhang, H. 等。** A data-driven method for joint estimation of topology and impedance in distribution networks using end-user voltage magnitude and power data. Sustainable Energy, Grids and Networks 47, 102385，2026 年 9 月卷期。[出版页](https://www.sciencedirect.com/science/article/pii/S2352467726002675)。取得出版摘要和介绍预览；未取得完整算法细节，不据此评价其全部理论保证。
3. **S3 — Fang, L.; Pengwah, A. B.; Andrew, L. L. H.; Razzaghi, R.; Muñoz, M. A.** Three-phase voltage sensitivity estimation and its application to topology identification in low-voltage distribution networks. International Journal of Electrical Power & Energy Systems 158, 109949，2024 年 7 月。[作者机构页面](https://research.monash.edu/en/publications/three-phase-voltage-sensitivity-estimation-and-its-application-to/)。核实摘要、方法组成和实验范围。
4. **S4 — IEEE Transactions on Smart Grid。** Low-Voltage Distribution Grid Topology Identification With Latent Tree Model. 13(3), 2158–2169，2022；DOI 10.1109/TSG.2022.3146205。[原始页面](https://ieeexplore.ieee.org/document/9696306/)。可检索到出版摘要；用于确认潜树、BIC、EM 已有先例。
5. **S5 — Liu, S. 等。** Confidence-Aware Topology Identification in Low-Voltage Distribution Networks: A Multi-Source Fusion Method Based on Weakly Supervised Learning. Energies 19(6), 1503，2026。[原文](https://www.mdpi.com/1996-1073/19/6/1503)。取得问题定义、方法、案例和结论文本；用于区分表计置信度与隐藏整树覆盖。
6. **S6 — IEEE CEEPE 2025。** Distribution Network Topology Identification Based on Co-Training of ResNet and Conformal Prediction. DOI 10.1109/CEEPE64987.2025.11033908。[出版记录](https://doi.org/10.1109/CEEPE64987.2025.11033908)。仅核实题名和出版记录；不推断其具体覆盖保证。
7. **S7 — Talkington, S.; Rangarajan, A.; de Alcântara, P. A.; Roald, L.; Molzahn, D. K.; Fuhrmann, D. R.** Error Bounds for Radial Network Topology Learning from Quantized Measurements. 2025 年 8 月 7 日预印本。[全文](https://arxiv.org/html/2508.05620v1)。核实量化模型、全节点观测和固定功率因数边界。
8. **S8 — Théate, T. 等。** Smart meters phase identification for topology verification: Practical challenges and insights from a case study. CIRED，2025 年 6 月。[作者机构记录](https://orbi.uliege.be/handle/2268/327542)。摘要提供真实 RESA 案例和不足 20% 表计覆盖的信息；未据此推断隐藏整树恢复率。
9. **S9 — Castin, M. 等。** Detection of topological errors in distribution networks using state estimation residual patterns. CIRED Brussels Workshop，2026 年 6 月。[作者机构记录](https://orbi.uliege.be/handle/2268/342287)。用于核实真实量测下局部错误定位方向。
10. **S10 — Kumarawadu, A.; Azim, M. I.; Khorasany, M.; Razzaghi, R.; Heidari, R.** Smart meter data-driven dynamic operating envelopes for DERs. Applied Energy 384, 125469，2025 年 4 月 15 日。[作者机构页面](https://research.monash.edu/en/publications/smart-meter-data-driven-dynamic-operating-envelopes-for-ders/)。用于确认灵敏度到 DOE 的已有迁移。
11. **S11 — Liu, B.; Braslavsky, J. H.** Robust Dynamic Operating Envelopes for DER Integration in Unbalanced Distribution Networks. 2022 年预印本，后续有修订。[论文](https://arxiv.org/abs/2212.03976)。用于鲁棒包络和客户动作不确定性的背景。
12. **S12 — Cavraro, G.; Kekatos, V.** Graph Algorithms for Topology Identification using Power Grid Probing. 2018。[论文](https://arxiv.org/abs/1803.04506)。摘要区分全节点电压量测和仅 probing 节点量测下的可恢复对象。
13. **S13 — Taheri, S.; Kekatos, V.; Cavraro, G.** An MILP Approach for Distribution Grid Topology Identification using Inverter Probing. PowerTech 2019 工作，2020 年 arXiv 版本。[论文](https://arxiv.org/abs/2004.02370)。用于主动数据、固定树拟合和结构优化的已有工作。
14. **S14 — Bissiri, P. G.; Holmes, C. C.; Walker, S. G.** A general framework for updating belief distributions. Journal of the Royal Statistical Society: Series B 78(5), 1103–1130，2016。[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5082587/)。用于损失驱动广义后验的概念依据。
15. **S15 — Wasserman, L.; Ramdas, A.; Balakrishnan, S.** Universal inference. PNAS 117(29), 16880–16890，2020。[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7382245/)。用于样本分离似然比和最大似然上界的统计接口。
16. **S16 — Yao, T.-H.; Wu, Z.; Bharath, K.; Baladandayuthapani, V.** Geometry-driven Bayesian Inference for Ultrametric Covariance Matrices. 2024 年起的预印本；作者页面列有 2025 年版本，未按已正式发表期刊文献引用。[作者页面](https://thyao.github.io/publication/ultramat/)，[预印本](https://arxiv.org/abs/2401.11515)。用于树空间与结构协方差统计的迁移参考。

17. **S17 — Angelopoulos, A. N.; Bates, S.** A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. 2021 年预印本，2022 年修订。[论文](https://arxiv.org/abs/2107.07511)。用于预测覆盖与交换性、时间序列和分布迁移的适用边界。

本地依据包括：[当前核心 README](D:/0-github_workspace/Topo/rnj_wzzt_core/README.md)、[统一研究报告](D:/0-github_workspace/Topo/rnj_wzzt_core/docs/research_study_20260908.md)、[根信息模型](D:/0-github_workspace/Topo/rnj_wzzt_core/docs/root_information_model.md)、[先验与人工核查](D:/0-github_workspace/Topo/rnj_wzzt_core/docs/prior_acquisition_design.md)、[理论文档](D:/0-github_workspace/Topo/docs/current_mainline_complete_proofs.tex)、[历史 GTLS 报告](D:/0-github_workspace/Topo/docs/gtls_topology_posterior.md)。本报告新增的统计构造和研究问题均为分析建议，未记作当前生产算法已经实现或已经实证验证的功能。
