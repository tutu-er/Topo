# 从图论审视拓扑恢复与数据可信度：概率理论深化调研

调研日期：2026-09-23。承接《拓扑辨识、特征电流、图论与置信度》2026-09-21报告。本篇聚焦理论及其迁移条件，不改动拓扑算法，不声称完成文献实验复现。

**结论：最值得推进的是“观测不确定性 → 合法图的置信集合 → 确定结构与歧义结构”的链条。图论负责描述哪些结构与数据相容，概率理论负责校准相容集合的覆盖率。数据质量则须分别考察物理一致性、统计模型、测量冗余、激励强度和对当前结构决策的实际信息量。**

本文明确区分三种内容：已有论文结果；由已有结果得到、在文内给出条件和推导的迁移命题；尚待验证的研究设计。后者不宣称为已成立的配电网新定理。

## 1. 先定义要相信的对象

| 对象 | 可问的问题 | 合适的输出 |
|---|---|---|
| 整张图 | 是否仍存在与数据同样相容的另一张图？ | 图置信集合、候选后验及模型条件 |
| 物理边 | 此线路是否直接连接两个已知节点？ | 同时有效的边存在/不存在判定 |
| 末端分裂或簇 | 哪组终端具有共同祖先？ | split/clade置信陈述、部分解析树 |
| 图性质 | 是否存在环、最大度是否超过阈值？ | 图性质检验或区间；候选空间应允许相反性质 |
| 某条测量 | 是否违反明确的物理或统计模型？ | 异常检验结果，而非未经建模的“真实概率” |
| 整批数据 | 是否能区分关心的图？ | 观测秩、响应间隔、信息矩阵、有效独立样本数 |
| 审核决策 | 接受后可能造成多大结构损失？ | 置信集合上的最坏损失上界 |

对隐藏节点网络，优先把边表述为**终端集合的分裂/簇**，以免不同潜在节点编号造成虚假差异。若存在不可辨识二度隐藏节点，目标应是最小约化树或等价类。

若算法从一开始只允许树，输出无环是算法约束，不是数据证明了径向性。若想检验径向性，备择模型必须允许相关网状结构。

## 2. 统一观测模型与可辨识性

以
\[
Y_t=f_{T,\theta}(u_t)+b_t+\epsilon_t
\]
表示数据：T是图，theta包括阻抗、未观测注入、根节点参考和其他必要参数；u_t为已知输入或激励；b_t为需明确约束的系统偏差/模型误差；epsilon_t为随机测量误差。若P/Q或编码幅值本身有误差，应把输入误差纳入模型，而不只在输出端放一个残差。

“数据可信”至少有三个彼此独立的层面：

1. **一致性：**数据是否违反指定的KCL/KVL/噪声假设？
2. **信息性：**即使数据完全正确，能否区分所关心的拓扑？
3. **代表性：**用于推断或校准的数据是否适用于当前时段/台区/探测策略？

若两个配置在全部已用输入下满足
\[
P_{T_1,\theta_1}(Y\mid u)=P_{T_2,\theta_2}(Y\mid u),
\]
则它们在本观测实验下不可区分。等先验二选一时，任何判别器的平均正确率至多1/2。贝叶斯后验比仍等于先验比。这是直接由相同观测分布推出的结论；重复采样不会解决结构不可辨识。

可通过增加观测点或设计新的u改变实验，使两种配置的响应分开。响应相同与响应不同但噪声过大应分别报告。

## 3. 第一优先路线：距离误差界与四点条件

### 3.1 图论基础

对标量电阻树，采用统一的单位和参考节点，令
\[
K=(B^\top\operatorname{diag}(r_e^{-1})B)^{-1},
\quad
d_{ij}=K_{ii}+K_{jj}-2K_{ij}.
\]
则d_ij是i到j路径上的电阻之和。实际由P/Q/V估计R/X灵敏度时，需首先确认该矩阵是否对应此K；电压幅值与平方电压的系数不同，不能默认省去因子2。

一个有限度量为树度量，当且仅当每四个标签a,b,c,d的三个量
\[
S_1=d_{ab}+d_{cd},\quad
S_2=d_{ac}+d_{bd},\quad
S_3=d_{ad}+d_{bc}
\]
中，两个最大值相等。若真实四叶子结构为ab|cd，且内部边长为b>0，则
\[
S_1<S_2=S_3,\qquad S_2-S_1=2b.
\]
这是Buneman树度量理论的核心联系。原始电压差方差通常不等于这里的加性电阻距离。

### 3.2 从同时误差界得到局部分裂保证（本文推导）

假设已经独立建立覆盖事件
\[
\Pr\{\max_{i<j}|\widehat d_{ij}-d^\star_{ij}|\le\varepsilon_\alpha\}
\ge1-\alpha .
\tag{1}
\]
必须是**所有相关距离同时**的误差界，不能用每项各95%的区间代替。

每个S的误差至多2epsilon，任意两个S的差的误差至多4epsilon。因此若
\[
\widehat S_2-\widehat S_1>4\varepsilon_\alpha,\qquad
\widehat S_3-\widehat S_1>4\varepsilon_\alpha,
\tag{2}
\]
则在事件(1)上，真实四点关系必为ab|cd。还可给出
\[
b\ge\frac{\min(\widehat S_2,\widehat S_3)-\widehat S_1
-4\varepsilon_\alpha}{2}>0.
\tag{3}
\]
此保证来自树模型加上已校准误差事件，不是由间隙自行生成的概率。

**手算例：**观测三个和为(4,6,6)，同时误差半径0.1，则比较差的误差预算为0.4，能认证ab|cd，内部长度下界0.8。若半径0.6，这个局部判据不再足以认证；不能强迫二叉分裂。其他观测仍可能提供额外信息。

从K出发可以利用误差相关性更紧地计算对比量。例如
\[
S_2-S_1=2(K_{ab}+K_{cd}-K_{ac}-K_{bd}),
\]
对角项完全抵消。如果各K条目的同时误差为eta，则此对比误差至多8eta；机械先转换为每个d的4eta误差再组合会得到更保守的16eta。更优做法是直接对该线性对比构建同时区间。

### 3.3 Atteson正确恢复半径与局部边保证

Atteson (1999)证明经典Neighbor Joining在真树距离的无穷范数扰动小于最短边长的一半时恢复真树拓扑。它是**确定性扰动定理**：只有和式(1)这样的概率事件组合后，才成为概率恢复保证。最短真边通常未知，不能把估计最短边不加校准地代入。

Mihaescu、Levy、Pachter (2009)进一步区分整树l_infinity radius与单边edge radius：NJ最优edge radius为1/4，不能把整树1/2结果直接改写成任意长边各自的1/2保证。

这些定理针对相应NJ算法和加性树输入。它们不自动证明带根RNJ、阈值收缩、负边截断或后续候选MILP流水线的有效性。

### 3.4 误差界本身从哪里来

可选路径包括正确似然下的同时参数区域、适用条件下的去偏估计与Gaussian multiplier bootstrap，以及满足依赖条件的时间块方法。应计入根参考共同误差、P/Q输入误差、回归偏差、线性化误差和跨时间依赖。若只控制随机误差，模型偏差仍可令“很窄的区间”偏离真值。

按天或事件分组重采样有助于保留依赖结构，但仅仅把数据分块并不自动获得严格覆盖；仍需相应依赖理论或外部独立校准。

## 4. 从局部结论到整张图置信集合

### 4.1 集合反演（本文条件性命题）

令T类包含目标真树，并允许需要保留的多分叉或零长度极限。定义
\[
\mathcal C_\alpha=
\{T:\exists \ell\in\mathcal L_T,\ 
\max_{i<j}|d_{ij}(T,\ell)-\widehat d_{ij}|\le\varepsilon_\alpha\}.
\tag{4}
\]
如果式(1)成立、真图和真长度属于指定类，则直接有
\[
\Pr(T^\star\in\mathcal C_\alpha)\ge1-\alpha.
\tag{5}
\]
证明只需把真参数代入式(4)。这不是对某篇论文的原句转述，而是误差区域投影的基本推论。

可以保留非均匀同时区间[l_ij,u_ij]、椭球区域，或R/X共享拓扑的联合区域；不必强制使用统一epsilon。若R/X两通道共同覆盖需作联合校准或合理alpha分配，不能把同一批数据产生的两通道当独立证据相乘。

### 4.2 输出确定分裂、可能分裂和部分解析图

以S(T)表示同一终端集合上的非平凡split集合，且C非空，定义
\[
S^-=\bigcap_{T\in\mathcal C_\alpha}S(T),\qquad
S^+=\bigcup_{T\in\mathcal C_\alpha}S(T).
\]
那么在覆盖事件上，同时有
\[
S^-\subseteq S(T^\star)\subseteq S^+.
\tag{6}
\]
因此可以输出确定主干和未解析局部区域，而无需给每条边编造一个概率。隐藏节点情况下，这比比较任意潜在节点编号更稳妥。

所有同时认证的四点关系应共同兼容。局部四点关系不兼容，可能提示误差界失效、非树机制或建模偏差；不能用多数投票后继续声称原有覆盖。部分四点关系即使兼容，也不保证唯一整树。

### 4.3 最坏结构损失与审核（本文推导）

给定数据产生的任意候选图That及预先明确的结构损失L，定义
\[
U(D)=\sup_{T\in\mathcal C_\alpha(D)}L(\widehat T(D),T).
\]
由式(5)可得
\[
\Pr\{L(\widehat T,T^\star)>U(D)\}\le\alpha.
\]
若接受规则是C非空且U(D)<=tau，那么
\[
\Pr\{\text{接受且真实结构损失}>\tau\}\le\alpha.
\tag{7}
\]
**式(7)是联合错误事件控制，不自动等于接受子集中的条件错误率<=alpha。**条件错误率需要额外选择性风险校准；简单相除只能得到alpha/Pr(接受)的上界。

如果C为空，应标记模型/数据/误差范围不相容或计算问题；不能利用空集的逻辑真值宣布所有边都确定。

### 4.4 计算方向决定保证是否保留

只枚举有限候选得到C_pool⊆C，是内近似：会让歧义看起来更小，可能夸大确定结构。它不能自动用于式(5)—(7)。

若有覆盖真实C的外近似C_outer，则交集中的确定分裂更少、最坏损失更大，虽保守但可保留保证。为排除一个危险替代拓扑，需要全局不可行/界证据；最坏损失U的数值计算需要可靠上界，搜索到的最大损失只是下界，不能据此接受。启发式没找到、搜索超时不等于替代不存在。固定一个候选解的最优性也不等于排除了其他图。

## 5. 另一条强理论路线：分裂似然比与Universal Inference

Wasserman、Ramdas、Balakrishnan (2020)提供有限样本检验和置信集方法。以下将其profile-likelihood思想写成拓扑形式。

把独立案例分为训练A与验证B；使用A获得任意正规化预测密度q_A。对给定拓扑T定义
\[
E_T(D_B)=\frac{q_A(D_B)}
{\sup_{\theta\in\Theta_T}p_{T,\theta}(D_B)}.
\tag{8}
\]
若真实数据来自(T*,theta*)，则分母至少为真似然，从而条件于A，
\[
\mathbb E_{T^\star,\theta^\star}[E_{T^\star}\mid A]\le1.
\]
由Markov不等式，
\[
\mathcal C_\alpha=\{T:E_T<1/\alpha\}
\]
覆盖真拓扑的概率至少1−alpha。对所有T反演时，无需仅因图的数量巨大而机械进行一次Bonferroni；覆盖失败对应的是把真T排除。

**迁移价值：**图T离散、阻抗及噪声参数连续、模型在零边界处不规则，都可由适当的原假设模型与profile likelihood处理。训练端可使用启发式，只要其输出定义了验证数据上的正规化密度；有效性与检验功效要分开。

**计算注意：**若只能得到null最大似然的上界U_T，则q_A/U_T<=E_T，保守有效。对负对数似然而言，需要其全局最小值的下界。局部优化返回的可行最小化目标是上界，会导致最大似然的下界，拿来做分母可能放大E并破坏错误率控制。

**适用边界：**未知噪声不能不加约束地让似然退化；正确数据模型、训练验证独立性/正确条件化、缺失机制处理仍是实质条件。“Universal”不代表任意失配模型均有效。

Park、Balakrishnan、Wasserman (Biometrika 2026)的Robust universal inference可在条件下推断失配模型中的最佳投影/近似投影；该目标未必就是物理真拓扑。不能将“最接近真实分布的线性图”与“真实非线性电网结构”混为一谈。

## 6. 概率图分布：矩阵树定理与边的联合概率

如果候选节点已知，考虑正权候选图上的生成树分布
\[
P(T\mid D)=\frac{1}{Z(w)}\prod_{e\in T}w_e(D),\qquad
Z(w)=\sum_{T\text{为生成树}}\prod_{e\in T}w_e.
\tag{9}
\]
矩阵树定理给出Z等于加权拉普拉斯的任一主余子式行列式。通过对log Z求导，有
\[
P(e\in T\mid D)=w_e b_e^\top L_w^\dagger b_e,
\qquad \sum_e P(e\in T\mid D)=n-1.
\tag{10}
\]
式(10)同时联系边后验、有效电阻和杠杆分数。

**手算例：**三节点三条候选边权w_ab=w_bc=4，w_ac=1。三棵树权重16、4、4，Z=24。边边际概率分别5/6、5/6、1/3，总和2。最可能的树概率为2/3，而(5/6)^2=25/36，并非同一值。边之间受树约束，不能独立相乘。


**迁移条件：**只有先验与数据似然形成适当边因子，或明确采用这种结构化条件模型时，式(9)才有相应概率解释。任意相关系数/残差指数化得到的w，最多先称结构化评分分布；矩阵树定理只保证计算正确，不证明它已校准。

完整AC潮流似然、隐藏节点数不确定、未知线路参数积分通常不按边简单分解；不能无条件套一个行列式完成后验。候选图遗漏真边的问题仍存在。如果候选图本来只有一棵树，所有现有边的生成树概率都是1，完全不代表数据已经验证它们。

## 7. 数据可信度：割空间、环空间与物理检验

### 7.1 环路一致性及其盲区

令B∈R^(m×n)为边—节点关联矩阵，y为边上的电位差观测。理想模型为y=Bv。令C的行构成cycle space的一组基，则CB=0，因此Cy=0是KVL型循环一致性。

若加性误差是Bz，则C(y+Bz)=Cy。也就是说，**gradient/cut-space误差对循环检验不可见**。在线性状态估计z=Hx+e中，属于col(H)的误差同样可被状态吸收，保持拟合残差不变。相关电力系统论文给出了这一不可观测性机制。

连通树m=n−1，环空间维数m−n+1=0。对一条边仅有一个电位差测量的设置，仅靠KVL循环检验没有冗余。不能据此说所有树上的坏数据检测均无效：重复/额外传感器可使量测图有冗余，KCL、可信注入和时间关系也可提供额外检验。

若固定图、C及高斯测量协方差Sigma都正确，则
\[
Q=(Cy)^\top(C\Sigma C^\top)^\dagger(Cy)
\]
在无偏模型下服从自由度rank(C Sigma C^T)的卡方分布。这是线性高斯模型下的直接推导。若C是同一批数据选出的图决定的，或Sigma估计、非高斯、存在输入误差，不能直接沿用这个精确分布。

### 7.2 KCL提供另一类约束

对边流f和可信节点净注入s，B^T f=s。KCL残差为B^T f−s，而属于ker(B^T)的circulation/cycle-space误差不可见。KCL与KVL的盲区不同；应结合测量类型、冗余和可信锚点判断。

发生不一致只能先定位“模型—测量关系不相容”，不自动确定是哪一块表故障。拓扑错误、未记录负荷/注入、时间不同步、符号错误和传感器误差，都可能表现为同样的残差。

### 7.3 特征电流的集合一致性

对于Y=HC+E式的编码电流模型，理想H描述下游隶属关系，行集合应层叠。可检查集合包含的传递性、平行分支的互斥性，以及接收记录是否支持同一棵树。

误检可制造假包含，漏检可破坏真实包含；完全相同的接收集合意味着在当前探测配置下可能无法区分。若要输出“该接收记录为真的概率”，需显式设置检出/漏检/误检机制并估计参数；集合一致性本身只给出约束。

用同一批记录先选树，再把不符合该树的记录全部删除，可能形成自我强化。应采用独立验证、预先规定的诊断规则，或对筛选步骤整体建立推断理论。

## 8. 图信号平滑性、鲁棒学习与数据的信息量

### 8.1 平滑性用于异常筛查

图信号能量
\[
x^\top Lx=\sum_{\{i,j\}}w_{ij}(x_i-x_j)^2
\]
衡量相邻节点差异。GSP可在明确的平滑或频谱模型下构造异常检测器。

但低能量不等于数据真实：共同偏移c1位于拉普拉斯零空间，平滑错误可逃逸；正常负荷突变、调压动作也可能带来高频成分。若L正是用x的平滑性拟合出来的，再用同一x的平滑性验证L，属于循环证据。

Total Least Squares及鲁棒图学习可处理节点/边误差或离群值并改善估计，但“鲁棒”优化目标本身不自动给出校准后的图置信集合。需要核对噪声分布、异常比例/结构和独立验证。

### 8.2 可观测性、激励秩和有效电阻

在线性加权观测y=A beta+epsilon、已知协方差Sigma下，Fisher/最小二乘信息矩阵为
\[
F=A^\top\Sigma^{-1}A.
\]
F的小特征值对应弱信息方向。样本数量很大但输入高度共线时，某些线路参数或图差异仍缺乏证据。

对独立噪声下的相对测量，F具有加权拉普拉斯形式，锚定或商去常数后，电位差估计方差联系有效电阻：
\[
\operatorname{Var}(\widehat v_i-\widehat v_j)
=(e_i-e_j)^\top F^\dagger(e_i-e_j).
\]
这可指导在哪些位置增设相对测量；它不是该电表准确率。实际线路电阻、测量网络有效电阻和生成树分布中的有效电阻需按各自权重定义区分。

对预白化设计矩阵A，杠杆h_kk是A(A^TA)^dagger A^T的第k个对角元。残差方差为1−h_kk；高杠杆点的错误可能被拟合吸收。高杠杆意味着该点有影响力、缺乏冗余，不能解释为更可信。

针对图歧义，实验设计更应考察不同可行图在新输入u下的响应分离，并允许各图自己的未知参数调整。若某候选差异始终能被未知参数吸收，则仅增大同类输入次数未必有效。

## 9. 稳定选择、图性质检验与树空间统计

**稳定选择。**Meinshausen–Bühlmann给出在特定假设下控制期望误选数量的结果；它不同于普通bootstrap出现率。即使有PFER/FDR控制，也不等于“每条边正确概率”或“整树95%正确”。Shah–Samworth的CPSS在较弱假设下控制低基础入选概率变量的选择；若要把这些变量解释为真伪边，仍需额外联系。

**图性质检验。**Neykov、Lu、Liu研究连接性、环和最大度等全局性质的统计检验及信息论下界。Lu等的skip-down方法进一步构造单调图不变量的置信区间。这说明不必恢复每条边才有有意义的结构推断。但这些论文中的图主要是条件依赖图；迁移至电力物理图需建立对应关系。

**BHV树空间。**Willis (2019)的树值置信集针对一组树分布的Fréchet均值，使用树空间几何及渐近理论。它并不直接覆盖产生电力量测的单张物理真树。把bootstrap恢复出的树当成独立真树样本会改变推断对象并忽略每棵估计树的误差。

**校准过的bootstrap树集合。**Susko的minBP方法将最小split支持度当统计量并通过额外bootstrap校准集合阈值；它提供从稳定度走向树集合的范例。不能简单采用“bootstrap>=95%即95%置信”的规则。

## 10. 主动追加量测：置信序列与停止时机

固定样本95%方法通常不支持“每次看结果，够好就停止”仍保持95%。Howard等关于confidence sequences的理论，通过非负超鞅/e-process等构造
\[
\Pr\{\forall t,\ T^\star\in\mathcal C_t\}\ge1-\alpha .
\]
若激励u_t可预测，即只依赖此前信息，并且条件观测模型正确，可以研究适合主动探测的序贯似然构造。经典Universal Inference具有序贯扩展，但不能据此断言任何鲁棒失配版本都已具备同样保证。

本段是假设满足后可采用的理论路线；还需要处理真实负荷时间依赖、主动策略、物理模型失配、拓扑在采样过程中变化及计算近似。拓扑发生改变时，固定T*假设失效，需另建变化点/分段模型。

## 11. 适合本项目的理论组合与验证目标

建议优先推进以下组合，而不是先把多个相关评分加权：

1. 定义目标为末端约化树及其splits，并保留不可辨识等价类。
2. 对原始观测模型建立联合不确定性区域，明确EIV、根参考共同误差、时间相关与模型偏差。
3. 从联合区域得到距离/共享路径对比的同时界；用四点关系筛出可认证分裂。
4. 构造所有相容图的集合或保守外近似，输出确定主干与未解析部分。
5. 用独立验证的分裂似然比补充模型比较；计算无法完成时报告未排除的替代。
6. 选择最能区分剩余替代图的新量测/激励；反复监测则使用适当的序贯方法。

建议验证指标：整图集合覆盖率、确定split的同时错误率、集合大小/结构直径、空集率、拒识率、接受且损失超阈值的频率，以及在候选真值缺失、隐藏二度节点、根共同误差、时间相关、错误传感器和AC/线性失配下的变化。不能只统计成功返回单树的案例。

以上是文献基础上的研究路线；尚未据此实现或证明整个现有软件流水线。

## 12. 论文证据与检索入口

后续条目分别标注原始论文结果、条件及迁移边界。Google Scholar可用完整题名加双引号；IEEE Xplore对非IEEE文献提供主题词，不保证收录原文。链接为可重复使用的检索入口，未声称逐条记录数据库命中数。





### G1. A Note on the Metric Properties of Trees

Peter Buneman。Journal of Combinatorial Theory, Series B 17(1):48–50, 1974。DOI：10.1016/0095-8956(74)90047-1。

**论文结果：**四点条件刻画树度量，提供距离是否可由加权树表示的结构判据。

**适用边界：**不提供数据误差分布或置信水平；先要证明所用电气距离可加。

[一手来源](https://homepages.inf.ed.ac.uk/opb/homepagefiles/phylogeny-scans/metricproperties.pdf) · [DOI](https://doi.org/10.1016/0095-8956(74)90047-1)。

- Scholar：`"A Note on the Metric Properties of Trees"`。[检索](https://scholar.google.com/scholar?q=%22A%20Note%20on%20the%20Metric%20Properties%20of%20Trees%22)。
- Xplore相邻主题：`"tree metric" AND "four point"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22tree%20metric%22%20AND%20%22four%20point%22)。不保证Xplore收录该非IEEE原文。

### G2. The Performance of Neighbor-Joining Methods of Phylogenetic Reconstruction

Kevin Atteson。Algorithmica 25:251–278, 1999。DOI：10.1007/PL00008277。

**论文结果：**经典NJ的整树l_infinity安全半径为最短真边的1/2，具有最坏情况最优意义。

**适用边界：**确定性充分条件；未知真最短边、非加性输入、不同算法和零长边需要另行处理。

[一手来源](https://link.springer.com/article/10.1007/PL00008277) · [DOI](https://doi.org/10.1007/PL00008277)。

- Scholar：`"The Performance of Neighbor-Joining Methods of Phylogenetic Reconstruction"`。[检索](https://scholar.google.com/scholar?q=%22The%20Performance%20of%20Neighbor-Joining%20Methods%20of%20Phylogenetic%20Reconstruction%22)。
- Xplore相邻主题：`"neighbor joining" AND "error"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22neighbor%20joining%22%20AND%20%22error%22)。不保证Xplore收录该非IEEE原文。

### G3. Why Neighbor-Joining Works

Radu Mihaescu; Dan Levy; Lior Pachter。Algorithmica 54(1):1–24, 2009。DOI：10.1007/s00453-007-9116-4。

**论文结果：**将NJ与quartet consistency联系，证明单条split的最优edge radius为1/4。

**适用边界：**保证叶集合分裂，不直接给隐藏物理节点身份；不能混同整树1/2安全半径。

[一手来源](https://arxiv.org/abs/cs/0602041) · [DOI](https://doi.org/10.1007/s00453-007-9116-4)。

- Scholar：`"Why Neighbor-Joining Works"`。[检索](https://scholar.google.com/scholar?q=%22Why%20Neighbor-Joining%20Works%22)。
- Xplore相邻主题：`"neighbor joining" AND quartet`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22neighbor%20joining%22%20AND%20quartet)。不保证Xplore收录该非IEEE原文。

### G4. Using Minimum Bootstrap Support for Splits to Construct Confidence Regions for Trees

Edward Susko。Evolutionary Bioinformatics 2:129–143, 2006；电子发表记录为2007。DOI：10.1177/117693430600200030。

**论文结果：**以minBP作为统计量，使用双重bootstrap校准阈值，再寻找满足支持度约束的兼容树，形成近似覆盖的树区域。

**适用边界：**阈值不是直接95%；原结论不是无假设有限样本精确覆盖，时间依赖和搜索遗漏需要另行处理。

[一手来源](https://journals.sagepub.com/doi/abs/10.1177/117693430600200030) · [DOI](https://doi.org/10.1177/117693430600200030)。

- Scholar：`"Using Minimum Bootstrap Support for Splits to Construct Confidence Regions for Trees"`。[检索](https://scholar.google.com/scholar?q=%22Using%20Minimum%20Bootstrap%20Support%20for%20Splits%20to%20Construct%20Confidence%20Regions%20for%20Trees%22)。
- Xplore相邻主题：`"bootstrap" AND "tree" AND "confidence"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22bootstrap%22%20AND%20%22tree%22%20AND%20%22confidence%22)。不保证Xplore收录该非IEEE原文。

### G5. A Large-Deviation Analysis of the Maximum-Likelihood Learning of Markov Tree Structures

Vincent Y. F. Tan; Animashree Anandkumar; Lang Tong; Alan S. Willsky。IEEE Transactions on Information Theory 57(3):1714–1735, 2011。DOI：10.1109/TIT.2011.2104513。

**论文结果：**Chow–Liu结构学习的错误指数由最容易发生的路径真边—非边互信息排名反转决定，给出大偏差分析及有限样本上界。

**适用边界：**严格正有限字母表树分布和iid数据；统计Markov树不自动等于配电物理树，渐近指数不等于当前图后验。

[一手来源](https://arxiv.org/abs/0905.0940) · [DOI](https://doi.org/10.1109/TIT.2011.2104513)。

- Scholar：`"A Large-Deviation Analysis of the Maximum-Likelihood Learning of Markov Tree Structures"`。[检索](https://scholar.google.com/scholar?q=%22A%20Large-Deviation%20Analysis%20of%20the%20Maximum-Likelihood%20Learning%20of%20Markov%20Tree%20Structures%22)。
- Xplore题名：`"Document Title":"A Large-Deviation Analysis of the Maximum-Likelihood Learning of Markov Tree Structures"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Document%20Title%22%3A%22A%20Large-Deviation%20Analysis%20of%20the%20Maximum-Likelihood%20Learning%20of%20Markov%20Tree%20Structures%22)。

### G6. Tractable Bayesian Learning of Tree Belief Networks

Marina Meilă; Tommi Jaakkola。Statistics and Computing 16(1):77–92, 2006。DOI：10.1007/s11222-006-5535-3。

**论文结果：**通过可分解先验及特定完全观测模型，利用矩阵树定理解析计算树分布归一化与相关边际。

**适用边界：**需要边因子结构；一般AC似然不满足，固定支持图也可能遗漏真边。

[一手来源](https://link.springer.com/article/10.1007/s11222-006-5535-3) · [DOI](https://doi.org/10.1007/s11222-006-5535-3)。可读UAI版本：https://arxiv.org/abs/1301.3875。

- Scholar：`"Tractable Bayesian Learning of Tree Belief Networks"`。[检索](https://scholar.google.com/scholar?q=%22Tractable%20Bayesian%20Learning%20of%20Tree%20Belief%20Networks%22)。
- Xplore相邻主题：`"matrix tree theorem" AND Bayesian`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22matrix%20tree%20theorem%22%20AND%20Bayesian)。不保证Xplore收录该非IEEE原文。

### G7. Confidence Sets for Phylogenetic Trees

Amy Willis。Journal of the American Statistical Association 114(525):235–244, 2019。DOI：10.1080/01621459.2017.1395342。

**论文结果：**在BHV树空间中对树分布的Fréchet均值构造基于log-map和渐近理论的置信集合。

**适用边界：**目标是树分布均值，不是自动覆盖原始物理图；均值位于边界、输入树本身有估计误差时需额外处理。

[一手来源](https://arxiv.org/abs/1607.08288) · [DOI](https://doi.org/10.1080/01621459.2017.1395342)。

- Scholar：`"Confidence Sets for Phylogenetic Trees"`。[检索](https://scholar.google.com/scholar?q=%22Confidence%20Sets%20for%20Phylogenetic%20Trees%22)。
- Xplore相邻主题：`"phylogenetic trees" AND "confidence sets"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22phylogenetic%20trees%22%20AND%20%22confidence%20sets%22)。不保证Xplore收录该非IEEE原文。

### S1. Universal inference

Larry Wasserman; Aaditya Ramdas; Sivaraman Balakrishnan。PNAS 117(29):16880–16890, 2020。DOI：10.1073/pnas.1922664117。

**论文结果：**分裂似然比构造有限样本置信集，支持非规则模型、nuisance的profile likelihood及序贯扩展。

**适用边界：**需要正确的概率模型和数据使用；复合原假设最大似然应精确计算或保守上界。

[一手来源](https://arxiv.org/abs/1912.11436) · [DOI](https://doi.org/10.1073/pnas.1922664117)。

- Scholar：`"Universal inference"`。[检索](https://scholar.google.com/scholar?q=%22Universal%20inference%22)。
- Xplore相邻主题：`"universal inference" OR "split likelihood ratio"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22universal%20inference%22%20OR%20%22split%20likelihood%20ratio%22)。不保证Xplore收录该非IEEE原文。


### S2. Time-uniform, nonparametric, nonasymptotic confidence sequences

Steven R. Howard; Aaditya Ramdas; Jon McAuliffe; Jasjeet Sekhon。The Annals of Statistics 49(2):1055–1080, 2021。DOI：10.1214/20-AOS1991。

**论文结果：**在相应超鞅和尾部条件下构造所有时刻同时有效的区间，支持数据依赖停止。

**适用边界：**不自动适用于任意负荷时间序列；主动探测须采用可预测设计与正确条件观测模型。

[一手来源](https://arxiv.org/abs/1810.08240) · [DOI](https://doi.org/10.1214/20-AOS1991)。

- Scholar：`"Time-uniform, nonparametric, nonasymptotic confidence sequences"`。[检索](https://scholar.google.com/scholar?q=%22Time-uniform%2C%20nonparametric%2C%20nonasymptotic%20confidence%20sequences%22)。
- Xplore相邻主题：`"confidence sequences" OR "anytime valid"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22confidence%20sequences%22%20OR%20%22anytime%20valid%22)。不保证Xplore收录该非IEEE原文。

### S3. Robust universal inference for misspecified models

Beomjo Park; Sivaraman Balakrishnan; Larry Wasserman。Biometrika 113(2):asaf070, 2026；2025年在线发表。DOI：10.1093/biomet/asaf070。

**论文结果：**通过拆分与相对拟合检验，对失配模型中的投影或近似投影建立相应有限样本推断；不同版本保证不同。

**适用边界：**部分结果只保证与近似投影集合相交；投影图不必是真实物理图。该鲁棒版本的任意时刻扩展不能直接声称已完成。

[一手来源](https://academic.oup.com/biomet/article/113/2/asaf070/8321921) · [DOI](https://doi.org/10.1093/biomet/asaf070)。

- Scholar：`"Robust universal inference for misspecified models"`。[检索](https://scholar.google.com/scholar?q=%22Robust%20universal%20inference%20for%20misspecified%20models%22)。
- Xplore相邻主题：`"universal inference" AND misspecified`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22universal%20inference%22%20AND%20misspecified)。不保证Xplore收录该非IEEE原文。

### S4. Honest confidence regions and optimality in high-dimensional precision matrix estimation

Jana Janková; Sara van de Geer。TEST 26(1):143–162, 2017。DOI：10.1007/s11749-016-0503-5。

**论文结果：**去偏nodewise Lasso获得精度矩阵低维参数的均匀渐近正态及有效区间，不要求irrepresentability。

**适用边界：**需iid、sub-Gaussian、谱有界及s=o(sqrt(n)/log p)等条件；逐项区间不是整图同时区间，统计边未必是物理边。

[一手来源](https://www.research-collection.ethz.ch/server/api/core/bitstreams/1c3e7a03-b965-4b93-808e-99743bc042d9/content) · [DOI](https://doi.org/10.1007/s11749-016-0503-5)。

- Scholar：`"Honest confidence regions and optimality in high-dimensional precision matrix estimation"`。[检索](https://scholar.google.com/scholar?q=%22Honest%20confidence%20regions%20and%20optimality%20in%20high-dimensional%20precision%20matrix%20estimation%22)。
- Xplore相邻主题：`"precision matrix" AND "confidence intervals"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22precision%20matrix%22%20AND%20%22confidence%20intervals%22)。不保证Xplore收录该非IEEE原文。

### S5. Exact post-selection inference, with application to the lasso

Jason D. Lee; Dennis L. Sun; Yuekai Sun; Jonathan E. Taylor。The Annals of Statistics 44(3):907–927, 2016。DOI：10.1214/15-AOS1371。

**论文结果：**对Lasso选择事件进行条件化，利用截断正态分布构造选择后的参数推断。

**适用边界：**精确结果依赖固定设计、Gaussian误差及已刻画的选择；额外CV、筛边或挑选满意结果必须计入流程。

[一手来源](https://www.stat.cmu.edu/~ryantibs/statml/lectures/Lee-Sun-Sun-Taylor.pdf) · [DOI](https://doi.org/10.1214/15-AOS1371)。

- Scholar：`"Exact post-selection inference, with application to the lasso"`。[检索](https://scholar.google.com/scholar?q=%22Exact%20post-selection%20inference%2C%20with%20application%20to%20the%20lasso%22)。
- Xplore相邻主题：`"post selection inference" AND lasso`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22post%20selection%20inference%22%20AND%20lasso)。不保证Xplore收录该非IEEE原文。

### S6. Stability selection

Nicolai Meinshausen; Peter Bühlmann。Journal of the Royal Statistical Society Series B 72(4):417–473, 2010。DOI：10.1111/j.1467-9868.2010.00740.x。

**论文结果：**在噪声变量exchangeability、基础选择器等条件下，以半样本稳定选择控制期望误选数量。

**适用边界：**入选率不是后验；PFER不等于FDR、FWER或全树正确率；异质边交换性不能默认。

[一手来源](https://arxiv.org/abs/0809.2932) · [DOI](https://doi.org/10.1111/j.1467-9868.2010.00740.x)。

- Scholar：`"Stability selection"`。[检索](https://scholar.google.com/scholar?q=%22Stability%20selection%22)。
- Xplore相邻主题：`"stability selection" AND "graphical model"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22stability%20selection%22%20AND%20%22graphical%20model%22)。不保证Xplore收录该非IEEE原文。

### S7. Variable selection with error control: another look at stability selection

Rajen D. Shah; Richard J. Samworth。Journal of the Royal Statistical Society Series B 75(1):55–80, 2013。DOI：10.1111/j.1467-9868.2011.01034.x。

**论文结果：**CPSS使用互补半样本，在较弱假设下控制低基础入选概率变量的选入，基本界对有限配对次数成立。

**适用边界：**其控制对象不是自动等同真实伪边；更强界需要附加形状假设。

[一手来源](https://arxiv.org/abs/1105.5578) · [DOI](https://doi.org/10.1111/j.1467-9868.2011.01034.x)。

- Scholar：`"Variable selection with error control: another look at stability selection"`。[检索](https://scholar.google.com/scholar?q=%22Variable%20selection%20with%20error%20control%3A%20another%20look%20at%20stability%20selection%22)。
- Xplore相邻主题：`"complementary pairs stability selection"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22complementary%20pairs%20stability%20selection%22)。不保证Xplore收录该非IEEE原文。

### S8. Combinatorial inference for graphical models

Matey Neykov; Junwei Lu; Han Liu。The Annals of Statistics 47(2):795–827, 2019。DOI：10.1214/17-AOS1650。

**论文结果：**研究连接性、环、最大度等图性质的结构检验，以及由图结构复杂度决定的信息论下界。

**适用边界：**主要为Gaussian条件依赖图；迁移到物理电网前须证明观测分布与物理拓扑的映射。

[一手来源](https://arxiv.org/abs/1608.03045) · [DOI](https://doi.org/10.1214/17-AOS1650)。

- Scholar：`"Combinatorial inference for graphical models"`。[检索](https://scholar.google.com/scholar?q=%22Combinatorial%20inference%20for%20graphical%20models%22)。
- Xplore相邻主题：`"graphical models" AND ("connectivity" OR "hypothesis testing")`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22graphical%20models%22%20AND%20(%22connectivity%22%20OR%20%22hypothesis%20testing%22))。不保证Xplore收录该非IEEE原文。

### S9. Adaptive Inferential Method for Monotone Graph Invariants

Junwei Lu; Matey Neykov; Han Liu。arXiv:1707.09114, 2017；本轮仅核预印本版本。

**论文结果：**skip-down框架对单调图不变量构造检验和置信区间，可在弱于完全图恢复的条件下推断性质。

**适用边界：**图不变量的选择、信号强度、统计图模型和渐近条件需要对应；不能把强制的径向约束当检验结果。

[一手来源](https://arxiv.org/abs/1707.09114)。


- Scholar：`"Adaptive Inferential Method for Monotone Graph Invariants"`。[检索](https://scholar.google.com/scholar?q=%22Adaptive%20Inferential%20Method%20for%20Monotone%20Graph%20Invariants%22)。
- Xplore相邻主题：`"graph invariants" AND "confidence"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22graph%20invariants%22%20AND%20%22confidence%22)。不保证Xplore收录该非IEEE原文。

### D1. How to Learn a Graph from Smooth Signals

Vassilis Kalofolias。AISTATS, PMLR 51:920–929, 2016。

**论文结果：**以图平滑性及边权正则化学习图，提供可扩展优化算法。

**适用边界：**平滑是先验与目标，不是数据鉴真证据；同批数据拟合和验证会循环。

[一手来源](https://proceedings.mlr.press/v51/kalofolias16.html)。

- Scholar：`"How to Learn a Graph from Smooth Signals"`。[检索](https://scholar.google.com/scholar?q=%22How%20to%20Learn%20a%20Graph%20from%20Smooth%20Signals%22)。
- Xplore相邻主题：`"graph learning" AND "smooth signals"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22graph%20learning%22%20AND%20%22smooth%20signals%22)。不保证Xplore收录该非IEEE原文。

### D2. Detection of False Data Injection Attacks in Smart Grids Based on Graph Signal Processing

Elisabeth Drayer; Tirza Routtenberg。IEEE Systems Journal 14(2):1886–1896, 2020。DOI：10.1109/JSYST.2019.2927469。

**论文结果：**对电网状态做图高通筛查，IEEE14节点AC算例检出传统残差法漏掉的部分异常。

**适用边界：**需正常状态平滑且异常进入可见高频；低频/零空间误差、真实运行变化与图错误要区分。

[一手来源](https://arxiv.org/abs/1810.04894) · [DOI](https://doi.org/10.1109/JSYST.2019.2927469)。

- Scholar：`"Detection of False Data Injection Attacks in Smart Grids Based on Graph Signal Processing"`。[检索](https://scholar.google.com/scholar?q=%22Detection%20of%20False%20Data%20Injection%20Attacks%20in%20Smart%20Grids%20Based%20on%20Graph%20Signal%20Processing%22)。
- Xplore题名：`"Document Title":"Detection of False Data Injection Attacks in Smart Grids Based on Graph Signal Processing"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Document%20Title%22%3A%22Detection%20of%20False%20Data%20Injection%20Attacks%20in%20Smart%20Grids%20Based%20on%20Graph%20Signal%20Processing%22)。

### D3. Statistical ranking and combinatorial Hodge theory

Xiaoye Jiang; Lek-Heng Lim; Yuan Yao; Yinyu Ye。Mathematical Programming 127:203–244, 2011。DOI：10.1007/s10107-010-0419-x。

**论文结果：**将成对比较边流分解为梯度及循环不一致成分，区分局部curl与全局harmonic部分。

**适用边界：**原任务为排序；迁移电网时必须区分有符号电位差、支路电流和非负路径距离，不能混用约束。

[一手来源](https://web.stanford.edu/~yyye/hodgeRank2011.pdf) · [DOI](https://doi.org/10.1007/s10107-010-0419-x)。

- Scholar：`"Statistical ranking and combinatorial Hodge theory"`。[检索](https://scholar.google.com/scholar?q=%22Statistical%20ranking%20and%20combinatorial%20Hodge%20theory%22)。
- Xplore相邻主题：`"Hodge decomposition" AND inconsistency`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Hodge%20decomposition%22%20AND%20inconsistency)。不保证Xplore收录该非IEEE原文。

### D4. False Data Injection Attacks against State Estimation in Electric Power Grids

Yao Liu; Peng Ning; Michael K. Reiter。ACM Transactions on Information and System Security 14(1), Article 13, 2011。DOI：10.1145/1952982.1952995。

**论文结果：**在线性状态估计中刻画保持坏数据残差不变的不可观测扰动子空间。

**适用边界：**这是残差检测盲区的理论证据，不是未知图条件下的完整数据真实性判定。

[一手来源](https://reitermk.github.io/papers/2011/TISSEC2.pdf) · [DOI](https://doi.org/10.1145/1952982.1952995)。

- Scholar：`"False Data Injection Attacks against State Estimation in Electric Power Grids"`。[检索](https://scholar.google.com/scholar?q=%22False%20Data%20Injection%20Attacks%20against%20State%20Estimation%20in%20Electric%20Power%20Grids%22)。
- Xplore相邻主题：`"state estimation" AND ("unobservable" OR "bad data detection")`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22state%20estimation%22%20AND%20(%22unobservable%22%20OR%20%22bad%20data%20detection%22))。不保证Xplore收录该非IEEE原文。

### D5. Graph-Based Learning Under Perturbations via Total Least-Squares

Elena Ceci; Yanning Shen; Georgios B. Giannakis; Sergio Barbarossa。IEEE Transactions on Signal Processing 68:2870–2882, 2020。DOI：10.1109/TSP.2020.2982833。

**论文结果：**正则化TLS同时处理图模型及信号扰动，提供结构化和稀疏扩展；含驻点收敛和部分子问题的优化保证。

**适用边界：**一般图模型的优化保证不等于电网拓扑恢复或校准置信保证。

[一手来源](https://iris.uniroma1.it/handle/11573/1390351) · [DOI](https://doi.org/10.1109/TSP.2020.2982833)。

- Scholar：`"Graph-Based Learning Under Perturbations via Total Least-Squares"`。[检索](https://scholar.google.com/scholar?q=%22Graph-Based%20Learning%20Under%20Perturbations%20via%20Total%20Least-Squares%22)。
- Xplore题名：`"Document Title":"Graph-Based Learning Under Perturbations via Total Least-Squares"`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22Document%20Title%22%3A%22Graph-Based%20Learning%20Under%20Perturbations%20via%20Total%20Least-Squares%22)。

### D6. Graph Sparsification by Effective Resistances

Daniel A. Spielman; Nikhil Srivastava。SIAM Journal on Computing 40(6):1913–1926, 2011。DOI：10.1137/080734029。

**论文结果：**按有效电阻相关概率抽样，以O(n log n/epsilon^2)条边保持所有拉普拉斯二次型至1±epsilon。

**适用边界：**这是谱近似而非精确边恢复；杠杆分数衡量信息或不可替代性，不是测量可靠概率。

[一手来源](https://arxiv.org/abs/0803.0929) · [DOI](https://doi.org/10.1137/080734029)。

- Scholar：`"Graph Sparsification by Effective Resistances"`。[检索](https://scholar.google.com/scholar?q=%22Graph%20Sparsification%20by%20Effective%20Resistances%22)。
- Xplore相邻主题：`"effective resistance" AND ("leverage" OR sparsification)`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22effective%20resistance%22%20AND%20(%22leverage%22%20OR%20sparsification))。不保证Xplore收录该非IEEE原文。

### D7. Minimizing Effective Resistance of a Graph

Arpita Ghosh; Stephen Boyd; Amin Saberi。SIAM Review 50(1):37–66, 2008。DOI：10.1137/050645452。

**论文结果：**固定图上总有效电阻的预算权重分配为凸优化；连接成对差值测量的A-optimal实验设计。

**适用边界：**相对测量模型及噪声条件须匹配；实际电流注入需重新推导信息矩阵。

[一手来源](https://web.stanford.edu/~boyd/papers/eff_res.html) · [DOI](https://doi.org/10.1137/050645452)。

- Scholar：`"Minimizing Effective Resistance of a Graph"`。[检索](https://scholar.google.com/scholar?q=%22Minimizing%20Effective%20Resistance%20of%20a%20Graph%22)。
- Xplore相邻主题：`"effective resistance" AND ("experimental design" OR "sensor placement")`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22effective%20resistance%22%20AND%20(%22experimental%20design%22%20OR%20%22sensor%20placement%22))。不保证Xplore收录该非IEEE原文。

### D8. An outlier-robust smoothness-based graph learning approach

Hesam Araghi; Massoud Babaie-Zadeh。Signal Processing 206:108927, 2023。DOI：10.1016/j.sigpro.2023.108927。


**论文结果：**联合异常补偿与平滑图学习，BCD收敛到驻点；仿真表明在设定污染机制下改善估计。

**适用边界：**本轮核出版社摘要与可见正文；稀疏异常和平滑清洁信号假设不覆盖所有系统偏差，不输出真实性概率。

[一手来源](https://www.sciencedirect.com/science/article/abs/pii/S0165168423000014) · [DOI](https://doi.org/10.1016/j.sigpro.2023.108927)。

- Scholar：`"An outlier-robust smoothness-based graph learning approach"`。[检索](https://scholar.google.com/scholar?q=%22An%20outlier-robust%20smoothness-based%20graph%20learning%20approach%22)。
- Xplore相邻主题：`"graph learning" AND outlier AND smoothness`。[检索](https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=%22graph%20learning%22%20AND%20outlier%20AND%20smoothness)。不保证Xplore收录该非IEEE原文。

## 13. 补充：生成树的边替换间隙可给出可计算证书

以下是利用MST循环性质的直接推导，不冒充Tan等的概率定理。

在固定候选图上，设估计边权wh与总体边权w满足同时界max_e|wh_e−w_e|<=epsilon。对估计最小生成树That定义
\[
\widehat\Delta=
\min_{f\notin\widehat T}
\left[\widehat w_f-
\max_{e\in\operatorname{path}_{\widehat T}(f)}\widehat w_e\right].
\]
若Delta_hat>2epsilon，则每个非树边在整个误差盒中都严格比其树路径上的任何边更重，由cycle property可知That始终是唯一MST。

这证明总体边权下的最优树稳定；要推出物理树正确，还需要总体边权MST与真实电网一致的物理定理，以及真边未被候选图排除。若候选图本身只有一棵树，没有替代边，不能用空比较集宣称数据已验证真实结构。

## 14. 本轮置信集与上轮共形预测的区别

本轮主要对象是**同一未知固定网络的结构参数**：反复取得量测后，要求对真结构有频率学覆盖。可以依赖明确噪声模型、同时估计界、检验反演，而不一定需要许多已标注台区。

常规分类共形预测则是**跨新案例的标签预测**，需要相应交换性校准案例及标签定义。不能将同一固定网络的连续时间点直接充当许多独立新拓扑案例，或把两种覆盖混为一谈。

图置信集合覆盖不等于每个已观测数据条件下真图落入集合的概率；贝叶斯可信集合、参数置信集合和预测集合的概率对象应分别说明。

四点判据中的“内部边长”是四叶子诱导树上的内部路径长度，可能合并原图多条边；即使局部分裂已确认，也不能据此确认某一条具名物理导线。

若候选筛选步骤另有真结构保留率至少1−beta，再与alpha级推断结合，通常先由union bound得到1−alpha−beta的整体覆盖下界。没有筛选保真保证时，不应自行把beta当零。

## 15. 优先阅读与落地难度

| 顺序 | 文献/路线 | 适合解决的问题 | 主要工作量 |
|---|---|---|---|
| 1 | Buneman、Atteson、Mihaescu | 哪些终端划分足够可分辨 | 建立距离的同时误差界；处理多分叉与隐藏节点 |
| 2 | Universal Inference | 离散图与连续参数的整图置信集 | 构建正确观测似然与保守全局优化 |
| 3 | Hodge、状态估计残差盲区 | 哪些不一致能被检测，哪些无法自检 | 区分物理图、量测图、已知注入与未知注入 |
| 4 | Howard置信序列 | 主动追加探测并选择停止时机 | 条件噪声模型、依赖和可预测激励 |
| 5 | 矩阵树概率模型 | 在合法树空间中给出联合概率 | 推导可分解似然或明确评分模型并校准 |
| 6 | 图平滑/TLS/鲁棒学习 | 辅助异常筛查与稳定估计 | 独立验证，避免将优化目标当真实性 |

本报告保留了既有理论的适用条件，也给出若干简单但可复核的迁移推导。没有把上述通用论文解释为已经证明当前RNJ/层叠MILP完整软件流程具有95%物理拓扑保证。




