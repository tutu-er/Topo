# 简化RNJ/WZZT的数学结构

本目录只保留一个共同的训练目标、一套固定族拟合，以及W/T/E三种结构搜索域。RNJ、NJ与八个R/T/E组合共用观测定义、响应系数界和评分口径；W类共用路径验证选择规则。实现位置为 [trees.py](trees.py)、[optimization.py](optimization.py)、[algorithm.py](algorithm.py)。

## 1. 信息与响应模型

有$n$个观测终端、$T$个训练时点。$P,Q,Y\in\mathbb R^{T\times n}$按同一终端顺序排列；$Y_{ti}=V_{0t}^{2}-V_{it}^{2}$。算法接收训练和验证观测；真实树与独立测试观测只用于事后评分。

每个非空终端集合$C$有指示向量$z_C$。一条连接由其下游观测终端集合表示：

$$
R=\sum_{C\in\mathcal F}r_Cz_Cz_C^\top,\qquad
X=\sum_{C\in\mathcal F}x_Cz_Cz_C^\top,\qquad
\widehat Y=PR^\top+QX^\top.
$$

$r_C,x_C\in[0,u]$，默认$u=2$，是当前单位下的响应系数，不直接等于欧姆阻抗。两个集合必须包含或分离；交叉集合不能同时存在。默认基底为

$$
\mathcal B=\{\{1\},\ldots,\{n\},[n]\}.
$$

代码使用零基列位置，结果同时保存物理终端标签。基底不计入内部连接数$k(\mathcal F)$。

## 2. 一个固定族QP与一个结构目标

对每个固定族，预计算特征列$F^P_C=\operatorname{vec}((Pz_C)z_C^\top)$、$F^Q_C=\operatorname{vec}((Qz_C)z_C^\top)$。全部权重共同拟合：

$$
M(\mathcal F)=
\min_{0\le\theta\le u}
\frac{\|[F^P,F^Q]\theta-\operatorname{vec}(Y)\|_2^2}{nT}.
$$

以训练数据生成的RNJ/RX75、相对分组容差0.16支持族为公共参考，令$M_{\rm ref}=M(\mathcal F_{\rm ref})>0$。参考误差为零或无效时显式报错。全部结构使用

$$
\boxed{J(\mathcal F)=\frac{M(\mathcal F)}{M_{\rm ref}}+\alpha k(\mathcal F),\qquad\alpha=0.003.}
$$

固定族由SciPy有界最小二乘BVLS求解，等价于凸QP。保存可行目标和盒约束支撑平面下界；不强求秩亏时不同求解器给出相同系数。没有ridge或正阻抗下界。

每个提议先QP重拟合，再清理允许删除的数值零块。仅当$\max(r_C,x_C)\le10^{-9}$、且删除后归一化训练损失增量不超过$10^{-10}$时释放该块。基底与本次T的固定骨架受到保护。然后统一按

$$
\boxed{J_{\rm old}-J_{\rm proposed}>10^{-10}}
$$

接收。这里的数值容差直接作用于本目录的归一化目标；与旧实验内部缩放常数并不逐字等同。

可选相对改善参数$0\le\kappa<1$（`relative_gain`默认0）将普通结构接收改为

$$
J_{\rm old}-J_{\rm proposed}>\max(10^{-10},\kappa J_{\rm old}).
$$

它表示整个正则目标的相对下降，包含原内部连接数成本，不等于纯MSE相对下降。纯数值零块删除仅当无新增支持、删除块原权重≤零容差、前后活动内部族相同且训练拟合不劣时，仍只经过数值门槛；否则按普通相对规则。

同界最优拟合下，纯删块缩小可行域，最佳训练MSE不会下降。J允许用更少连接补偿拟合代价，因此退出不再另设“训练MSE必须改善”的门槛。

## 3. R：训练几何、候选与热启动

先拟合$Y=PR+QX$，施加对称、非负和逐项对角顺序约束$M_{ii}-M_{ij}\ge\delta_M$，$M\in\{R,X\}$。$\delta_M$由对称OLS一次确定，比例为$10^{-6}$；顺序约束不等于对角占优或半正定约束。

用正成对距离均值$\mu_R,\mu_X$归一化：

$$
S_\rho=\rho R/\mu_R+(1-\rho)X/\mu_X,\qquad d_i=(S_\rho)_{ii}.
$$

固定$\rho\in\{1,0.75,0\}$及相对分组容差$\eta\in\{0,0.04,0.08,0.16,0.24\}$，产生15个raw RNJ族。其全部原始支持构成$\mathcal P_R$，族本身作为热启动；不先按拟合零权重剪掉生成支持。

R开启也从$\mathcal B$开始，不强制保留两个小种子。R关闭时去掉这些结构候选与热启动；公共参考误差仍由相同RNJ参考族计算，保证计数惩罚尺度一致。

RNJ按共享根路径最大值合并终端组。NJ用同一几何和观测根锚点，$D_{ij}=d_i+d_j-2S_{ij}$、$D_{0i}=d_i$，执行经典Q准则；两基线均在选定族上QP重拟合和数值清零。它们是本目录的本地对照实现。

## 4. W：一个未知支持的原始WZZT

W每轮寻找一个新集合$C$，$2\le|C|\le n-1$，与当前族相容且不重复；全部旧权重同时重拟合。R开启时，还逐个拟合池中的相容单候选，进行QP和零清理，保留其中最好的清理后模型。它既提供原生MIQP热启动，也参与最终的提议比较；原生MIQP的未知支持域仍不受R池限制。

令$z_i\in\{0,1\}$选择终端，$g_{ij}=z_iz_j$。使用

$$
0\le g_{ij}\le z_i,\quad g_{ij}\le z_j,\quad
g_{ij}\ge z_i+z_j-1
$$

及有界乘积约束精确表示$u^r_{ij}=r_{\rm new}g_{ij}$、$u^x_{ij}=x_{\rm new}g_{ij}$：

$$
0\le u^r_{ij}\le r_{\rm new},\quad u^r_{ij}\le ug_{ij},\quad
u^r_{ij}\ge r_{\rm new}-u(1-g_{ij}),
$$

$u^x$同理。二元激活$a$满足$r_{\rm new},x_{\rm new}\le ua$；新预测项为$P(U^r)^\top+Q(U^x)^\top$。相容关系也由线性约束表示。因此整数可行解上乘积关系精确，连续松弛的平方目标为凸二次函数，W是凸MIQP。

原生W目标中的旧连接数为常数，新增成本为$\alpha a$。后续QP与零清理可改变规范化后的支持族；原生下界只对应原生编码的域和计数，不能直接冒充整个清理后搜索的全局证书。

## 5. T：全部有界子树切面的联合重构

T开启且当前最大直属子块数不超过$b_{\max}=5$时结束W，至多执行一次T。枚举当前每个节点$V$的所有完整子树切面

$$
V=B_1\mathbin{\dot\cup}\cdots\mathbin{\dot\cup}B_b,\qquad b\le b_{\max}.
$$

每个块是当前完整子树或单点，限制的是块数而非块内终端数。每个切面生成

$$
C_S=\bigcup_{a\in S}B_a,\qquad 2\le|S|\le b-1.
$$

一个五块切面至多25个候选；所有切面合并后的池可大得多。出现在这些候选中的旧内部支持允许释放，其余骨架固定。当前解保留在域内。所有区域一次共同求解，联合重拟合量测目标。

W在大区域停滞或预算耗尽时可跳过T。T后不回到W。

## 6. T/E共同选择模型与E范围

给定连接池$\mathcal P$及必选支持$\mathcal H$，每个普通内部候选有$a_C\in\{0,1\}$：

$$
0\le r_C,x_C\le ua_C,\qquad
a_C+a_D\le1\quad(C,D\text{交叉}).
$$

$\mathcal H$中的内部支持固定$a_C=1$。另加入层级计数强化约束$\sum_{C\subsetneq W}a_C\le|W|-2$。候选特征已固定，共同最小化归一化平方损失加$\alpha\sum_Ca_C$，因此T/E也是凸MIQP。

T的$\mathcal H$为保留骨架。E的$\mathcal H=\mathcal B$，候选池为

$$
\mathcal P_E=\mathcal P_R
\cup\bigcup_{\mathcal F\in\text{自身路径}}\mathcal F
\cup\mathcal P_T
\cup\{\{i,j\}:i<j\}.
$$

R关闭时$\mathcal P_R$只含基底。E至多求解一次，允许全部临时内部连接退出或替换，包括大块。有限池不保证包含真实连接；加大求解时间不能补出池外连接。

## 7. 输出、消融和预算

W类方法保存自身所有已接收模型，并加入单点响应基线。当前最终严格比较独立验证MSE，实际排序分数为$\mathrm{MSE}_{\rm val}/M_{\rm ref}$；正的共同参考只改变单位，不改变MSE排序。在这个规范化分数的$10^{-12}$内并列时，优先少内部连接，再按MSE分数与路径顺序；对应物理MSE容差为$10^{-12}M_{\rm ref}$。`selected`和`selected_mse_view`指向同一主选模型；`selected_mae_view`仅作诊断，不参与采纳。系数不重新拟合。RNJ/NJ基线只输出各自固定族的QP和数值清零结果，不与单点族择优。输出没有0.24收缩或完整RNJ整树额外备用。

R/T/E三位开关组成八格，详见 [ALGORITHMS.md](ALGORITHMS.md)。所有格共享参考尺度；E关闭的格把结构预算全部用于W/T，E开启的格默认先给W/T剩余结构预算的一半，再给E余量。模块效应包含这种明确的预算分配。

默认每方法/输入总记账预算14400秒，W单次MIQP上限3600秒，输出预留600秒。Gurobi具有原生TimeLimit；SciPy固定QP的300秒是求解返回后的软检查，BVLS另限5000次迭代。上游几何和SciPy调用均不能由此代码硬中断。超预算以budget_exceeded独立终态保留分母，不混入有效精度均值；公共预处理同时核对训练数据指纹，避免错用其他输入的参考尺度。

MIQP相对gap默认$10^{-4}$、归一化目标绝对gap$10^{-7}$；系数界、零容差和RNJ网格属于固定公开设置。本目录主要可调统计参数为$\alpha$，块宽会改变候选覆盖。有限域gap、低拟合误差或小型测试均不构成全网络树全局最优或唯一可识别性保证。
