# 直接比较候选的局部电压响应

本轮自行推导与实现。完整AC评分是原始高斯P/Q/V模型的局部拟合；响应评分是在局部线性物理关系上的近似评分。二者用相同带噪读数，但不是同一个似然的恒等变形。

## 1. 符号与数据

每时段有m=3个末端，p、q、v均为m维列向量，单位为标幺；观测加波浪号。原始表误差独立高斯，已知标准差sigma_p、sigma_q、sigma_v，固定仪表偏差已校准。块内没有额外未计量注入；块外变化经共同入口电压作用。训练长度N=288或672，独立留出96点。

候选树k的M是“末端×边”路径矩阵，M[j,e]=1表示边e在末端j的路径上。共享线路系数为r_e、x_e；R=M diag(r) M'，X=M diag(x) M'。各时段线路相同，输入P/Q不同。

D是2×3正交对比矩阵，D 1=0，DD'=I，例如第一行为(1,−1,0)/sqrt(2)，第二行为(1,1,−2)/sqrt(6)。它去掉共同电压水平，保留两维相对电压信息。

## 2. 完整AC参考评分

令f_k(r,x,p_t,q_t,b_t)为在入口幅值b_t下精确AC得到的末端幅值。原目标S为

S_k=min sum_t [||p_t−p_obs,t||²_{Sigma_p^-1}+||q_t−q_obs,t||²_{Sigma_q^-1}
                    +||f_k(r,x,p_t,q_t,b_t)−v_obs,t||²_{Sigma_v^-1}]。

J=S/2；所有物理域沿用旧批次。训练共同估计r/x和逐时状态。实际程序给多初值达到的局部值，不证明全局最小。留出时固定训练r/x，只估当期p/q/b；它评估模型解释新带噪计量的能力，不使用新真实P/Q，也不重新调整线路。

## 3. 解析消去P/Q

先考虑线性误差变量模型：x=(p',q')'，y=H x，H=[−2DR,−2DX]。观测为x_obs=x+epsilon_x、y_obs=y+epsilon_y，协方差Sigma_x和Sigma_y给定。令a=y_obs−H x_obs，Omega=Sigma_y+H Sigma_x H'。

固定H时，对隐变量x最小化两类标准化误差：

Q(H)=min_x {||x_obs−x||²_{Sigma_x^-1}+||y_obs−Hx||²_{Sigma_y^-1}}
    =a' Omega^-1 a。

证明：令d=x−x_obs，目标为d'Sigma_x^-1 d+(a−Hd)'Sigma_y^-1(a−Hd)。一阶条件给d=Sigma_x H' Omega^-1 a；代回并使用Omega=Sigma_y+H Sigma_x H'得到上式。Sigma_y正定，固定H的隐变量问题严格凸，解唯一。

因此x_hat=x_obs+Sigma_x H' Omega^-1 a。全记录得分为sum_t a_t'Omega_t^-1 a_t，每时段只求2×2线性系统，不在每次线路更新里求完整AC。

若w_t=Omega_t^-1 a_t，包络定理给

gradient_H Q=−2 sum_t w_t x_hat,t'。

它包括Omega随H变化的导数；只微分a会漏项。实现用显式隐变量直解及有限差分检查该公式。这里是对固定未知P/Q做profile（最小化），没有log det Omega项；若给P/Q分布并积分，才是另一种概率模型。

训练仍约束H属于候选树允许的响应：逐边|z|在原范围内、X/R在原范围内；H不是任意2×6矩阵。D M的共同主干列为零，该边系数不由局部评分估计。原幅值域的共享系数优化仍非凸，但只剩6或8个可见参数，4初值保留停点。

本轮实际使用y_obs=D(v_obs²−sigma_v²)。若原幅值误差为epsilon_v，则

y_obs=H x+D eta_AC+2D diag(v)epsilon_v+D(epsilon_v²−sigma_v²)。

后两项不构成精确高斯分布，eta_AC为真实损耗余项。主响应评分先冻结参考幅值v_ref=1，使用

Sigma_y=D diag(4 sigma_v² v_ref²+2 sigma_v^4) D'。

因此解析消元对所声明的线性误差模型准确；与完整AC评分的差别必须用实验量化。隐变量P/Q未加硬功率域，另报告其是否满足原域；不能据此宣称与原AC拟合完全等价。

## 4. 凸QP版本及界方向

固定正定权重Omega_bar，H对r/x仿射，所以

B=min_{theta in P} sum_t ||Omega_bar,t^-1/2(y_obs,t−H(theta)x_obs,t)||²

是凸QP。theta为可见r/x，P是原物理域的凸外包。保留分量范围、X/R线性范围、最小幅值弦及最大幅值圆的8个外切平面。设phi范围[a,b]，c=(a+b)/2，h=(b−a)/2，则

cos(c)r+sin(c)x >= z_min cos(h)

对每个原合法点成立：左边=|z|cos(phi−c)>=z_min cos(h)。这是线性外约束，不把非凸的r²+x²>=z_min²直接称QP。圆上界外切面也只放宽原域。

QP_N将权重冻结在公开域中点响应上，作为快速评分对照，没有全AC下界含义。

QP_U用公开v_max和H各元素的全域绝对上界构造Omega_bar，使Omega_bar >= Sigma_y+H Sigma_x H'（正半定意义）。其中H Sigma_x H' <= trace(H Sigma_x H')I；以各元素上界替代trace得到可预计算上界。由逆矩阵序及外域包含性，QP_U最优值不大于原物理域RESP-EIV最优值。这个界仅针对线性响应模型；真实AC还需处理eta。

任意QP可行点给最小值的上界。独立下界来自凸切线：f(theta)>=f(theta0)+g'(theta−theta0)。若P={l<=theta<=u,A theta<=b}，任意lambda>=0给

L=f(theta0)−g'theta0 + min_{l<=theta<=u}(g+A'lambda)'theta−lambda'b。

min_box按各分量符号选择端点。lambda可由LP求得，但下界有效性只需非负和上述显式算术，不依赖LP成功标志。返回max(0,L−guard)、primal与gap。机器计算使用浮点保护量，不是外向舍入的形式证书。gap不合格的记录保留。

## 5. 局部AC余项如何收紧

令ell_e=|I_e|²，f为e的严格后代边。精确电流/功率平衡给

C_e=|z_e|² ell_e+2 sum_f (r_e r_f+x_e x_f)ell_f，
D v²=−2D(Rp+Xq)−(DM) C。

所以D eta_AC=−(DM)C。第二项中的祖先传播必须保留。全末端共干满足DM[:,e]=0，其自身及所有交叉损耗贡献严格抵消；共同入口参考也抵消。

原域内电流有上界：|I_e|<=sum_{j in descendants(e)} |p_j+i q_j|/v_j。公共功率cap和v_min给第一层界。进一步用全部观测P/Q/V的同时高斯矩形与物理域相交，得到每个末端电流幅值范围[a_j,b_j]，进而

max(0,max_j(a_j−sum_{l!=j}b_l)) <= |I_e| <= sum_j b_j。

矩形使用全记录union bound，花费alpha_meter；不会把逐点95%区间当全记录95%域。z_min/z_max及角度范围再约束C_e，线性区间传播给D eta的非对称box。另算固定训练线路下的状态box，明确其只覆盖固定模型，不能用来拒绝整个未知参数候选族。

## 6. 条件性拒绝包络与停止门槛

允许余项delta_t属于上节box，固定Omega_bar的QP为

B_delta=min_{theta in P, delta_t in box_t} sum_t ||Omega_bar,t^-1/2(a_t(theta)−delta_t)||²。

二维delta可逐时作精确box投影消去。剩余共享系数的目标凸；保存有效下界L_delta而非用局部AC损失作下界。

真候选及真theta下，a_t=delta_t+xi_t+zeta_t；xi为线性高斯噪声，zeta=D(epsilon_v²−sigma_v²)。白化xi协方差不大于I，因而其范数由sqrt(chi_square_{2N,1−alpha_L})控制。对zeta，用全部原始epsilon_v的同时正态区间构box，枚举每时段8个box顶点求范数上界B_Q。这只是三维盒的解析审计，不枚举EV计划。

当真余项在box内时，L_delta <= B_delta <= ||W(xi+zeta)||²。三角不等式与union bound给条件性规则：

sqrt(L_delta) > sqrt(chi_square_{2N,1−alpha_L})+B_Q。

其真候选误拒上界为alpha_meter+alpha_Q+alpha_L。预定alpha=.05/.01，观测收紧版分配(.2,.2,.6)alpha。代码的历史字段alpha_pq实际是包含P/Q/V的计量矩形预算alpha_meter；使用V收紧电流分母也已花费该预算。自由度2N不扣拟合参数，因为用真参数这个可行点控制最小损失；不依赖Wilks近似。全时段噪声独立、已知量测尺度、真实物理点在声明域内、候选覆盖及余项box有效都是前提。

本轮先核对余项与候选差异尺度、QP界和独立含噪排序。若不满足精度门槛，只报告包络审计并保留候选；不把排序变成拓扑正确概率。共同结构是保留候选的支持集交集，只在所声明候选类内成立。

余项box通常不以零为中心。比较Hx的原始差时，使用box相对零的最大范数；比较box半宽时，应先把各模型预测移到Hx+box_center，再比较中心差。把半宽直接与原Hx差比较会少算非零余项中心。本轮保存这一初版后处理错误并修正，未影响QP或任何拒绝，因为测试一直未激活。两种一致口径都是固定拟合点尺度检查，不是对整个参数族的分离证明。
