# 本次后验DR接口的推导与证书范围

只讨论已冻结的有限候选、参数和历史尺度。Y为根平方电压减终端平方电压，H为P/Q到该响应的拟合矩阵；H不直接等于线路阻抗。条件于目标t，工作模型为`Y=H_t P+e`，`e|P ~ N(0,sigma² I)`，sigma²为报价前冻结的历史共同估计。

## 从后验事件到一个总失败界

令`L_j^h=log(w_t^h/w_j^h)`，新日的对数赔率增量为

\[
\Delta L_j=\frac{\|(H_t-H_j)P\|^2+2e^\top(H_t-H_j)P}{2\sigma^2}.
\]

历史权重在这里是已知常数，历史证据不再重复生成。记`B=(1-beta)/beta`、`R=sum_j exp(-L_j^h-Delta L_j)`，目标后验低于beta等价于`R>B`。对固定公共`0<lambda<1`，Markov不等式及非负分数幂的次可加性给出

\[
\Pr_t(w_t^{new}<\beta)
\le B^{-\lambda}\mathbb E R^\lambda
\le\sum_{j\ne t}c_j\exp[-T_j(u)],
\]
\[
c_j=B^{-\lambda}\exp(-\lambda L_j^h),\qquad
T_j(u)=-\log\mathbb E_t\exp(-\lambda\Delta L_j).
\]

不要求竞争事件独立。给定功率时，高斯指数矩为

\[
\mathbb E_e e^{-\lambda\Delta L_j}\mid P
=\exp\!\left[-\frac{\lambda(1-\lambda)}2
\frac{\|(H_t-H_j)P\|^2}{\sigma^2}\right].
\]

所以T_j>=0，对全部控制都成立。本次lambda=1/2，对应系数1/8。随机功率的指数矩沿用原谱／稠密高斯计算，没有数值积分或真实电压预测。若换用另一残差方差或相关误差法则，必须重新核查合法lambda和T_j的非负性。

## 原高斯功率法则下的同一个二次形式

令`z=(H_t-H_j)P/sigma ~ N(m(u),C)`、`k=lambda(1-lambda)/2`，且C不随控制改变。配方得到

\[
G=I+2kC,\qquad
T_j(u)=k\,m(u)^\top G^{-1}m(u)+\tfrac12\log\det G.
\]

本次保留原5%通道峰值标准差、同节点P/Q共享、节点独立及时间AR(.7)。C具有时间／节点Kronecker结构。`prospective.spectral_moment`复用原谱计算的分解；`pricing.GaussianGate`把同一个表达式分成节点旋转与时间旋转辅助变量，避免展开稠密资源×时间矩阵。两者已与独立稠密表达式及受控均值对照。

## 主要竞争者、统一尾部界与原设备图

对主要集合A，`R_tail=sum_{j outside A} c_j`是所有控制统一有效的尾部上界。若`R_tail<1-eta`，固定正额度epsilon_j，总和不超过`1-eta-R_tail`，并加入

\[
T_j(u)\ge\log(c_j/\epsilon_j),\quad j\in A,
\]

即可保证目标后验达到beta的条件概率至少eta。首版均分剩余额度，不把历史原始尾部概率代替R_tail。原设备功率、能量、效率、截止和小时请求映射保持，多个约束共用一个设备图，作用资源取并集。

该固定额度条件是原机会约束的充分条件。它的最优费用可以高于原问题；其求解器下界不能直接充当原问题下界。费用非负且零动作在原域可行时，零费充分证书能证明原费用问题最优值0。

## 容量检查的两种不同结论

逐时段由原设备交付上下界得到归一化均值各坐标的上下界，继而得到`||m(u)||² <= M`。由于G逆的算子范数不超过1，

\[
T_j(u)\le kM+\tfrac12\log\det G=:U_j.
\]

若某固定额度门槛大于U_j，该充分约束不可行。如果`sum_{j in A} c_j exp(-U_j)>1-eta`，连完整指数概率界也无法在该容量盒内压到要求；这仍不是实际失败概率的下界，不能据此证明原机会约束不可行。

另有可选的原机会约束必要检查。对高斯z，令Q=||z||²，则

\[
\mathbb E L_j=L_j^h+(\|m\|^2+\operatorname{tr}C)/2,
\quad
\operatorname{Var}L_j=\|m\|^2+\operatorname{tr}C+
\tfrac12\operatorname{tr}(C^2)+m^\top Cm.
\]

用M及C最大特征值分别上界均值和方差。仅当均值上界mu_bar小于`c=log(beta/(1-beta))`时，单侧切比雪夫界给出

\[
\Pr_t(L_j\ge c)\le
\frac{v_{bar}}{v_{bar}+(c-\mu_{bar})^2}.
\]

整库达标必然要求每个L_j>=c，因此该上界若低于eta，才能证明声明工作法则下的原机会约束在更大容量盒内也不可行。本批没有触发这一更强结论。

## 实际执行的边界

真族在库中，只证明结构资格；它不证明冻结参数恰为实际AC数据的条件均值。原根量测噪声、P/Q误差、参数误差、相关设备响应和饱和均留在独立实际模拟中。模型内的概率保证与实际AC频率分开记录。

实际交付沿用原`Devices.project(nominal_mean*draw)`。原合同费用按冻结名义计划记账；`fees(actual_delivery)`是执行损耗代理，不作为额外支付。
