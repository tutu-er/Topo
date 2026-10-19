# 方向1第二阶段：连续AC的有界局部软历史QP

此阶段使用原三末端历史计量及公开动作库。没有访问真拓扑、真线路、真未来路径或未来计量噪声，也没有重跑原80试验。有限atom距离不能替代较宽参数域的辨识检查；本阶段在连续历史拟合点附近让两边线路分别改变，仍只实现局部代理，不实现硬历史QCQP/MILP。

## 1. 条件模型与信息时序

候选g、h均有根0和同一个实际源接口1。beta_k=(r_k,x_k)属于R^(2m_k)，历史H共享一套线路；历史逐时状态w_kt=(p_kt,q_kt,s_kt^P,s_kt^Q)属于R^(2n+2)。总P/Q包含历史EV，不再重复加。完整平衡径向AC及计量均值复用likelihood.ac_predict，源始终通过接口1作用。

每时刻全部计量为末端P、Q、|V|和总表P/Q，维数d=3n+2；总表为根复功率V_0*conj(I_c)，包括同点源和损耗。根电压与固定偏差已独立校准；已知Sigma正定，本小试为对角、跨时独立Gaussian。代码也可加入已有独立源计量通道，但不允许删除总表锚点。当前不拟合自由bias。

先仅用历史y_H分别调用fit_candidate，得到已发现局部解(beta_hat_k,w_hat_kH)。成功状态不证明全域MLE；ftol终止也不保证严格驻点。未来公共末端基线来自同一组校准读数，不取某个模型的真状态：

\[
\bar p_i=|H|^{-1}\sum_{t\in H}(y^P_{it}-b_i^P),\quad
\bar q_i=|H|^{-1}\sum_{t\in H}(y^Q_{it}-b_i^Q).
\]

所有模型未来都使用p_t=bar{p}+u_t^P、q_t=bar{q}+u_t^Q。动作只加入一次，当前u^Q=0。未来根电压默认已校准历史根电压的均值，也可给公开已知序列。模型k的源预测为该模型历史估计的平均值bar{s}_k；位置都固定为接口1。源预测本身不是无偏或覆盖保证；它是明确的条件设计假设，后续需检查预测误差。

两种源域：fixed令delta{s}_kt=0；local_refit令每模型每未来时刻P/Q分别在[-.02,.02]pu内调整，并与绝对源物理界[-.5,.5]相交。后者是声明的局部预测域，不是任意源轨迹的稳健保证；两边可以选不同源变化。未来总P/Q没有独立自由拟合，避免选择baseline=-EV令所有动作退化。

## 2. 完整历史导数和软代价

令W_H^T W_H=Sigma_H^(-1)。在模型k的历史拟合点处，计算白化均值导数

\[
J_{\beta,k}=\partial(W_H\mu_{k,H})/\partial\beta_k,
\quad J_{w,k}=\partial(W_H\mu_{k,H})/\partial w_{k,H}.
\]

二者包括所有P/Q/V/总表通道。末端P/Q对beta导数为零，对其真实状态导数为1/sigma；源不是各电压通道独立误差，其影响由同点AC导数产生。J_w的状态列按时刻排列。

消去线性化历史状态的补偿能力：

\[
H_k=J_{\beta,k}^{\top}(I-J_{w,k}J_{w,k}^{+})J_{\beta,k}\succeq0.
\]

证明：对指定delta{beta}，min_delta{w}||J_beta delta{beta}+J_w delta{w}||^2等于||(I-J_w J_w^+)J_beta delta{beta}||^2。秩亏时用Moore--Penrose逆，不能用非秩揭示QR的列数假装满秩。

实现对J_w做秩揭示SVD，投影J_beta，再QR压缩为factor F_k，满足F_k^T F_k=H_k。SVD/rank仅是投影及诊断，不用全方向最小奇异值给动作打分。

H_k是Gauss--Newton局部历史代价。只有在正规内点驻点并采用残差线性化时，才能把它解释为历史profile损失增加的近似；非零残差的实际Hessian还有均值二阶项。历史状态补偿在该投影中不加活动约束，必须检查其原模型回代，不能说这是精确历史likelihood支持域。

## 3. 未来均值差与真正的有界QP

对每个公开动作u，在两模型的beta_hat和各自bar{s}_k处求解原AC，得到白化差

\[
d(u)=W_N[\mu_g(\hat\beta_g,\bar p+u^P,\bar q+u^Q,\bar s_g)
-\mu_h(\hat\beta_h,\bar p+u^P,\bar q+u^Q,\bar s_h)].
\]

在这个动作的预测点计算未来线路导数A_g,A_h和逐时同点源导数G_g,G_h。固定公共总P/Q，因此没有可独立抵消EV的未来末端状态列。线性化残差为

\[
r_{\rm lin}=d+A_g\delta\beta_g-A_h\delta\beta_h
+G_g\delta s_g-G_h\delta s_h.
\]

令z堆叠上述允许的变量，A包含相应带符号导数，F是历史factor的块对角扩展，源列不另加历史惩罚，则

\[
q_{gh}(u)=\min_{l\le z\le v}
\{\|d+Az\|^2+\|Fz\|^2\}.
\]

线路step逐分量受物理R/X盒和局部信赖域交集约束：

\[
l_{\beta,k}=\max(\beta_{\min}-\hat\beta_k,-.30\hat\beta_k),
\quad v_{\beta,k}=\min(\beta_{\max}-\hat\beta_k,.30\hat\beta_k).
\]

R/X物理范围均为[1e-6,.1]pu。源step如前述；fixed版不引入零宽度源变量。只有box线性约束，目标为平方范数，故这是凸QP，而不是硬历史QCQP。写成C=[A;F]、b=[-d;0]的有界线性最小二乘，用scipy.lsq_linear的BVLS/精确线性子求解器。数值坐标缩放改变条件数，不改变物理目标或box。

若源版放大的box包含所有固定源step，且两版本都在相同初始预测处构造线性模型，则q_local_refit<=q_fixed。这是代理内层的集合包含性质；若某版原AC回代失败，要保留失败，不能用删掉失败动作后的排名证明此性质。

对两个以上候选取最小pair分数；对有效充电动作取最大分数。无EV保持相同观测长度，作为分数/实际结果对照。选择完全在历史及公开预测一侧完成。

## 4. QP最优性、原AC回代与误差

记录BVLS状态、迭代数、scaled optimality和active bounds。对可行z的凸二次F_QP(z)=||Cz-b||^2，梯度g=2C^T(Cz-b)。切平面满足F_QP(v)>=F_QP(z)+g^T(v-z)。在box内逐分量选角点v_j=l_j（g_j>=0）或upper_j（g_j<0），即得可计算的切平面下界；与非负下界0合用记录numeric QP gap。浮点计算不是严格区间证书，此界仅属于这个线性QP。

原AC未来回代使用beta_hat+delta_beta、同一公共baseline+EV及bar{s}+delta{s}，计算实际均值差d_AC，记录

\[
Q_{\rm AC}=\|d_{\rm AC}\|^2,\quad
E_T=\|d_{\rm AC}-(d+Az)\|,
\quad E_{T,rel}=E_T/\max(\|d_{\rm AC}\|,\|d\|,1).
\]

同时保存代理未来残差、历史惩罚、原AC方程/功率平衡和所有节点电压。relative分母包含初始差，因此必须结合原始E_T及proxy/actual距离判断；它不是近零分离误差界。

另用最小范数历史补偿delta_w=-J_w^+J_beta delta_beta回代历史AC，保存实际历史loss及其相对原局部解的增量。允许该增量为负，并明确这是一个候选状态的诊断值。历史状态或未来状态若超出物理域、原AC不收敛或任一pair求解失败，该动作标为失败，不丢候选、不增加分数。

正的q不证明非线性距离正，也不证明拓扑正确。即使线性QP内层被精确求解，它仍使用局部历史二次代价和未来Taylor展开；q没有自动成为原AC下确界的上下界。回代得到实际可行候选值，也没有求遍连续AC参数、历史状态或全部平衡点。该阶段不应用有限atom的拒绝阈值。

## 5. 接口和最小验证

local_symmetric_scores(candidates,history,model,actions,historical_fits=...,source_mode='fixed'或'local_refit')返回顶层actions[name]、selected和历史诊断。每action含success/score/pairwise/failures；每pair保存上述优化、步长和回代值。也可用source_modes一次计算两个variant；此时结果在variants[mode]中。若任何历史候选失败，顶层actions为空并返回historical_fit_failure。

调用者已有历史fits时，不重复拟合。根实验将仅用原66001..3、67001..3历史读数构造这些fits；连续历史拟合的初值可来自历史数据挑出的有限atom，不能来自实际未来响应。选好动作后再生成独立未来噪声，调用完整连续AC拟合检查实际辨识结果。

独立检查包括：bounded标量闭式解与活动边界；无约束解落在box内的两参数闭式解；源域包含固定源时最小值单调；坐标缩放不变；秩亏历史补偿的独立least-squares核对；全P/Q/V/master线路和逐时源的中央差分；已知偏差下公共baseline；历史-only端到端预测、原AC回代、失败保留。没有把测试写成单纯镜像实现。

固定均值toy中，两模型未来为a+u和b-u、历史代价为a^2+b^2，代理最小值为4u^2/3；u=0一阶梯度零但有二次区分。受box z in[-.25,.5]的min(2+z)^2+z^2在z=-.25取得3.125。这些闭式值独立验证QP核心，不代替AC算例。

## 6. 来源与下一阶段门槛

依据当前likelihood.py的完整AC、计量模型和已验证_response_derivatives，以及既有symmetric_math.md的软历史推导。本稿的AC局部扩展与bounded-QP实现由项目独立推导；本轮没有新增外部文献结论或声称复现论文实验。

先检验原三末端局部计算的最优性、回代误差和实际新噪声结果。若30%信赖域下Taylor误差大，应缩小/顺序更新并保留原结果；不能把调过的代理算作同一冻结方法。若放开同点源令信息消失，或新动作在完整观测拟合中仍弱于恒充，应按当前域解释失败。只有明确预测域、历史局部近似质量和独立评价之后才讨论硬QCQP或扩网；本阶段不要求EV必胜。
