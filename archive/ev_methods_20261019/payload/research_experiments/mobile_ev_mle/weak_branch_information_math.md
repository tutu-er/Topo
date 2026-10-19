# 弱支路的完整 AC 信息上界

## 结论和适用范围

在以下明确条件下，对任一三终端弱 cherry，存在一个固定的 star 竞争解释，使 12 点历史加 8 点充电观测的完整计量/动作过程分布满足

\[
D_{\rm KL}(P_{\rm cherry}\Vert P_{\rm star})
\le 0.10715701486040206 < 0.11.
\]

这里比较的是完整非线性恒功率 AC 的终端总 P/Q、终端电压幅值和含网损的主表 P/Q；没有使用 LinDistFlow、局部 Taylor 代理或仅电压通道。该界对符合功率和电量限制的任意共同因果控制器成立，包括使用历史观测和独立随机化的自适应控制器。

该结论是**物理解释域内的一对过程分布难以区分**。构造出的 star 叶边可能超出 nominal POP 的逐边缩放范围，尽管仍在代码的完整物理拟合盒内。因此这个界不是 nominal 四拓扑混合总体的校准定理，不是某条历史的正确拓扑概率，也不说明已有 MILP 或拟合算法达到该界。

本文件为自推导。源码核对基于只读 `C:\Users\23761\.codex\worktrees\36a1\Topo`；实际入口为 `research_experiments/mobile_ev_mle/adaptive_milp_study.py`。该文件树没有 `adaptive_independent_study.py`，故未把未定位文件当作证据。数值核验的源码 SHA-256 见 `weak_branch_information_validation.json`。

## 1. 源码模型与精确常数

`simulation.generate` 在 n=3 nominal 情形给出

\[
\begin{aligned}
p_j^0&=.12+c+e^p_j,& |c|&\le .012,& |e^p_j|&\le .004,\\
q_j^0&=.04+.25c+e^q_j,&&& |e^q_j|&\le .004,\\
p_s&\in[.065,.135],&q_s&\in[.02,.05].
\end{aligned}
\]

弱共用边 nominal 为 \(z_s=.00003+\mathrm i .00002\)，root stem 为 \(.005+\mathrm i .0035\)，终端叶边为 \((.028+.003j)+\mathrm i(.011+.002j)\)。`adaptive_milp_study.POPULATION` 对每条边 R、X 独立乘以 [.8,1.2] 内的缩放。由精确有理数平方比较可统一取

\[
\begin{gathered}
|S_j^0|\le M_b=.145,\qquad
|S_i^0-S_j^0|\le d_b=.012,\qquad |S_s|\le .15,\\
|z_j|\le Z_\ell=.045,\qquad
|z_i-z_j|\le\Delta Z_\ell=.022,\\
|z_0|\le Z_0=.008,\qquad |z_s|\le Z_s=.000044.
\end{gathered}
\]

例如 \(.136^2+.047^2<.145^2\)，两端共用背景项相消后 \(.008^2+.008^2<.012^2\)。叶边差异可以用全域 R 极值差 .0184 和 X 极值差 .0092 保守控制，其平方和小于 .022²。所有平方比较、正裕量及后续 KL 运算均使用 `Fraction` 精确计算；小数只用于展示。

EV 条件为一个设备、仅充电、单位追踪增益、纯 P、每时刻最多一端非零。令 \(u_t\) 为该时刻总 EV 有功，pu 基准 20 kW，则

\[
0\le u_t\le .3,
\qquad\sum_{t=1}^{20}u_t\le \frac{6\ \mathrm{kWh}}{20\ \mathrm{kW}\times .25\ \mathrm{h}}=1.2,
\qquad\sum_tu_t^2\le .3\sum_tu_t\le .36.
\]

前 12 点 \(u_t=0\)，后 8 点计量；6 kWh 是 grid energy，不能再乘充电效率或把行驶能耗加到这个 grid budget 上。行驶、接入和 SOC 约束只进一步缩小允许控制集，所以不需要路线枚举。放电、额外 Q、多个同时充电的 EV、更多观测点、不同噪声或 stress 扰动均不在本定理中。

测量定义取自 `likelihood.ac_predict` 和 `simulation.model`：物理电压输出为幅值 \(|V_j|\)，主表为 \(S_{\rm master}=\overline{\sum_k I_k}\)（root 电压为实数 1），包含所有网损；独立计量噪声为 \(\sigma_V=.00012\)、\(\sigma_{\rm master,P}=\sigma_{\rm master,Q}=.001\)。3 个终端 P、3 个 Q、3 个 V、主表 P/Q 共 11 个随机通道。root 为已知 1，相同已校准偏置在模型间相消。

## 2. 固定的 cherry—star 构造

保留相同 root stem、相同 source node 1、相同 source 路径和每个终端的相同总 P/Q 路径。cherry 的 i、j 两端共用边为 \(z_s\)，其叶边为 \(z_i,z_j\)。star 删除这条共用边，并固定设置

\[
z_i^{\star}=z_i+2z_s,\qquad z_j^{\star}=z_j+2z_s;
\]

其他叶边和 root stem 不变。此转换在实验开始前确定，不依控制动作变化。

用 \(B\operatorname{diag}(z)B^T\) 构造阻抗矩阵，删除零注入内部 junction 对应的坐标，保留 source1 和三个终端的四维系统。令 \(d=e_i-e_j\)，其中 source 坐标为 0，则精确有

\[
Z_{\star}-Z_{\rm cherry}=z_sdd^T.
\tag{1}
\]

cherry 的 pair 子块除共同 root stem 外为
\(\begin{bmatrix}z_i+z_s&z_s\\z_s&z_j+z_s\end{bmatrix}\)，star 为 \(\operatorname{diag}(z_i+2z_s,z_j+2z_s)\)，所以 (1) 的符号和系数都固定。source 行差为零，但 source 电压和主表响应仍可能间接改变，不能假定它们完全相同。

无注入 junction 的删除在本模型中是精确消元；其电压按原支路关系重建。`ac_general_v9.RadialCandidate` 的 support/B、`likelihood.ac_predict` 的独立 R/X AC 方程与这里相同；并未调用 v9 中已知固定 X/R 的 outer-LP 来支持独立 R/X 结论。

## 3. 两个系统高电压解的存在与唯一性

对任意时刻，两系统的 reduced AC 均为

\[
V=F_Z(V)=\boldsymbol1-Z\overline{S/V},
\tag{2}
\]

除法为逐坐标。最多一个终端充电时

\[
\sum_k|S_k|\le M_{\rm tot}=.15+3(.145)+.3=.885,
\qquad \max_j|S_j|\le M_{\max}=.445.
\]

star 叶边模长不超过 \(Z_\ell+2Z_s=.045088\)。两模型所有 reduced 行的加权模长和都不超过

\[
C=Z_0M_{\rm tot}+(Z_\ell+2Z_s)M_{\max}
=.02714416.
\tag{3}
\]

cherry 终端行的弱边贡献是 \(Z_s(|S_i|+|S_j|)\)，可由 \(2Z_sM_{\max}\) 控制，故同一个 C 也覆盖 cherry。source 行及原完整矩阵的零注入 junction 行更小。

取 \(\mathcal B=\{V:\|V-\boldsymbol1\|_\infty\le .1\}\)。在这个复向量球内 \(\min_k|V_k|\ge .9\)，于是

\[
\|F_Z(V)-\boldsymbol1\|_\infty\le C/.9
=.03016017777777778<.1,
\]

\[
\|F_Z(V)-F_Z(W)\|_\infty
\le K\|V-W\|_\infty,
\qquad K=C/.9^2=.03351130864197531<1.
\]

Banach 不动点定理给出两系统在球内的 AC 解。并且任意两个 AC 解只要所有 loaded 坐标幅值均不小于 .9，就由同一个倒数差恒等式满足该收缩不等式，所以两解必相同。这不排除低电压的其他平衡点。重建的内部 junction 亦在半径 .1 球内，因而原完整 AC 解存在且满足 [.9,1.1] 的电压域。

## 4. 电流对比与非线性电压差

使用 cherry 的电流 \(I_k=\overline{S_k/V_k}\)。当 EV 位于 pair 中时，把未充电的一端记为 j；EV 在 pair 外或未充电时任选 j。这样 \(|S_j|\le M_b\)，且 \(|S_i-S_j|\le d_b+u_t\)。令 \(L=|I_i-I_j|\)，则

\[
L\le \frac{d_b+u_t}{.9}
 +\frac{M_b}{.9^2}|V_i-V_j|.
\]

共用 stem 与共用弱边在 pair 电压差中精确相消，

\[
V_i-V_j=-z_iI_i+z_jI_j,
\qquad
|V_i-V_j|\le Z_\ell L+\Delta Z_\ell M_b/.9.
\]

因为 \(1-M_bZ_\ell/.9^2=3571/3600>0\)，解此标量不等式得到

\[
L\le L_0+L_1u_t,
\quad L_0=\frac{203651}{14462550}\approx .01408126506044923,
\quad L_1=\frac{4000}{3571}\approx1.1201344161299356.
\tag{4}
\]

把 star 方程在 cherry 解处相减，(1) 给出直接扰动
\((Z_\star-Z_{\rm cherry})I=z_sd(I_i-I_j)\)。用 star 的收缩常数 K，

\[
\delta_t:=\|V_\star-V_{\rm cherry}\|_\infty
\le\frac{Z_s(L_0+L_1u_t)}{1-K}=a+bu_t,
\tag{5}
\]

\[
a=\frac{2240161}{3494472755800}
\approx6.410583674695593\times10^{-7},
\qquad
b=\frac{891000}{17472363779}
\approx5.099481737387423\times10^{-5}.
\]

终端幅值差满足 \(||V_{\star,j}|-|V_{{\rm cherry},j}||\le\delta_t\)。主表不能略去，但由相同 S 及 root=1 有

\[
|\Delta S_{\rm master}|
\le\sum_k\frac{|S_k|}{|V_{\star,k}||V_{{\rm cherry},k}|}\delta_t
\le\frac{M_{\rm tot}}{.9^2}\delta_t.
\tag{6}
\]

## 5. 完整 Gaussian 信息界与检验含义

固定相同外生路径、参数和一个动作后，终端 P/Q 均值完全相同。Gaussian 协方差也相同。每时刻 KL 因而满足

\[
\mathrm{KL}_t\le\frac12\left[
\frac{3}{\sigma_V^2}
+\frac{(M_{\rm tot}/.9^2)^2}{\sigma_{\rm master}^2}\right](a+bu_t)^2.
\tag{7}
\]

主表 P/Q 的贡献是 \((\Delta P^2+\Delta Q^2)/\sigma_{\rm master}^2=|\Delta S_{\rm master}|^2/\sigma_{\rm master}^2\)，此处不能额外乘 2。式 (7) 的电压项使用幅值噪声 .00012，不能改成平方电压噪声或把 \(Y_{\rm drop}\) 再计一次。

20 点的路径电量界给出

\[
\sum_t(a+bu_t)^2\le20a^2+2ab(1.2)+b^2(.36),
\]

\[
D_{\rm KL}\le
\frac{3668918839949454447125}{34238718246575913269706}
<\frac{11}{100}.
\tag{8}
\]

由 Pinsker 不等式，

\[
\operatorname{TV}(P_{\rm cherry},P_{\rm star})
\le\sqrt{.11/2}=.2345207879911715.
\]

对这对过程做等先验二元检验，任何检验的平均错误率至少
\((1-\operatorname{TV})/2\ge .38273960600441426\)。若检验对包含某个 null 的物理解释域统一具有 size .05，则在这对中的相应 alternative 上 power 至多 \(.05+\operatorname{TV}\le .2845207879911715\)。这些是所构造过程对的检验限制；uniform-size null 域必须含该构造 star 过程。本轮实际门控只对 nominal mixture 作边际校准，并不具备整个物理域统一 size .05 的已验证前提，不能套用该功效界。它们也不是某次选择正确的概率或 nominal mixture 的平均准确率。

### 总变差与检验界的直接推导

定义 TV(P,Q)=sup_A|P(A)−Q(A)|。将完整读数/动作映射为事件指标 1_A，由 log-sum 不等式得到 KL(P||Q)≥KL(Bern(P(A))||Bern(Q(A)))。固定 q=Q(A)，二元 KL 对 p 的二阶导数为 1/[p(1−p)]≥4，在 p=q 的值与一阶导数都为0，积分两次给 KL(Bern(p)||Bern(q))≥2(p−q)²。取达到总变差的事件或取上确界，就得到 TV≤√(KL/2)；边界概率为0或1时按极限处理。

令 A 是拒绝 star 的事件。等先验二元平均错误为 [P_star(A)+P_cherry(A^c)]/2=[1−(P_cherry(A)−P_star(A))]/2≥(1−TV)/2。若 P_star(A)≤α，则 P_cherry(A)≤α+TV。随机化检验也成立：增加共同独立随机数，或对0到1的拒绝函数用层集积分。这个推导没有把设计分或拟合损失转成拓扑正确概率。

## 6. 自适应控制和共享 latent process

令 L 表示实验开始前确定的 cherry 阻抗参数以及完整外生背景负载/source 路径，以 cherry 参数为基准定义共同 \(\mu(dL)\)。star 参数不是独立再抽，而是同一 L 中 cherry 参数的固定转换（pushforward）。source 路径在两模型中相同；不允许竞争 source 在看到控制后任意改写路径。独立随机化 R 的分布也相同，且与 L、计量噪声独立。

共同控制器的动作核为
\(\pi_t(da_t\mid Y_{<t},a_{<t},R)\)。在比较同一个条件历史时，两模型的动作核完全相同，动作项对 KL chain rule 的贡献为零。条件于 L、R 和当前共同过去，逐时新计量具有相同协方差的 Gaussian 形式；自适应或 latent 积分后的整条轨迹通常不再是联合 Gaussian。式 (7) 对该条件历史下所执行的同一动作逐点成立。因此

\[
D_{\rm KL}(P^{L,R,\pi}_{\rm cherry}\Vert P^{L,R,\pi}_{\rm star})
\le \frac12 W\,\mathbb E_{\rm cherry}
\left[\sum_t(a+bu_t)^2\right]\le .10715701486040206,
\]

其中 W 是式 (7) 的括号。功率与电网电量限制必须逐路径成立。控制器在两模型共同的历史/动作信息上定义，设备可行集只依赖可观测设备状态及公开约束；不能依赖未知真拓扑、真实源或模型特有的隐状态/abort。这些条件下，自适应性不改变最后一个界。共同控制器并不要求两个模型各自模拟时得到相同动作序列；需要的是在**相同已观测历史**下采用相同动作核。

将 L、R 积分掉，数据处理不等式使实际观测/动作轨迹的 marginal KL 不超过以上 joint KL。这里揭示 latent path 只是为了得到更强观察者的信息上界，并不把未来 latent truth 提供给控制器。如果两模型采用不同 latent law、不同 bias/noise、动作依赖 truth 或非因果未来数据，就不能直接沿用这段证明。

## 7. 验证、复现与未覆盖项

`weak_branch_information.py` 保存精确有理数常数和证明裕量；`test_weak_branch_information.py` 检查 nominal caps、球不变性、收缩、current inequality 的代数解、电量二阶界、完整 Gaussian 权重、所有三种 pair 的 reduced Z 恒等式，以及 reduced/full AC 和主表一致性。测试还明确构造一个 star 叶边越出 nominal POP 边箱、却留在物理拟合盒的案例，防止误报范围。

`weak_branch_ac_audit.py` 使用项目实际 `ac_predict` 和 `RadialCandidate`，并与独立四维不动点计算核对；覆盖三种 pair、六组 R/X 缩放以及四种固定控制序列，共 72 条 20 点路径、1440 个时刻。控制序列只用于本信息界的放宽域；changing_sites 未被声称满足移动/接入限制。没有运行拟合、策略选择、校准或完整路线枚举。

本次数值验证全部通过，最大 actual Gaussian KL 为 .035236562306136913，最大电压差/解析界比值为 .8681351169193876；reduced/full 电压最大误差为 \(2.93\times10^{-15}\)，最大功率平衡残差为 \(1.67\times10^{-15}\)。这些数字是有限样本核查；uniform 界的依据是前述有理数不等式和完整 AC 推导。

复现命令（在研究工作树根目录内）：

```powershell
& 'D:\apps\miniconda3\envs\Topo\python.exe' -m pytest research_experiments/mobile_ev_mle/test_weak_branch_information.py -q -p no:cacheprovider
& 'D:\apps\miniconda3\envs\Topo\python.exe' -m research_experiments.mobile_ev_mle.weak_branch_ac_audit --output research_experiments/mobile_ev_mle/outputs/weak_branch_information_replay.json
```

两模型的 root/source/叶边与终端负载均在 `likelihood.fit_candidate` 默认物理盒内，电压存在性已经证明。该构造不保证落入某次 history-centered 局部 trust box，也不把局部 refit、soft-history proxy、候选 bank 覆盖或 solver gap 当作统计证书。

该审计没有修改冻结主策略或 calibration 文件；合入后的来源哈希与测试另保存。下一步若研究 nominal mixture 的识别错误或 gate calibration，须另以其实际 star 参数分布和完整评估协议为对象，不能用本构造替代。
