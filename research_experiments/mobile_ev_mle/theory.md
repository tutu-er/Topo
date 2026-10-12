# AC 计量似然与 EV 模型判别：当前理论边界

日期：2026-10-12。研究代码位于 `research_experiments/mobile_ev_mle/`，生产 `rnj_wzzt_core` 默认流程未改。

## 1. 对象、符号与最小概率模型

候选径向树为 $h\in\mathcal H$，线路为 $e=1,\ldots,m_h$，被测末端为 $i=1,\ldots,n$，时刻为 $t\in H\cup N$。所有潮流量使用 pu；EV 可执行性以 kW/kWh 表达后显式换算。负荷消耗为正。线路 $z_e=r_e+\mathrm j x_e$，$r_e,x_e>0$，历史和新增数据共享同一组 $r_e,x_e$，不固定 X/R。

逐时未知状态记作 $w_t=(p_t,q_t,s_t^P,s_t^Q,v_{0t})$。$p_t,q_t$ 是末端**总**负荷，已经包含 EV。共同源 $s_t^P,s_t^Q$ 是额外未计入末端读数的净消耗，正值表示消耗、负值表示注入；每个时刻可不同，但所有时刻通过同一个已声明的共同接入点 $j_s$ 进入 AC 方程。首个算例令 $j_s$ 为公共干线下端；不同候选中的节点编号不是物理身份，必须对应同一个公共接口。

设 $F_h(r,x,w_t)=0$ 为完整平衡单相径向 AC 方程。也可由经残差核验的潮流函数消去复电压和线路电流。根相角固定为零，消除相角 gauge；已校准根电压首先视为可信已知。观测向量

\[
y_t=(\widehat p_t,\widehat q_t,\widehat v_t,\widehat P_{0t},\widehat Q_{0t})
      =m_h(r,x,w_t)+b+\varepsilon_t,
\qquad \varepsilon_t\sim N(0,\Sigma_t).
\]

$m_h$ 包含末端 P/Q、电压幅值和总表 P/Q 预测。$\Sigma_t\succ0$ 是先验校准、固定的随机计量协方差，首个算例使用独立通道和跨时独立 Gaussian 误差。仪表偏差 $b$ 是跨时固定的系统误差，不能当作每时刻新抽样的随机噪声。首阶段将已校准的非零偏差作为已知 correction，**不同时自由估计所有 P/Q/V 偏差、根电压和共同源**。可选 `fit_bias=True` 入口只在提供独立 `Calibration` 偏差读数时联合估计固定P/Q/V偏差；主试验仍使用已校准 correction。估计的 $b$ 必须与线路参数共同跨历史、新数据共享，并给出校准锚点、参数域和可识别性诊断。

共同源是条件确定的 nuisance sequence，没有默认的时间平滑先验或固定 Q/P。它与随机计量误差的统计含义不同：误差有声明分布，源只有声明物理位置和参数域。逐时 15 分钟数据被当作一个稳态 AC 点，是算例假设；一般不能将 AC 非线性作用于均值等同于瞬时潮流均值。

不要同时把原始电压和由同一电压计算的压降当作独立观测。若只使用平方压降 $\widehat v_0^2-\widehat v_i^2$，噪声一般非 Gaussian 且含共同根误差，需要改变似然。本研究先使用电压幅值。

## 2. 先辨识哪些参数

### 2.1 源、仪表偏差和线路参数会混淆

在一阶公共压降模型

\[
d_{it}=a_{h,it}(r,x;p,q)+c_t+b_i^V,
\]

令 $b_i^{V\prime}=b_i^V+\delta$，$c_t'=c_t-\delta$，所有预测不变。因此仅从该模型的电压观测无法确定公共偏差和共同源水平。这是该简化模型中的严格等价，不是声称完整非线性 AC 在任意观测配置中存在相同精确 gauge。

对公共干线，一阶源响应是 $2(r_0s_t^P+x_0s_t^Q)\mathbf1$。任给 $\gamma_t$，取 $s_t^{P\prime}=s_t^P+x_0\gamma_t$，$s_t^{Q\prime}=s_t^Q-r_0\gamma_t$，该响应不变。在没有独立总表信息时，仅一组共同压降不能分别确定两个逐时源分量。允许同时改变 $r_0,x_0$ 又增加混淆；AC 损耗可能提供很弱的额外信息，不能据此一开始放开全部参数。

首阶段固定已校准偏差和根电压，纳入根总表 P/Q。总表预测满足

\[
P_{0t}=\sum_i p_{it}+s_t^P+\sum_e r_e|I_{et}|^2,
\quad Q_{0t}=\sum_i q_{it}+s_t^Q+\sum_e x_e|I_{et}|^2.
\]

它为源的两个分量提供独立锚点；线路仍需足够多的空间、时间和 P/Q 变化。总表不是完美源测量，损耗也需由同一候选线路解释。

### 2.2 局部诊断和模型判别分开

令白化残差对共享线路参数及逐时 nuisance 的 Jacobian 分别为 $J_\beta,J_w$。在内点、适当光滑与正则条件下，局部剖面信息为

\[
\mathcal I_{\mathrm{prof}}=J_\beta^\top(I-J_wJ_w^+)J_\beta.
\]

其秩和奇异值是**局部参数诊断**：秩亏说明一阶共享参数方向可被逐时状态吸收。数值满秩只支持该拟合点附近的一阶可估性，不证明全局唯一、候选覆盖或拓扑可区分。报告需同时给出 scaling、秩阈值、参数数、边界活动和多初值差异。最小奇异值可以诊断弱参数，不能替代针对竞争结构重拟合的设计准则。

## 3. 完整 AC 似然与局部拟合

对已知 Gaussian 协方差和偏差，固定拓扑的负二倍对数似然为

\[
-2\ell_{h,D}(\beta,w_D)=L_{h,D}(\beta,w_D)+C_D,
\quad L_{h,D}=\sum_{t\in D}\|y_t-b-m_h(\beta,w_t)\|_{\Sigma_t^{-1}}^2,
\]

\[
C_D=\sum_{t\in D}\{d_t\log(2\pi)+\log\det\Sigma_t\},
\quad F_h(\beta,w_t)=0.
\]

线路参数 $\beta=(r,x)$ 跨时共享，末端真实 P/Q 和共同源均按时刻拟合。末端 P/Q 的 Gaussian 残差使它们不能被任意改变而没有计量代价；这就是本阶段明确的 errors-in-variables 处理。若有偏差校准 uncertainty，可选入口纳入独立 calibration observation $c=b+\eta$ 的 Gaussian loss，且联合拟合时只计入一次：$L_{h,D}^{\mathrm{joint}}=L_{h,D}+\|c-b\|_{\Omega^{-1}}^2$，normalizer同时加入calibration项。不能用每时偏差追随噪声。

数学上 $L_{h,D}^*=\inf_{\beta,w_D}L_{h,D}$，$\ell_{h,D}^*=-(L_{h,D}^*+C_D)/2$。实现用有界 nonlinear least squares 多初值求最佳可行局部点，输出 $\widehat L_{h,D}$ 和 $\widehat\ell_{h,D}=-(\widehat L_{h,D}+C_D)/2$。它们是**最佳已发现局部拟合**；既没有全局下界，又没有证明涵盖所有 AC 解分支，不能命名为已认证全局 MLE。

潮流失败、超时、未收敛、参数贴边分别记录。失败不能把候选的似然设为零并声称统计拒绝。AC 残差合格也只是数值物理检查，不是全局优化证书。不同候选应使用相同观测通道、协方差、参数域与计算预算。新增数据必须重新拟合**同一组**线路参数，不能分别拟合历史和新增后相加。

## 4. Dette–Titoff 迁移：历史代价加未来差异

来源：Dette–Titoff (2009), *Optimal discrimination designs*, Annals of Statistics 37(4), 2056–2082, DOI [10.1214/08-AOS635](https://doi.org/10.1214/08-AOS635)，[开放电子重印全文](https://arxiv.org/pdf/0908.1912)。本次读取第 2 节（PDF 页 2–9），以及第 4.2 节（PDF 页 17–19），并视觉核对页 4、5、18。不是仅摘要依据；未声称逐页阅读整篇。

原文第 2 节式 (2.4) 固定参考回归函数，对竞争参数取最小后选择最大化差异的设计；第 4.2 节式 (4.1) 将平方差推广到分布 KL。原文还明确参考参数依赖和序贯/robust 处理的必要性。它没有直接证明本项目 AC、共同源、历史 penalty 或自适应候选的有限样本误拒绝保证。下列公式和证明是本项目在明确 Gaussian 假设下的迁移。

从观测历史得到参考候选 $r$、参数 $\widehat\beta_r$、未来基线/共同源 forecast；动作 $a$ 给出可执行地点和 P/Q 曲线，预测未来计量均值 $\widehat\mu_{r,t}(a)$。参考不使用真拓扑、未来负荷真值或未来噪声。future baseline/source forecast 错误会影响设计，初步实验应声明所用预测规则，不能把参考当成已知真模型。

历史剖面损失和代价定义为

\[
L_{h,H}(\beta)=\inf_{w_H}L_{h,H}(\beta,w_H),\qquad
R_h(\beta)=L_{h,H}(\beta)-L_{h,H}^*.
\]

针对参考 $r$ 与竞争 $h$，选择

\[
T_{rh}(a)=\inf_{\beta,w_N}\left\{
R_h(\beta)+\sum_{t\in N}
\|\widehat\mu_{r,t}(a)-m_h(\beta,w_t;a)\|_{\Sigma_t^{-1}}^2
\right\}.
\]

该内层同时重新拟合历史状态、历史和新增共享阻抗，以及新增逐时源 P/Q。对可拟合偏差同样共享。源不可在未来改为每测点独立输出。原始末端总 P/Q 用于实际联合拟合；设计阶段仅在基线 forecast 上施加一次 EV 曲线，不能在实际计量中再加一次 EV。

已有历史候选差异 $g_{rh}=L_{h,H}^*-L_{r,H}^*$ 可加入，得到

\[
g_{rh}+T_{rh}(a)=\inf_{\beta,w_H,w_N}
[L_{h,H}(\beta,w_H)+L^{\mathrm{forecast}}_{rh,N}(a;\beta,w_N)]-L_{r,H}^*.
\]

实现可以直接计算右侧，避免把局部 penalty 强行截断为非负。动作分数例如 $S(a)=\min_{h\ne r}[\widehat g_{rh}+\widehat T_{rh}(a)]$，只在预先列出的可执行动作中最大化。允许通过多个数据拟合参考和 source forecast 情景取 min/平均缓解参考依赖，但这需要实际实现与额外计算预算，不能由文献讨论自动宣称已 robust。

### 命题 1：Gaussian KL 的因子

若未来两模型对同一观测向量给出 $N(\mu_r,\Sigma)$、$N(\mu_h,\Sigma)$，协方差同一且已知，则

\[
2D_{\mathrm{KL}}(N(\mu_r,\Sigma)\|N(\mu_h,\Sigma))
=(\mu_r-\mu_h)^\top\Sigma^{-1}(\mu_r-\mu_h).
\]

证明：将 Gaussian log density 之差对参考分布取期望。均值为零的 centered 随机项抵消，两个 trace 项和 determinant 项因同协方差抵消，仅剩二分之一均值差二次型。独立时刻 KL 相加。不同或依参数变化的协方差必须保留 trace、log determinant 和维数项，不能继续只用平方差。

### 命题 2：历史 penalty 非负及计算例外

如果 $L_{h,H}^*$ 是声明域内的真实 infimum，则每个可行 $\beta$ 都满足 $R_h(\beta)\ge0$，因为全域 infimum 不大于固定参数的剖面值。若基准仅为局部 $\widehat L_{h,H}$，竞争重拟合找到更好的历史参数会使计算的 penalty 为负。这是基准优化不充分的诊断，不是负信息，也不能直接裁剪到零后当作精确设计。应以新参数重启历史拟合、更新基准并重算动作；若预算不足保留优化不稳定标记。内层局部最小结果可高于真实 infimum，可能夸大可区分性；多初值降低此风险而不消除它。

### 直觉

竞争结构可以改变线路来追随 EV 响应，但改变后还要解释已有历史。设计寻找这种共同解释最吃力的动作。若历史和未来分别拟合线路，历史代价消失；若未来源改成独立输出误差，结构差异也容易被吸收。若弱支路在允许的激励下仍能被竞争模型完整解释，低分/未决是正确的研究结果。

## 5. 可执行约束与公平对照

动作包括时隙地点 $j_t$、连接指示 $c_t$、功率 $u_t^P,u_t^Q$。仅可在有接入许可的地点和时段连接；行驶/插拔期间 $c_t=0$ 且充放电为零。地点变更需要至少 $\tau_{ij}$ 个时隙，且可达关系明确。

\[
|u_t^P|\le c_tP_j^{\max},\quad
(u_t^P)^2+(u_t^Q)^2\le c_t(S_j^{\max})^2,
\quad E_{t+1}=E_t+\Delta t(\eta_c[u_t^P]_+-[-u_t^P]_+/\eta_d)-E_t^{\mathrm{move}}.
\]

$E_{\min}\le E_t\le E_{\max}$，终端能量满足任务要求，电量对照统一 grid-side $\sum_t u_t^P\Delta t$ 或 battery-side 能量口径并声明，不能混用。初步单地点动作仍需检查从初始地点到该地点的 travel deadline。当前实现采用仅充电、固定 charger Q/P（默认0）的有限动作库，不提供独立 Q 优化或放电；上述一般约束用于说明扩展物理域。运行时声明初始地点是否已就位；`initial_site=None` 表示初始移动不计入本时段，不能同时声称已经计入出发移动。库内地点切换扣除 `travel_energy`，主算例每次切换0.01 pu-h=0.2 kWh电池能量。实际模型的 voltage/current 运行域与设备功率域分别检查；用局部拟合预测通过不等于现场安全证书。

首个实现的充电策略为grid energy=6 kWh、max power=6 kW、battery capacity=60 kWh、效率0.95、初始SOC0.4、SOC域[0.1,0.9]，时隙15分钟。初步对照：同长度无 EV 观测；同电量、固定地点恒充；固定地点调制；随机可行动作；数据驱动设计动作。所有策略使用相同观测长度和噪声口径；EV 策略使用相同电量、功率和接入预算。无 EV 对照的电量为零，只对齐观测时长。设计器只能读取历史及公开 forecast/约束；真拓扑用于生成数据和离线计分，不进入选择器。

## 6. 校准、候选漏真与失败分类

候选统计量可记作 $D_h=\widehat L_{h,H\cup N}-\min_k\widehat L_{k,H\cup N}$。这是有限库内相对 deviance，不是概率。非嵌套拓扑、边界阻抗、弱辨识、逐时 nuisance、候选选择和自适应 EV 不满足自动套用固定自由度 $\chi^2$ 的条件。

Parametric bootstrap 若使用：在候选 $h$ 的已拟合参数与 nuisance、已声明随机误差下生成历史；每个 replicate **重新运行被动候选生成、reference 拟合、EV 选择、未来生成、联合拟合和统计量**，使用无泄漏的未来 forecast。固定原来挑中的动作而只重抽未来噪声，不会校准完整自适应流程。bootstrap 的 threshold/p-value 是条件于 plug-in 参数、模型和优化预算的近似，不能转换为拓扑正确概率。源是逐时确定 nuisance 时须说明 replicate 固定/预测其序列的方法，不可无依据地给它随机 prior。

另一个可复现口径是预声明合成参数族下独立 calibration 和 evaluation seed cohort：前者计算阈值，后者报告真实结构误拒绝次数和排错能力。它检验该参数族/策略/噪声的**经验校准**，不是现场保证。sample 小时需保留分母与区间：即使 $0/M$ 次误拒绝，仍不能声称零风险；一侧 95% Binomial 上界为 $1-0.05^{1/M}$。对几个情景报告平均率还不能代表每个情景或全参数域都有同一误拒绝率。

候选漏真须另外标记。仅有一个错误候选胜出仍可能有很小相对 deviance；需要 absolute fit-loss adequacy（同样通过声明流程校准）及残差渠道检查，仍可能漏检与真结构近似等价的库外解释。不把良好 absolute fit 解释为真拓扑在库内。

保留：源位置/偏差错误造成的模型失配；弱支路或不可接入地点造成的信息不足；预算、AC 失败和局部盆地差异造成的计算问题。统一输出阻抗误差、似然差、候选覆盖、误拒绝率、错误候选排除率、保留真且排除所有错候选的次数、动作可行性和各阶段耗时。

## 7. 已有组件的实际限制

| 组件 | 可复用部分 | 不能直接继承的假设 |
|---|---|---|
| `ac_measurement_fit.py::fit_ac_pqv` | 原始 P/Q/V + 可选总表加权残差，真实末端 P/Q EIV，线路跨时共享，AC 物理计算 | source 位置和幅值输入固定；局部 least squares；不给自由 bias 和源幅值 MLE；传入的 sigma 固定，不由残差重估 |
| `passive_bridge_v9.py` | observation-only RNJ/WZZT 排名，`certified_supports=()`，可删除/交换/重建 support | 有限候选库；WZZT 代理和候选预算影响覆盖；源 nominal 入口当前要求非负 P；候选排名不认证 |
| `ac_general_v9.py::RadialCandidate` | laminar support 到物理径向树、公共 source junction、共享路径矩阵 | degree-2 无负荷链聚合；单相平衡、径向、同一公共源点；源位置和未测内部负荷并非任意 |
| `ac_general_v9.py::ac_batch` | 数值 AC 固定点和 equation residual 思路 | 原函数固定 root=1、`x=kappa*r`；新 likelihood 必须用独立 r/x 的适配器；不能直接复用 `solve_outer` 作为统计判据 |

本文没有从旧外 LP 未决推断 MLE 无效，也没有从新的局部拟合成功推断 MLE 必然有效。完成标准是明确合成条件下有效、信息不足、失配和计算失败的边界，再决定是否扩大网络。
