# 当前数学模型与实现的独立核对

2026-10-18。范围为C盘研究分支的`reference_benchmark`及其实际依赖，和D盘隔离开发目录的`ac_block_fit`。本次仅阅读源码及`current_model.md`，没有运行新的拟合、规划、统计或GPU试验。以下路径以两个目录的各自根为准；对应文件绝对路径和哈希见同名JSON。

## 1. AC与计量目标：已对应

固定候选k，m为非根编码节点/支路数，n为末端表计数，T为时段数。R、X是跨T共享的逐边标幺参数；s_t=(s_t^P,s_t^Q)为当期未计量源，正数表示消耗。根复电压取实正值v_0t。

令C把n个末端负荷放入m个节点，B为根路径矩阵，e_1为固定源接口。程序求解

\[
L_t=C(p_t+\mathrm i q_t)+e_1(s_t^P+\mathrm i s_t^Q),\quad
Z=B\operatorname{diag}(R+\mathrm iX)B^\top,\quad
V_t=v_{0t}{\bf1}-Z\overline{L_t/V_t}.
\]

电压由完整常功率AC固定点求出；隐式导数在收敛且非奇异的所选AC分支上成立。它没有以量测误差为目标直接迭代电压。对应`likelihood.py:168–224`，固定源在191行、末端总P/Q在192行、Z在194行。

当前p=P_obs-b_P、q=Q_obs-b_Q是固定输入。输出h_t包含n个电压幅值及根总表P/Q，y_t为相同通道的偏差修正观测，W_t为输出标准差倒数对角阵。目标为

\[
r_t=W_t(h_t-y_t),\qquad
F_k=\tfrac12\sum_t\|r_t\|^2,\qquad
\ell_k=-F_k-\tfrac12\sum_{t,c}\log(2\pi\sigma_{tc}^2).
\]

`FitResult.loss=2F`。输入P/Q随机误差没有进入此残差或传播为输出协方差。总表P/Q必须提供；本接口不接独立源遥测。对应`ev_fixed_train_fit.py:89–124,165–176`，`reference_benchmark/fitting.py:218–244`。因此应称“完整AC条件输出拟合”，与同时估计真实末端P/Q的误差变量MLE区分。

## 2. 参数范围与电压验收：语义需保持

R/X满足给定盒及rho_low R<=X<=rho_high R；源P/Q各有逐时盒。实际参考算例的rho范围默认[0.05,5]，R/X允许0，根电压和仪表偏差固定。公开上界来自完整资产清单的R/X总量，不按真拓扑路径逐边收紧。对应`reference_benchmark/cases.py:104–129`、`representation.py:230–258`。

求解器使用

\[
X_e=L_e(R_e)+\alpha_e[U_e(R_e)-L_e(R_e)],\quad
L_e=\max(X_{min,e},\rho_{low}R_e),\quad
U_e=\min(X_{max,e},\rho_{high}R_e),\quad 0\le\alpha_e\le1.
\]

截面非空限定R的范围；零宽坐标被适当删除或保留为冗余截面。角点处的R/alpha坐标驻点未必给出原R/X锥的一阶证书。对应`ev_rx_fit.py:32–77,104–147`。

电压[0.8,1.2]是最终结果验收条件，未作为TRF/GN每一步的不等式。超界结果被标为不可用；程序并未另外求解一个电压约束下的最小二乘。对应`reference_benchmark/fitting.py:192–203`与隔离`nonlinear.py:178–180`。实际[0.9,1.1]运行电压带又是另一个执行审计项（`reference_benchmark/study.py:55–68`）。

## 3. 源口是否固定：固定角色，另有物理适配前提

每个候选都在全集support边末端的节点1注入源P/Q；源不会逐时或按竞争者换到不同支路。`RadialCandidate.from_supports`把全集support排在第一位，`likelihood.py:191`固定使用它。

真值适配器额外要求物理源口是所有量测母线的最近共同上游节点（LCA），即完整support串联链的末端；否则拒绝适配（`reference_benchmark/representation.py:140–146`）。这一前提使串联聚合不跨过源注入。

这覆盖“一个共同根侧P/Q接口”，没有覆盖任意支路上的多个独立新能源注入。源口与某个表计的已知共址关系没有作为候选硬约束：实际`planning_inputs`不传source_bus（`cases.py:239–248`）；各候选相关虚拟叶边仍可估计。因此“固定共同接口”准确，“已强制所有物理共址信息”过强。

## 4. QR的等价对象

令theta为自由R/alpha坐标；G_t、S_t分别为白化残差对共享theta和当期二维源的导数。沿尺度D后的局部有界子问题为

\[
\min_{l\le\eta\le u}\tfrac12\sum_t\|r_t+G_t\eta_\theta+S_t\eta_t\|^2
+\tfrac\lambda2\|\eta\|^2.
\]

活动变量固定后，对自由源矩阵[S_t;sqrt(lambda)I]作QR，保留正交补残差再回代源。这保持该活动面上线性阻尼子问题的最优值；外层仍需用完整非线性AC决定接受。对应隔离`linear.py:67–156,159–260`、`nonlinear.py:54–100`。它不是完整非线性源profile的闭式消元，也不是复制原TRF轨迹。

无阻尼且自由源可不受界地最小化时，投影为I-S_t S_t^+；若共享线路步长有界，约简后仍须保留那些界。`current_model.md`式(5)建议显式写出这一条件。

独立审计报告固定原尺度的投影梯度。当前GN内部停止却使用当前scale，且可能由相对目标或步长停止（`nonlinear.py:35–48,96–99,115–123`）；固定原尺度指标是另报项。因此“共同验收口径”应与“求解器实际停止条件”分开。

## 5. 三个EV规划目标及梯度

规划固定每个候选的一组历史R/X，并固定各自源点预测。令mu_k,t(0)为无EV预测输出，H_k,tj是完整AC对j点1kW有功功率的导数；其中包含1/base_kw换算。当前EV无功为0，但总表Q响应仍有线路无功损耗效应。对应`g_precision/context.py:19–25,44–66`。

对白化差值定义d_ab,t=W_t(mu_a,t(0)-mu_b,t(0))、g_ab,tj=W_t(H_a,tj-H_b,tj)。单EV每期最多一个非零接入功率，故同刻跨地点功率乘积为0；无需假设电网不同支路弱耦合。实际任务为

\[
D^{SQ}_{ab}=\tfrac12\sum_{t,j}\|g_{ab,tj}\|^2u_{tj}^2,
\]
\[
D^{AQ}_{ab}=\tfrac12\sum_t\|d_{ab,t}\|^2+
\sum_{t,j}d_{ab,t}^{\top}g_{ab,tj}u_{tj}+
\tfrac12\sum_{t,j}\|g_{ab,tj}\|^2u_{tj}^2.
\]

程序用全部候选无序对的最小任务值，最大化max_u min_ab D_ab；没有删掉难竞争者（`g_precision/coefficients.py:138–166`）。导数为

\[
\partial D^{SQ}_{ab}/\partial u_{tj}=\|g_{ab,tj}\|^2u_{tj},\quad
\partial D^{AQ}_{ab}/\partial u_{tj}=d_{ab,t}^{\top}g_{ab,tj}+\|g_{ab,tj}\|^2u_{tj}.
\]

G-SQ在0功率处梯度为0；MILP利用的是完整二次系数。G-AQ的背景项是未来无EV预测差，不是历史似然损失。历史通过候选、训练R/X及源预测进入，但当前目标没有直接加入历史likelihood损失、WZZT模糊块权重或线路响应不确定性。

P-SQ是sum w_tj u_tj^2，w是逐点电压表方差倒数的公共归一化权重，不是平均值。本轮sigma_V各点各时段相同，故w=1（`ev_rx_precision.py:24–36`，`reference_benchmark/cases.py:98`）。它对拓扑差异不作区分。

实际主选择器调用`full_pipeline/solver.py`，而非早期`g_precision/milp.py`。共享平方弦s=(a+b)u-ab满足0<=s-u^2<=(b-a)^2/4，任务二次系数均非负，故MILP最佳界用于上界；合法功率代回真二次式给可达下界。其界只覆盖冻结的二次代理（`full_pipeline/solver.py:143–213,216–307`）。

完整AC预测分数另行记录及检查物理可行性；目前不会用该分数重新选出所有计划中的最好者（`reference_benchmark/selector.py:206–240`）。规划内没有源或线路重拟合。最终FIX/JNT再对历史与新增量测联合解释（`inference.py:186–299`）。

## 6. EV任务与公平性

各域共享初始24kWh、终端30kWh、功率11kW、效率0.95、总充放电吞吐<=15kWh，移动<=1次、驾驶0.2kWh、移动/接线15分钟及工作人员30分钟预算。移动是可选项；父计划完整继承。对应`selector.py:117–137`、`ev_mobile_service_domain.py:270–348`、`ev_mobile_v4_domain.py:25–54`。

这是相同电池服务任务，非严格相同电网净电量；损耗与驾驶会改变取电。工作人员成本作为资源预算，不直接从信息目标扣分。移动后至少0.25kWh服务避免完全无用行程，但不消除驾驶补电对平方目标的影响。后续空间价值分析应明确拆开新增电量、择时、位置及接线窗口。

## 7. 对current_model.md的收束意见

主体AC、条件似然、R/alpha及G目标公式正确。需要保持的修订：

1. WZZT/RNJ提供候选/修正；AC初值由历史V平方与P/Q回归获得，不是直接使用WZZT阻抗初值。
2. 无阻尼投影式补充共享线路步长范围/自由活动面的前提。
3. 固定尺度gamma是共同独立审计指标，不等同于动态尺度求解器内部停止条件。
4. 保留“相同电池服务任务不等于严格同电网电量”的说明。
5. 源位置写成固定共同根侧接口，并说明真值适配的LCA条件及共址信息尚未使用。

## 8. 后续接入QR的最小干预与验收

不改本轮冻结批次。新协议中仅在AC拟合接口加入显式、可记录的后端选择：legacy、QR、QR后失败回退legacy。历史拟合、FIX源拟合和JNT拟合必须统一经过该接口，不能只换历史阶段而漏掉后处理。现有FitResult、残差、物理域、候选保留和成本统计不变；不使用跨模块全局monkeypatch。

回退只依赖已经到达数据、数值失败、完整AC验收及预先固定的精度规则。双方使用相同起点、域和预算；所有失败及额外费用都保留。固定线路源问题可采用已验证的逐时二维/批量求解，联合线路问题采用QR/BVLS。GPU作为独立可选后端。

建议门槛：块导数与独立导数核对；完整AC/功率平衡<=2e-10；源与R/X域及最终电压检查通过；原尺度梯度和损失同时报告；返回值不劣于可行起点；全部相关候选而非只看赢家；在零边/活动界/秩亏和H12/48/192均保留失败。角点另检查原R/X实际可行方向，不能仅以alpha梯度为证。

降低计算时间与改变候选排序分别验证。一旦新求解器改变可用候选、训练R/X、规划或最终分数，随后校准与评价使用新冻结版本，重新独立校准；不能沿用旧冻结阈值宣称同一错误率。
