# 逐时高斯DR与原验证裕量：当前实现

2026-11-01。只用原paper15/b2/.0005及历史两族；完整288训练/288验证，R/X和.16输出映射固定。代码是连续SOCP，无场景整数变量，无Big-M，无总费用上限。主风险对象是原L1验证裕量，而不是拓扑真实性。

每资源每槽优化有符号均值u_it及a_it>=|u_it|，报价pi_it=a_it/lambda，lambda=.5。响应U_it=u_it+cv*a_it*Z_it，Z_it逐节点逐槽独立标准高斯，cv=sqrt(.025/3)，Q响应=.35U。最优可取a=|u|：增大a只提高费用和方差，不提高激励均值。于是均值为方向×lambda×价格，标准差为cv×lambda×价格。报价/方向均由连续解决定。

费用按双向计划调节量结算：C=.25 sum(a_it²/lambda)，预付且随计划固定。这是显式新合同，不沿用旧‘仅实际削减电量付费’语义，避免取消能量等式后增负荷免费。原1kW基线/2kW组件上限形成u∈[-1,1]均值指令界；无总功率、无全天净能量等式。高斯实现无有限支撑，均值界不保证每次交付界；测试按原组件界如实统计越界，不裁剪/重抽。

预测只用三个历史training完整P/Q/Y日。v_b=[vecP_b,vecQ_b,vec rF_b]，mu=mean(v_b)，L=[v_b-mu]/sqrt(2)，v=mu+Lz。保留跨节点时段、P/Q与参考误差共同变化，但协方差秩仅2；这是小样本Gaussian工作假设。validation只用于历史损失常数，不参与分布拟合。预测DR电压响应采用冻结F，故rF分布与价格无关；rC增加(DeltaR DeltaP+DeltaX DeltaQ)。F并非用真值选出。

以微p.u.²为单位，d=96*15，K=.6(HC-1.03HF)-1e-6，alpha=.4/d，beta=1.02：

M=K+alpha(||rC||1-beta||rF||1)。

固定signC/signF=历史残差均值符号（零取+1），G=K+alpha(signC'rC-beta signF'rF)。共同Gaussian基线给出mG0及bG，DR梯度g_it=alpha sum_j signC_tj*(DeltaR_ji+.35DeltaX_ji)*1e6/base_kw。

mG=mG0+sum(g_it u_it)，sigmaG=norm([bG;g_it*cv*a_it])。

恒等式M=G+2alpha sum[-signC*rC]+-2alpha beta sum[-signF*rF]+。只扣F的翻号上界得到充分条件：epsilonG=.05、epsilonF=.05，epsilon_j=.05/d；t_j=max(0,-signF_j muF_j+Phi^-1(1-epsilon_j)*sigmaF_j)。然后要求mG-Phi^-1(.95)*sigmaG-2alpha beta sum(t_j)>=target。

target=.2 max(-E[M_zero],0)+.0001，期望按Gaussian绝对值公式解析计算，事前固定。此次E[M_zero]=1.685144微为正，target=.0001；它不是每个未来随机日各自120%缺口的概率要求。

主求解min C，全部约束为SOC/线性，NonConvex=0。另预锁naive消融去掉翻号项，使用Phi^-1(.90)；它只保证有符号G。max_bound诊断在相同均值界最大化修正下界，不能当作达标最低费用报价。

如果主修正报价可行才另锁AC批。未达此门控，本轮只用冻结后的2000新Gaussian整段路径评价原M、原weighted_selection.score、费用和资源越界。模型仿真与新AC效果分开报告。
