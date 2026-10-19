# C5：实际规划动作的有限工程先验 EIG 审计

`adaptive_eig_audit.py` 是阶段 C5 的独立事后审计模块。它对阶段 C 已冻结的
MILP、恒充、固定/随机、旧动作库以及无 EV 动作做实际 AC 均值重播，再估计
其预期拓扑信息量。审计中的 shortlist 选择不改变已经执行的 phase C 动作。
此阶段不重新做频率学校准，也不扩大物理试验分母。

## 1. 输入边界与模型范围

模块只取每个 NPZ 的
`history_p/q/v/master_p/master_q/root_v`，只取每个 trial JSON 的
`records[*].method`、`records[*].action.p` 和 `.q`。JSON 的普通解码会建立
语法树，但选择/回放只访问上述白名单；不取 `case`、生成种子、真拓扑、
真线路、未来计量或噪声字段。NPZ 按需索引，不将全部字段转成字典，也不
解码无关 object 数组。记录的观测/动作哈希绑定的是实际使用字段，不是
混合文件的完整字节。输入字段改变时已有审计配置不能悄悄复用。

当前审计针对最初无 EV 的 pilot 历史。历史总 P/Q 不被减掉或重新添加 EV；
如以后审计多轮历史，须给先验路径加入已实际执行的历史 EV 注入一次，再
重新定义/冻结模型，不能直接把本模块当成该更新已实现。

历史 root_v 是已知物理输入，在 atom AC 回放中按历史逐时值条件化。未来
root_v=1 是显式声明的预测输入，不读取未来根计量。固定偏差由
`simulation.model(3)` 的独立校准值定义，误差方差已知；无新增自由偏差。
末端控制 `action.p/q` 的单位是 pu，基准功率 20 kW，与当前 driver 的保存
接口一致。每个 future 均值加入这份实际 P/Q 一次，不换成最接近的旧动作。
路线、SOC、移动和能量可行性仍由原 operations checker 审计；此模块不因
只有 P/Q 就宣称已再次检查完整路线。

pilot 的生成模型使用逐时随机负荷/源和逐边连续 R/X 扰动。本审计下述两条
正弦路径、全局阻抗缩放和三个内部支路档位是**预先声明的有限工程先验**，
刻意不同于该生成总体。它不是连续 proper prior 的求积网格，不是 full AC
自由参数族的覆盖集，不提供该总体中的拓扑正确概率或统一误拒绝控制。

## 2. 80 个 atom 与分层先验

拓扑标签固定为
\(\mathcal H=\{\mathrm{star},01,02,12\}\)，包括 star 和三个 cherry。
每个 atom \(a\) 包含整段 H+N 的共同线路 \((R_a,X_a)\)、末端背景
\((p_a^0(t),q_a^0(t))\) 和同物理 node1 的源路径
\((s^P_a(t),s^Q_a(t))\)。源在各电压输出不能独立变化。

线路全局 R 缩放与 X 缩放独立，各在 \(\{.85,1.15\}\) 中取值；背景/源
联合路径来自 `audit_common.latent_path(path_id)` 的两个固定档位。star 不含
内部支路强度变量，因此有 \(2\cdot2\cdot2=8\) 个 atom。每个 cherry 的
内部 R/X 档位为
\((.00003,.00002),(.006,.004),(.012,.007)\) pu，对应 weak/medium/strong，
故各有 \(2\cdot2\cdot3\cdot2=24\) 个 atom，共 80 个。H 与 N 使用同一份
线路数组；路径包含整个时段，不能用 future 数据选择路径档位。

默认 \(\Pr(h)=1/4\)，topology 内部归一，而非把所有 atom 均匀后让 star
先验质量变成 1/10。令 star 质量为 \(s\)，cherry 内 weak 质量为 \(w\)，则

\[
p_a=\begin{cases}s/8,&h(a)=\mathrm{star},\\
\frac{1-s}{3}\frac{w}{8},&h(a)\ne\mathrm{star},\ a\text{ 为 weak},\\
\frac{1-s}{3}\frac{1-w}{2\cdot8},&h(a)\ne\mathrm{star},\ a\text{ 为 medium/strong}.
\end{cases}
\]

默认 \((s,w)=(1/4,1/3)\)，所有 atom 质量严格为正，和为 1。先验只由上述
声明生成，历史数据只更新权重。历史 root 条件化不改变拓扑/线路/路径先验。
任何必要 atom 或 atom/action AC 求解/物理域失败均使该历史审计失败，保留
失败记录；不删掉不利 atom 后重新归一。

## 3. Gaussian 更新与混合预测

令 \(y_H\in\mathbb R^{D_H}\) 为由已知计量标准差白化的历史观测，
\(\mu_{a,H}\) 为同一白化规则下包含固定偏差的 AC 均值。根电压不作为有
误差的计量块重复加入。独立单位协方差 Gaussian 密度为

\[
f_a(y_H)=(2\pi)^{-D_H/2}
\exp\{-\tfrac12\|y_H-\mu_{a,H}\|^2\}.
\]

Bayes 公式消去共同密度常数，得到

\[
w_a(H)=\frac{p_a\exp(-L_{a,H}/2)}{\sum_b p_b\exp(-L_{b,H}/2)},\qquad
L_{a,H}=\|y_H-\mu_{a,H}\|^2,\qquad
\pi_h(H)=\sum_{a:h(a)=h}w_a(H).
\]

这些有限和由 logsumexp 精确到浮点数精度计算，不按历史损失删 atom。
`log_evidence` 是白化坐标下的有限模型证据；对原始计量单位比较证据，还
须加相应 whitening Jacobian。共用 whitening 常数不影响这里的权重或 EIG。

给实际控制 \(u\)，背景总 P/Q 为 \(p_a^0+u_P,q_a^0+u_Q\)，AC 方程产生
白化均值 \(\mu_{a,N}(u)\)。预测不是代入历史局部 MLE 的一个 Gaussian，
而是积分有限参数/状态不确定性的混合：

\[
q_h(y\mid H,u)=\sum_{a:h(a)=h}\frac{w_a(H)}{\pi_h(H)}
\phi_{D_N}(y-\mu_{a,N}(u)),\qquad
q(y\mid H,u)=\sum_h\pi_h(H)q_h(y\mid H,u).
\]

同一 posterior atom 顺序、已知协方差和 meter 偏差必须用于全部动作。
这里 source 先验只有两个联合路径，不是 full AC likelihood 中逐时自由的
源参数；将其结果迁移到自由参数联合重拟合需要另做检验。

## 4. 拓扑 EIG 的等式及含义

条件互信息定义及 Bayes 公式给出

\[
\begin{aligned}
I(u\mid H)&=I(h;Y_N\mid H,u)
=\sum_h\pi_h\int q_h(y)\log\frac{q_h(y)}{q(y)}\,dy,\\
\pi_h(y)&=\frac{\pi_hq_h(y)}{q(y)},\\
I(u\mid H)&=\int q(y)\sum_h\pi_h(y)
\log\frac{\pi_h(y)}{\pi_h}\,dy
=H(\pi)-\int q(y)H(\pi(y))\,dy.
\end{aligned}
\]

证明：将 \(\pi_h q_h/q\) 代入互信息的 log 比值，使用
\(q\pi_h(y)=\pi_hq_h\)，得到 posterior KL 的期望。展开 KL 中的
\(\log\pi_h(y)-\log\pi_h\)，并用 \(\int q\pi_h(y)=\pi_h\)，得到熵差。
因此 \(0\le I(u\mid H)\le H(\pi)\)。这个上界约束**期望** EIG，不约束
每个观测的 pointwise posterior KL。它是拓扑信息量；总 atom 信息还含
拓扑内线路/路径信息，不能把二者混为同一目标。

EIG 的非负性、熵上界和有限后验概率是声明模型内的结论。它们不说明真实
pilot 数据属于该支持，不说明 posterior 最高质量的拓扑在真实系统中正确。
本审计也不将 EIG 等同于 Gaussian plug-in KL、复合似然排错功效或 full AC
联合优化后的期望差；混合积分与最小化操作不是同一件事。

## 5. 固定选择后验证与误差记录

每个拓扑层固定抽 \(M\) 个样本；层内按 \(w_a/\pi_h\) 抽 atom，再加独立
Gaussian 计量误差。不同拓扑层独立，不同动作共用 atom/noise 随机数。
对每个模拟观测计算 posterior KL 被积函数 \(G\)，估计为

\[
\widehat I=\sum_h\pi_h\overline G_h,\qquad
\widehat{\mathrm{SE}}(\widehat I)^2=\sum_h\pi_h^2s_h^2/M.
\]

两动作差的标准误使用同样随机数下逐样本差的层内方差，保留协方差；不将
两边 SE 当成独立相加。熵差 estimator 独立作为等式核查，保存二者之差及
其 SE。RB 条件期望对普通 iid 混合抽样有方差分解证明；固定层样本数时
不能据此声称对全部分层配置严格优于旧 log-ratio estimator。

默认一条 2048/层积分流选 shortlist winner，随后冻结这个动作。在
2048/层、8192/层两个预算上各用三条新的、彼此独立的可复现随机流，验证
其相对全部已规划动作的 EIG 差。验证阶段即使出现别的最高估计也不换赢家。
独立重复估计的平均 SE 为 \(\sqrt{\sum_j\mathrm{SE}_j^2}/J\)。选择估计
有 winner 偏乐观，优先报告冻结动作的独立验证差、配对 SE、逐 seed 值及
两预算变化。三条积分流只量化给定历史/先验的 MC 误差，不是三个独立网络。
EIG 和差估计允许很小的负浮点/MC 结果，不截断后宣布收益。

默认 baseline 审计至多前 8 个输入历史；完整先验敏感性只对预先按文件名
排序的前 2 个历史，使用 \(s\in\{.1,.25,.5\}\)、\(w\in\{.25,.75\}\) 六个
组合，各重复同样的固定选择/独立验证流程。可显式增大
`--sensitivity-limit`。质量敏感性不等于支持敏感性：两个 latent path、线路
档位分辨率、固定偏差和源预测仍是未扩展的假设。

## 6. 下溢、支持失配与可报告结论

全部 log atom/topology 质量保存。指数后的普通浮点质量可能下溢为零，
这不等于数学上删掉该拓扑。RB 分层按浮点质量加权，极小贡献也可能下溢。
保存零显示质量数、atom/conditional ESS、当前拓扑熵，以及从保留的 double
log 质量计算的 log 熵项。后者是数值诊断，不是任意精度熵证书；舍入为零的
dominant log 质量仍有其数值限度。

当熵低于预声明的 \(10^{-10}\) nat，动作排序标为数值排序；不以约
\(10^{-14}\) 的有符号误差宣称 EV 改善/恶化。完全相同 future 均值哈希的
动作作为同预测组报告。零 EIG/SE 可能是有限先验已饱和或下溢，不能证明
完整连续 AC 族中没有剩余信息。

额外保存最小 atom 历史损失、损失/计量维数及 P/Q/V/master P/Q 各通道白化
残差 RMSE。它们是支持适合度诊断，不自动是卡方检验或已校准 GOF p 值。
若有限 bank 的历史拟合很差却后验极尖锐，合理解释包括有限支持失配造成的
错误确信，而非“已经可靠辨识”。ESS 是有限后验浓度，非 MCMC 收敛指标。

结果还绑定 atom 清单、prior、H 实际观测、meter、H/future 均值、实际动作及
源码 SHA-256。固定均值缓存只对精确控制/同 atom/meter 复用，不量化或
换动作。未来计量与真值均不进入这些输入哈希。WZZT/RNJ 来自同一 H，
本审计不会再将其摘要乘为独立似然。

## 7. 调用与核验

集成到 repo 后调用示例：

```powershell
& 'D:\apps\miniconda3\envs\Topo\python.exe' -m research_experiments.mobile_ev_mle.adaptive_eig_audit `
  --input 'research_experiments/mobile_ev_mle/outputs/adaptive_milp/pilot_v1' `
  --output 'research_experiments/mobile_ev_mle/outputs/adaptive_milp/eig_c5' `
  --limit 8 --sensitivity-limit 2 --selection-samples 2048 `
  --validation-samples 2048 8192 --validation-repeats 3
```

本模块的纯/小型测试核查：80 atom 与全部 topology 质量、star 不重复强度、
先验敏感性归一、已知历史 root 的条件 AC 均值、EV 单次注入与独立直接 AC
构造、禁止字段访问哨兵、毒化 NPZ 未解码、动作顺序/相同预测/精确缓存、
冻结选择与实际积分预算、binary Gaussian 独立 quadrature、相同预测零信息、
下溢诊断、小型真实 RB 回放和输出输入哈希边界。测试通过只证明这些接口
和小型数值检查；不替代真实 pilot 审计或连续 proper-prior 研究。
