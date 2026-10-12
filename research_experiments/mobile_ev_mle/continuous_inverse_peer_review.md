# Continuous AC inverse: independent peer review

2026-10-13. 只读独立审查；未执行项目程序、pytest、求解器或 MC，未改研究工作树及主源码，未操作 Git。本报告是唯一写入交付。既有测试数字来自已保存记录，不是本审查重新运行的结果。

**结论：接受声明 nominal 模型下的高压物理坐标替换及 prior 归一化数学论证。** 这一结论条件于文档的完整范围：三末端 star / 三种 cherry，root=1 pu，逐时背景和源 iid，共享于历史/未来的逐边 R/X，单 EV 非负 P≤.3、Q=0，固定外部校准 bias，独立且已知标准差的 Gaussian meters。它不支持任意拓扑、任意动作、现场 prior 或数值 EIG 排序精度。扩展查看 sampler 后发现两项实现契约缺口，已通知 ROOT 和 sampler agent，见下文；它们不否定 inverse 数学论证。

## 审查对象与版本

- `continuous_ac_inverse_math.md`，最终读取含已删去 `tau<=1` 旁句的版本；SHA-256 `74769e8eb2911eb19a3ba50720b896b217f51511651798b944ca44a038829b04`。
- `continuous_ac_inverse.py`；SHA-256 `978dcc07fd8b3b83923ed5943a773e071bce7ebf520cbcb770f056d7e1799b84`。
- `test_continuous_ac_inverse.py`；SHA-256 `b7ff9a50fe32ba93b9f912bed055f5a5461f61af8a8b5f2a9075e95f03551b56`。
- 对照工作树 `research_experiments/mobile_ev_mle/continuous_predictive_eig_protocol.md`、`likelihood.py:168-224`、`simulation.py:29-95`，并快速查看 staging `continuous_eig_sampling.py` 的 proposal、latent likelihood、H bank 和 future likelihood。sampler 正在修改，以下行号指本次读取时的版本。

## 支持的主论证

1. **正向完整覆盖及高压唯一性。** 把 root、shared、terminal 路径贡献逐项展开后，任何允许 prior/action 输入的 bus-impedance 加权行和均受 W<.041 控制；root 下源负荷只走 root edge，不走 cherry shared edge。闭球 `||V-1||∞≤.05` 的自映射与收缩界成立。两个所有母线电压幅值均≥.9 的解直接满足差分收缩界 `.041/.9²<1`，不需要假定该幅值区域凸。结论是选定高压模型的唯一性，不能改写为所有 AC 根唯一。

2. **prior 输入全部映入主表矩形。** 功率守恒给 source+terminal consumption+逐边损耗。root 支路电流上界用四个负荷的幅值和，shared 用两个末端，terminal 用各自幅值；P/Q 损耗上界方向正确。由非负负荷与 R/X，P0,Q0≥0；独立算术复核所得 P0<.866766、Q0<.203269，故 `[0,1]×[0,.5]` 不漏 nominal prior 质量。

3. **反演完整分支与单射。** 对固定端部总 P/Q，leaf 使用 plus-discriminant 高压根，`l=2|s|²/(A+sqrt(D))` 和 `dl/dt=-l/sqrt(D)` 正确。cherry 的 G、G′、B′及隐函数导数均正确，G(.94)<tau、G(tau)≥tau、G′>.9991 给出唯一 junction root。`u>|zs||L|` 确认 shared 高压根。leaf/junction/root 均远离零和 discriminant 边界。所有这些界是逐域不等式，不依赖20项测试。

4. **反演 Jacobian 与密度方向。** 根边电流 `conj(S0)` 给 node-1 receiving power `S0-z0|S0|²`；随后 `Fsrc=S0-z0|S0|²-Sdown(tau)`。实导数为 `J=I-2(r0,x0)^T(P0,Q0)-dSdown/dtau * grad(tau)`，与代码149-157行一致。全域 `||J-I||∞<.019` 使矩形上的 Fsrc 单射并保证 detJ>0。固定端部 P/Q 后 density 因子必须是 `|det dsource/dmaster|`；code183行使用此量，不能改为倒数。把 P/Q、beta 等不变坐标一起考虑，完整 Jacobian 为块三角，额外交叉导数不改变该 determinant。多时刻时因子为逐时 determinant 的乘积，beta prior 只积分一次。

5. **背景 marginal proper。** 对给定六维背景 P/Q，c 的区间交表达正确；Q 条件反解得到乘4的区间端点，但没有额外.25密度乘数。`Lc/(.024*.008^6)` 是原先 normalized conditional densities 对 dc 的积分。由 Tonelli/Fubini，先对全部 P/Q 积分，每个 conditional uniform 因子质量1，最后 c 的质量1；因此背景 marginal 全域归一化，不必用高维有限网格来证明。源 prior 为 normalized 独立 rectangle uniform `1/(.07*.03)`。空区间长度零处理正确。

6. **Gaussian proposal cancellation。** 以已知 bias 修正后的观测作中心、以精确已知 meter sigma 作尺度的八维未截断 Gaussian proposal，等于 P/Q/master measurement likelihood 作为真实坐标的函数。proposal 自带 normalizing constants，故 likelihood/proposal 抵消；V likelihood、背景/源 prior 与 detJ 必须保留，code462-482行如此实现。背景评价用 `total-action`，forward 使用 `base+action`，EV 各加入一次。此结论要求独立时间/通道误差、固定 bias/root 和正确 sigma；估计 bias、不确定 root、相关 meters 或 source 独立 telemetry 会改变 integral。各量保持 pu，不能把 ohm/kW 直接放入此变换。

## 数值包络独立算术复核

以下仅计算文档的保守常数，不是运行项目或采样实验。

| 对象 | 按写明包络算出的值 | 文档采用界 |
|---|---:|---:|
| root / shared / leaf 最大 impedance 幅值 | .007323934 / .024149535 / .044594171 | <.0074 / .0242 / .0446 |
| EV leaf / base leaf / source 最大负荷幅值 | .438525940 / .143892321 / .143961801 | <.439 / .144 / .144 |
| W | .0401334 | <.041 |
| 自映射 / 球内 Lipschitz | .042245685 / .044469142 | <.044 / <.046 |
| P0 / Q0 损耗含内上界 | .866765677 / .203268773 | <.867 / <.204 |
| EV / base leaf 的 sqrt(D) 下界 | .901748927 / .927109891 | >.9017 / >.9271 |
| G(.94) 上界 | .967466222 | <.968 |
| G′ 下界 | .999194580 | >.9991 |
| shared sending derivative 上界 | .021618941 | <.0217 |
| ||J-I||∞ 上界 | .018678546 | <.019 |

这些数值没有挤压到舍入边界；所有严格不等式方向均保守。实际 proof 应读作实数算术保证，不是所有浮点输入都必在40次迭代内收敛的形式化证书。

## 已发现并修复的文档陈述

旧 §4 的 `tau<=1` 对其允许零个别 R/X 的保守域不成立。反例 `r0=.006,x0=0,P0=0,Q0=.5` 给 `tau=1.000009`。engine agent 已删除此旁句；后续全部界仅用 `tau>=.9838`、junction `u>=.94`，所以不改变主证明。当前读取版本已核实修复。

## sampler 扩展审查：支持与需修复项

支持：逐 time 独立 latent clouds 的 sample means 相乘，条件于 beta 是联合 H likelihood 的无偏估计；同 cloud 跨 beta/mode 是合法 common random numbers。七个 prior component 质量归一化；完整 `priorpdf/qproposal` 保留，H 只进入一次。future likelihood 在同 beta particles 上评价，没有重新独立积分 beta。`GaussianUniformProposal` 的 scaled-phi 密度含 `1/prod(canonical_beta)`，multivariate Gaussian normalization 和 mixture 权重完整；prior 外样本保留零 target weight且不重抽。历史后验自归一化、有限粒子库、nested-log bias仍然存在，外层 SE 不覆盖它们。

需修复、已通知 ROOT：

- **观察支持。** sampler `MeterObservation:85-86` 拒绝带噪 v≤0，但其已声明未截断 additive Gaussian meter 赋该事件严格正概率，不能通过“极少发生”得到完整 proper observation model。允许任意有限 noisy-V读数、保持物理 root>0，或明确更换为 truncated Gaussian 并改 likelihood/生成；不能静默重抽。本审查建议前者，ROOT已发修订要求。此项不影响真实 latent AC 电压为正的证明。
- **数值失败不能作为概率零。** `latent_likelihood:477-483` 对 inverse failure 和 prior/domain 外点统一赋零，而 `invalid_inverse_count` 混计所有这些点；history gate 未区分证明支持域内的数值故障。入口应核实 root=1、动作单节点P∈[0,.3]、Q=0。对 background prior density有限、master在R、beta在支持内的点，若 inverse valid/high/cert失败，应报 numerical failure 或使本次积分失败，不应排除为概率零。真正 prior 外的 Gaussian proposals 仍应零权。不声称当前 nominal 小样本出现过此故障；这是所写 math contract 与 fail handling 的缺口。

## 既有验证证据的界限

源码阅读显示测试覆盖四拓扑强/弱、project AC 比较、实中心差分和 forward reciprocal Jacobian、rectangular extremes、广播、背景 c 直接积分及源边界。有限点可支持实现准确度，不能代替上述全域 proof。

`validate_continuous_ac_inverse.py` 实际计算 comparison 和 finite source-mass quadrature，但 `regression_pytest.passed=20` 是 driver 内硬编码 metadata，并未执行 pytest 或读取原始日志。因此 JSON 单独不足以证明20项测试确已通过，应关联真实 pytest tool output/provenance。ROOT表示将核对原始输出及修改前后证据。本审查不复述它为重新验证通过。

最终边界：数学 properness 已支持；floating-point engine 的既有精度检查需依真实记录确认；sampler 两项契约修订需 ROOT 核对；重要权稳定、连续后验近似与 prospective EIG 排序仍由积分诊断决定。没有从本审查得到 global MLE、全拓扑证书或现场正确率。
