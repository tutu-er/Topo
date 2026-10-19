# 标量profile下一阶段构造：解析审计

数学稿为 ev_fixed_train_scalar_profile_math.md。本包只保存解析函数和两槽极小MILP的审计，不含真实AC曲线拟合/曲率认证，不运行研究选择器。每拓扑唯一训练参数保持不变，未增加模型集。

复现测试：

    python -m pytest research_experiments/mobile_ev_mle/test_ev_fixed_train_scalar_profile_checks.py -q --basetemp D:/0-github_workspace/Topo/outputs/ev_train_validation_stage_20261015/outputs/pytest_scalar_profile_fresh

复现两种分段精度的小MILP（输出文件须为新文件）：

    python -m research_experiments.mobile_ev_mle.ev_fixed_train_scalar_profile_checks --output outputs/scalar_profile_toy_fresh.json

四项测试通过，覆盖：

1. \(f(p,s)=((p-s)^2+s^2)/2\) 的解析profile \(d=p^2/4\) 和Schur曲率 \(d''=1/2\)，以及时段独立这一分解前提。
2. 凸、凹二次函数的双向插值误差；“平方残差”不能自动使完整profile对功率凸。
3. 约束源 \(0\le s\le1\) 时活动面切换，\(d'\) 仍为1-Lipschitz，因此全区间误差界有效。
4. 两槽功率和为2、功率范围0–2的MILP。解析最大分数为1，恒充为.5；粗分段给下值.875/上界1.125，细分后为.96875/1.03125。这里直接用解析最优值检查最大化方向；数值求解状态不承担真实AC的全局证明。

unit.xml保存4项测试；toy_milp.json保存连续权重、区间二进制、功率、incumbent和best-bound；verification.json绑定数学/代码/测试和证据哈希、Python/NumPy/SciPy版本。主研究数据读取0、AC调用0、研究选择器调用0。

下一阶段真正需要完成的是：真实AC标量profile的可靠端点界及全区间曲率证据、带原操作网络的MILP接入、物理回代和新独立评价。当前只有数学构造与小例审计。

数学稿第8节另说明现有误差门：只有全源域的统一残差误差界才给profile距离双向界；当前单个proxy源witness核查只给完整AC距离上界。也给出解析反例：旧线性残差加QP已经精确时，粗D折线仍有误差。因此不建议未经同数据误差/成本核查就用粗D插值替换当前策略。独立数学核查另见 math_independent_review.md。
