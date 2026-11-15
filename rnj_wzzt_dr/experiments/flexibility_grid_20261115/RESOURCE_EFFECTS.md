# 四网络资源配置、网格效果与负例分解

2026-11-15。基于已经完成的36格和4608条执行记录，只读核算与绘图；没有新增拟合、优化、AC计算或随机路径。完整网格结果见[RESULTS.md](RESULTS.md)。

本批的主要问题已经定位到**冻结电压响应在采购方向上失准**。Soumalas的两个明显负例中，候选之间的预测距离增长，实际响应偏差却产生更大的反向评分。下面把资源配置、实际调用和评分变化分开呈现。

进一步的参数QP、历史Q激励、公共响应项及误差尺度核查见[RESPONSE_BIAS.md](RESPONSE_BIAS.md)。

## 图与范围

|图|本地PNG与PDF同名|含义|
|---|---|---|
|四网络配置|[resource_configuration.png](../../../outputs/flexibility_resource_configuration_20261115_v1/resource_configuration.png)|真实物理树的离线展示；逐终端PV、EV、热水与无功合同|
|36格恢复变化|[effects_by_network.png](../../../outputs/flexibility_resource_effects_20261115_v1/effects_by_network.png)|四网络×三方法；每格无采购→采购恢复数、费用；缺真候选与回退均标记|
|36格资源调用|[resource_use_by_network.png](../../../outputs/flexibility_resource_effects_20261115_v1/resource_use_by_network.png)|每格实际交付的绝对P/Q总量及无功使用率|
|负例节点与时段|[negative_Q_requests.png](../../../outputs/flexibility_negative_figures_20261115_v2/negative_Q_requests.png)|两组Soumalas的冻结Q请求；同时列P请求，避免误读为全部纯Q|
|负例评分分解|[negative_evidence_decomposition.png](../../../outputs/flexibility_negative_figures_20261115_v2/negative_evidence_decomposition.png)|同一对候选的原优势、距离收益、响应误差、AC余项、量测变化及最终优势|

每个网络的九格共用同一物理日、预测、资源位置与合同。变化的是训练/验证采样密度及量测标准差。因此这些图描述固定配置在不同数据条件下的效果，没有比较重新布点或扩容的收益。每格只有一份预测、16条配对交付与量测路径。

## 资源配置及真实调用

|网络|Q供给节点/可测终端|双向Q合同合计，kvar|Q合同容量小时，kvarh|非PV的P合同上限合计，kW|
|---|---:|---:|---:|---:|
|Pengwah18|14/18|29|696|3.242|
|Flynn16|13/16|28|672|2.961|
|Paper15|11/15|31.5|756|4.736|
|Soumalas11|9/11|21|504|3.518|

Q为另行假设的独立补偿合同，全天可用；正值为吸收、负值为注入。根与隐藏节点没有配置资源。Q合同容量与已验证的设备铭牌容量分开；图中没有从PV登记功率推算夜间逆变器能力。

P合同包括PV弃光、EV移时和热水调整。EV及热水仍受时窗、能量缓冲与末端恢复约束，表内P上限之和不等于可同时持续交付的功率。**12个PV合同在本批预测与实际背景的“净反送”可用量代理下均为0，实际PV调用也全部为0。** 这是当前可用量规则的结果，不代表居民PV实际没有发电；若要允许自用PV弃光，需单独获得/建模PV出力，不能直接从正净负荷中识别。

图中无功使用率为“16条路径平均绝对Q交付量÷全天合同容量小时”。正负调用都累加，分母不乘2。该量描述一天使用强度，不是每小时峰值利用率，也不是储能容量。

|设置、方法|恢复：无采购→采购|实际P，kWh|实际Q，kvarh|Q使用率|结算费，元|
|---|---:|---:|---:|---:|---:|
|Pengwah，96/48，P/Q 1%、V 0.1%，正态|5→16|0|13.765|1.98%|1.688|
|同格，固定指数|5→16|0|35.351|5.08%|6.268|
|Soumalas，96/48，P/Q 0.25%、V 0.025%，正态|16→16|0.105|31.232|6.20%|6.748|
|同格，固定指数|16→0|0|40.332|8.00%|7.948|
|Soumalas，96/48，P/Q 0.5%、V 0.05%，正态|16→3|21.344|238.820|47.38%|20.782|
|同格，固定指数|16→0|0|72.165|14.32%|9.226|
|同格，含预测误差指数|16→0|约0|71.766|14.24%|9.813|

表中调用量和结算费都是执行均值。费用包含示例容量费和调用费，不含设备建设成本。明显负例在零采购下本来16/16正确，资源也没有整体用尽；直接扩容不是这些负例的首要修正。

## 偏差怎样抵消距离收益

令每时刻功率向量为 \(x=[p;q]\)，\(H_t\)为冻结目标响应，\(D=H_t-H_j\)为它与某一竞争者的差；\(H_*\)是用真实线路构造的一阶响应，只用于事后解释。真实干净AC量测写为

\[
y_{AC}=H_*x+r_{AC}.
\]

历史与新日累计平方误差的优势为 \(L=(S_j-S_t)/(2\sigma_s^2)\)，正值有利于目标。直接展开平方项，得到

\[
L=h+\underbrace{\frac{\sum\|Dx\|^2}{2\sigma_s^2}}_J
+\underbrace{\frac{\sum\langle(H_*-H_t)x,Dx\rangle}{\sigma_s^2}}_{b_{response}}
+\underbrace{\frac{\sum\langle r_{AC},Dx\rangle}{\sigma_s^2}}_{b_{AC}}.
\]

因此购买前后的实测优势变化，恰好等于距离变化、响应误差变化、AC近似余项变化和量测变化的和。响应误差项取决于误差朝向哪个候选，而不只是误差大小。量测变化同时包括P/Q和电压表计影响。

下面每行始终固定同一对候选，均为16条路径的平均；编号为该冻结库的原生候选编号。目标5具有正确内部结构，其完整响应支持及参数仍可能有误。

|Soumalas，96/48|距离变化|响应误差变化|其中R块|其中X块|AC余项变化|量测变化|原优势→最终优势|
|---|---:|---:|---:|---:|---:|---:|---:|
|低噪声，固定指数，5对6|+11.362|−45.328|+8.235|−53.563|−2.631|−1.403|10.337→−27.663|
|中噪声，固定指数，5对6|+10.890|−48.471|+6.531|−55.002|−0.812|−0.412|11.232→−27.573|
|中噪声，正态，5对原竞争者6|+118.014|+204.547|+10.069|+194.477|−10.986|−2.872|11.232→319.935|
|同一正态动作，5对新赢家3|+558.101|−693.236|+37.914|−731.151|+5.724|+3.177|65.623→−60.611|

这次分解把此前笼统的“实际偏差”进一步拆开：**冻结响应误差主导退化，AC一阶近似余项相对较小。** 正态方案对原竞争者6的收益很大，却在另一个比较方向上失效；候选3从一开始就在规划约束中。对候选3，规划预测优势646.83，改用实际交付P/Q但保持规划均值模型后仍为576.85，干净AC实际值却为−61.94。主要落差发生在电压响应模型，而非这份功率预测。

两格目标R/X完整响应矩阵的相对误差分别为5.46%/25.44%、10.23%/45.97%。这是完整矩阵的误差，包含公共、末端和内部响应成分，不能直接读为逐条线路阻抗误差。R/X投影分解也不是“移除P或Q资源”的反事实试验；候选差D同时包含P和Q。

## 为什么换成指数约束反而选了不利方向

固定功率指数条件同时奖励平均优势、惩罚评分方差。共同根表误差使多个测点同向的预测差受到更大方差惩罚。Soumalas中噪声例中，正态动作对5→6的规划均值/方差为150.02/822.62，固定指数动作为14.63/23.45。后者主要让103吸收无功、104/106/107/108注入无功，压低了模型方差，但这个组合恰好放大了实际X响应的不利误差。

这解释了更保守条件为何可以改变节点组合并恶化结果。尚未进行固定lambda或初值的独立消融，不能据此断言改变其中一个参数就能消除负例。

正态中噪声负例保留共同LP初值；两种指数负例则在共同池没有可行初值时，由原生求解器找到新点。它们都通过各自模型约束。因此初值和限时会影响采购点，但不能统一解释所有负例。费用上下界尚未全部闭合，详细状态及一张撤回下界的报价见[RESULTS.md](RESULTS.md)。

已缓存的完全交付、无新日表计误差AC也出现指数负例。全批母线电压0.995746–1.023752 pu、资源约束核验通过。因此本次主要问题不是电压越限或设备未交付。线路热额定值仍缺失，没有据此给出热安全结论。

## 复核与复现

108个主方法网格行的请求/交付P/Q、容量小时分母、费用及恢复数，与独立资源核算逐项差0。物理分解回放512个有向比较，native SSE误差0、恒等式误差最大2.84e−13；R/X分解与独立脚本最大差2.27e−13。PNG均已目视核查，PDF同步导出。

新增入口均只读已有完成数据，输出目录必须不存在：

```powershell
python -B -m rnj_wzzt_dr.experiments.flexibility_grid_20261115.plot_resources --run outputs/flexibility_grid_execution_20261115_v1 --out outputs/new_resource_configuration
python -B -m rnj_wzzt_dr.experiments.flexibility_grid_20261115.plot_effects_by_network --report outputs/flexibility_grid_report_20261115_v1/report.json --out outputs/new_resource_effects
python -B -m rnj_wzzt_dr.experiments.flexibility_grid_20261115.response_bias_diagnosis --run outputs/flexibility_grid_execution_20261115_v1 --ids soumalas11__2024-01-04__s2__m0p5 soumalas11__2024-01-04__s2__m1p0 --target-native 5 --competitors 6 3 --out outputs/new_response_diagnosis
python -B -m rnj_wzzt_dr.experiments.flexibility_grid_20261115.plot_negative_cases --diagnosis outputs/flexibility_grid_negative_mechanisms_20261115_v1/results.json --waterfall outputs/flexibility_resource_visuals_20261115_analysis/recommended_figure_data.json --run outputs/flexibility_grid_execution_20261115_v1 --out outputs/new_negative_figures
```

数据与图在本地忽略目录，提交仅包含可复用代码和汇总说明。配置来源见`flexibility_resource_configuration_20261115_v1/configuration.json`；独立调用核算见`flexibility_resource_usage_audit_20261115_v1`；响应分解与等价检查见`flexibility_resource_response_bias_replay_20261115_v1`；作图收据保留输入与输出哈希。
