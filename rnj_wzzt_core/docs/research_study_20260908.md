# 根信息、RNJ+wzzT与人工核查：统一研究方案

更新：2026-09-08。研究对象是**已知根参考点、末端P/Q/V可测、隐藏节点零注入的最简有根树**。研究目标是检验在哪些数据条件下可以同时改善根部分区、末端连接与参数估计，而不是预先把所有消融结果写成联合方法必胜。

本轮沿用已经核验的reference物理场景，不再随算法得分调阻抗、根波动或随机种子。新增实验入口、经典NJ基线、预算适配器和人工查询工具均与生产默认区分；原有运行结果保留。

## 1. 重新组织科学问题

三个问题应按信息流连接起来：

1. **根信息如何进入估计？** 区分根标签/位置、根电压量测、根支路归属、根侧P/Q。准确电压直接校正；有噪电压按公共误差协方差加权；无根电压需要说明公共项先验和规范自由度。
2. **RNJ与wzzT分别贡献什么？** RNJ产生树和稳定末端组，wzzT在声明的候选域内联立拟合层状支持与R/X权重。消融分别检查候选覆盖、冻结/收缩、结构搜索、原端参数重拟合和验证选择。
3. **哪些少量人工事实最值得核查？** 对完整候选树集提出可查证的yes/no问题，按结构分歧下降和成本排序；核查后更新候选和约束，再从原数据重新拟合。

“末端与根部都正确”是需要分区实证检验的目标；隐藏节点的任意算法编号不对应现场设备，最简树也不识别不可观测的串联二度节点细分。

## 2. 合理利用根信息的算法

完整推导见[根信息模型](root_information_model.md)。令A=[P,Q]，Theta=[R;X]，Z=Uref²−V_measured²，a=Uref²−U0_measured²，可用

\[
\min_{R,X,c,b}\frac12\sum_t\|Z_t-A_t\Theta-b_s-c_t\mathbf1\|^2_{W_t}
+\frac12\sum_t w_{0,t}(c_t-a_t)^2
+\frac{\lambda_c}{2}\|c\|^2
+\frac{\alpha_R}{2}\|R\|_F^2+\frac{\alpha_X}{2}\|X\|_F^2,
\]

并施加对称、非负、ordered约束。准确根量测相当于硬锚定；噪根用有限权重；无根时不能构造虚假的a或读取仿真真值。R与X分别是平方电压灵敏度矩阵，真值为共同路径标幺阻抗的两倍。

本轮实现是常方差近似和现有解析公共项消元：残差协方差Sigma=sigma_v²I+sigma_0²11ᵀ，对应现有n*gamma记号下gamma=sigma_v²/(n*sigma_0²)。R/X共用alpha，公共项gamma独立；逐时异方差、OU平滑和alpha_R/alpha_X分别调参属于可扩展内容，没有计入本轮已验证算法。

自由公共项允许R→R+u11ᵀ、X→X+v11ᵀ并由c抵消，因而共同干线绝对阻抗不唯一；但两两距离不变，不能据此断言所有相对拓扑都不可恢复。已知根分支归属和已知公共支路参数也能提供结构/规范锚点，根侧P/Q总量主要用于功率一致性及损耗约束。

变压器电压变化进入灵敏度估计已有相关研究，不能把“加入公共项”本身作为未经比较的新颖性主张。[Flynn等原作者机构页面](https://research.monash.edu/en/publications/an-improved-algorithm-for-topology-identification-of-distribution/)

## 3. 固定算例与消融协议

[运行前固定的协议](../../artifacts/root_information_study_20260908/study_protocol.json)保留本轮原始设计。所有新物理数据均为reference：四个合成馈线、线路倍率1、三种负荷/天气日型、根均值1.02 p.u.、背景尺度0.003 p.u.、有时间相关性。

| 实验 | 条件安排 | 目的 |
|---|---|---|
| 根信息研究 | 4网络×N16/32×3重复×2表计噪声×3根观测模式 | 比较c=0、自由c、双ridge、可用时的表计锚定 |
| 主消融 | 4网络×N8/16/32×3重复，共36条件 | 相同RNJ候选池下分离结构搜索、稳定块冻结和收缩 |
| 候选扩展对照 | 4网络×N16×3重复，共12条件 | 固定one-edit候选域，诊断缺失候选；不按主实验胜负挑网络 |
| 人工核查示例 | 预先固定paper15、N8、repeat0 | 从100棵完整bootstrap树生成核查问题，不用真值排序 |

reference中的根背景尺度0.003 p.u.是过程参数，不是强制每次实现的样本标准差。以主图flynn16的三个独立测试日为例，实际根电压标准差为0.002744、0.003543、0.003122 p.u.，约为对应根表计误差标准差的14–18倍；动态线路平方压降与目标量测扰动的RMS比为4.74、3.46、9.93。终端真实电压范围0.994385–1.019312 p.u.，全部物理点AC潮流收敛。**这组根波动不是近似恒定的小扰动，线路辨识信号也没有靠旧阻抗倍增来放大。** 0.1%噪声档的根模型结果用于检验退化；结构消融本身只覆盖0.02%噪声，不能直接推广到高噪声。

![reference的根与终端电压、公共变化与线路压降变化](../../artifacts/root_information_study_20260908/figures/reference_scenario_trace.png)

[物理逐点值](../../artifacts/root_information_study_20260908/figures/reference_scenario_trace_values.csv)和[波动统计/来源指纹](../../artifacts/root_information_study_20260908/figures/reference_scenario_trace_statistics.json)均来自已归档的独立测试96点网格，没有插值平滑或重新生成场景。图为合成场景的内部物理诊断，不代表某条真实馈线的实测分布。

每重复使用三个独立数据批次：train=3r、validation=3r+1、test=3r+2。训练与调参各3N行，独立测试288行；所以低数据预算必须写成6N可用于训练/选择，而不能只宣传3N。N从完整96点物理日抽取，场景/端噪声与根轨迹保持配对。

主消融方法包括：R-RNJ、X-RNJ、RX75-RNJ、固定RNJ树的L1重拟合、同一RX75几何且加入已知根点的经典NJ、全子集wzzT、共享有限池wzzT、稳定块冻结联合、稳定块收缩联合。经典NJ的短内部边收缩系数从预设0/.04/.08/.16按验证误差选择；它使用和RNJ相同的根深度信息，不能通过不给基线根信息来制造优势。

经典NJ采用全局Q准则，RNJ变体采用最大共享根路径合并。本轮是可复核的算法基线，并非Flynn、Pengwah等全部原文方法的完整复现。[Ni与Tatikonda](https://arxiv.org/abs/0809.0158)，[Saitou与Nei原文](https://doi.org/10.1093/oxfordjournals.molbev.a040454)

所有前向结构法的搜索预算一致：12秒总搜索、2秒单次求解，保留原最大结构规模与扩界机制；这不是包含bootstrap、回归和重拟合的整流程时限。有限池逐候选LP只有在全部候选均认证最优时才有单步证书；整条前向路径仍为贪心。全子集MILP可能在同一单步时限下只返回部分结果，必须单列，不能推广为无限求解时间下的精度比较。

收缩组严格映射原始one-edit候选池，排除切穿冻结组的候选，不在约化空间悄悄重新生成更宽的候选域。覆盖率同时报告原池、实际映射池及与冻结约束兼容的有效候选。真值只用于这些诊断计分。

不同结构方法最终都在相同原终端训练数据上固定结构重拟合R/X，使电压误差可比；原模型与原空间误差另存。验证选择还将联合树与完整RNJ基线比较，平局选RNJ；测试集不参与选择。拓扑正确与单独R/X系数精确是不同结论。

## 4. 分区指标与可视化口径

- 整体：全部非平凡终端clade的precision/recall/F1及完整恢复数。
- 根部：每棵树各自的极大proper clade；补齐未覆盖singleton形成根分区，再报分区完全恢复。
- 末端：每棵树各自的极小非平凡clade，以及每个终端最小包含clade的精确率，用来检查末端父分支归属。
- 估计：R与X分别报告相对Frobenius误差、距离误差、独立测试残差。
- 计算：预处理、bootstrap、结构搜索、原端重拟合分别计时；complete、partial、refit_failed和group_failed分别记录。

全终端共同干线不进入非平凡clade评分；图的隐藏节点不使用现场设备编号。每幅热图保存原始矩阵、端标签、显示顺序和逐点表，R与X各自共用同一色标，真实/数值零用灰黑表示。误差图保留零位置上的假非零估计。

代表图在全部结果完成后选取：验证选择后的最大改善、原始收缩结果的最大退化，以及无冻结候选法的全恢复/RX较准确示例。最后一项在全恢复且相对RNJ改善的条件中，最小化max(R误差,X误差)。这些都是明示的后验展示规则，没有用于调参，也不能取代全部48条件的统计。

## 5. 已完成的实验结果

### 5.1 根信息：用量测锚定公共方向

144个条件、528个最终模型、1872次QP全部完成，无模型失败。每格为4网络×3重复×N16/32共24个配对条件的平均RX75-RNJ F1；N嵌套，不是24个独立物理重复。

| 根信息/电压量测噪声 | 当前c=0 | 自由公共项 | 公共项与RX双ridge | 根表计锚定+RX ridge |
|---|---:|---:|---:|---:|
| 准确根，0.02% | 0.8875 | 0.7364 | 0.8825 | 0.8825 |
| 有噪根，0.02% | 0.8890 | 0.7364 | 0.8952 | 0.8945 |
| 无根量测，0.02% | 0.1535 | 0.7364 | **0.7966** | 不适用 |
| 准确根，0.10% | 0.4145 | 0.0504 | 0.4883 | 0.4842 |
| 有噪根，0.10% | 0.4299 | 0.0504 | 0.4588 | 0.4471 |
| 无根量测，0.10% | 0.1093 | 0.0504 | **0.4256** | 不适用 |

据此，准确根不应无条件放开公共项；噪根应把表计精度转换为公共方向权重；无根时双ridge比完全遗漏公共项有明显拓扑收益。后者仍不能识别任意共同干线绝对阻抗。自由公共项能减小某些同刻profile残差，却可能严重损害R/X及独立预测，不能以此认定模型更好。

效果随N不同：0.02%噪根时，锚定组N32的F1从0.9076提高到0.9439，但N16从0.8705降到0.8452。根表计锚定在这批数据中没有全面优于验证选gamma的双ridge。对应独立预测RMSE的配对平均改善只有0.32%和1.04%；无根双ridge为4.13%和5.27%，拓扑改善大于绝对电压预测改善。

![根信息、噪声与样本量的配对结果](../../artifacts/root_information_study_20260908/figures/root_information_comparison.png)

[根模型全量指标](../../artifacts/root_information_study_20260908/root_model/metrics.csv)、[分层均值与标准差](../../artifacts/root_information_study_20260908/root_model/aggregate.csv)、[详细数学和结果](root_information_model.md)。这组实验只比较回归与RNJ，没有把根锚定与后面的wzzT同时组合；两个实验的收益不能相加。

### 5.2 结构消融：最强的是RNJ候选+wzzT，不是全部冻结/收缩

主实验为事先固定的36个条件，以下均为等权均值。根/末端列分别使用各自区域clade F1，整树恢复要求全部非平凡clade集合完全一致。数据含有噪根观测，各方法使用同一数据和当前c=0回归。

| 方法 | 整体F1 | 根部F1 | 末端F1 | 整树完全恢复 | 总耗时均值/秒 |
|---|---:|---:|---:|---:|---:|
| 同根信息的经典NJ+验证收缩 | 0.7850 | 0.9222 | 0.4005 | 10/36 | 0.495 |
| R-RNJ | 0.7472 | 0.9722 | 0.3796 | 1/36 | 0.093 |
| X-RNJ | 0.2852 | 0.4058 | 0.0412 | 0/36 | 0.094 |
| RX75-RNJ | 0.8389 | 0.9861 | 0.5370 | 7/36 | 0.093 |
| RNJ固定树L1重拟合 | 0.8389 | 0.9861 | 0.5370 | 7/36 | 0.179 |
| **RNJ候选+wzzT，不冻结** | **0.9158** | **0.9861** | **0.8151** | **14/36** | **3.024** |
| RNJ稳定组冻结+wzzT | 0.8684 | 0.9792 | 0.7321 | 12/36 | 3.853 |
| RNJ稳定组收缩+wzzT | 0.8803 | 0.9792 | 0.7440 | 12/36 | 3.578 |

无冻结组的代码标识是`wzzt_shared_pool`。它使用RNJ产生的候选池，因而本身就是RNJ+wzzT混合方法，不是“不使用RNJ”的纯wzzT。相对RX75-RNJ，整体F1提高7.69个百分点，末端F1提高27.81个百分点，36条件中24胜、6平、6负；相对同根信息的经典NJ为24胜、4平、8负。末端父分支归属正确比例从0.7181提高到0.8475。根分区完全恢复均为35/36，说明优势主要来自纠正末端过细分，**本批没有额外提高根部均值**。

因此应把论文主线表述为“RNJ提供可计算的结构候选，wzzT利用数据选择层状结构”，把自动冻结/收缩作为有条件的加速选项。主消融为控制耗时仅用12次bootstrap，不能把这次低重复数下的冻结误判泛化为所有bootstrap实现都无效。被选中冻结块的真值精度为N8的0/8、N16的5/11、N32的16/17；低样本稳定并不保证正确。核实过的人工连接约束与数据内部的稳定组也不是同一种证据。

各N的无冻结组整体F1为0.8547、0.9270、0.9656；对应RNJ为0.7386、0.8705、0.9076。它在本批三个数据档均有平均收益，仍存在逐条件退化，且四个小网络与三次重复不构成统计显著性或普遍优势证明。

![全部主条件的分区、参数与时间指标](../../artifacts/root_information_study_20260908/figures/ablation_overview.png)

### 5.3 参数误差、验证选择与求解速度的边界

拓扑改善不保证所有数值指标都最好。RNJ输入矩阵的平均R/X相对误差为23.12%/61.71%，无冻结候选法为15.73%/41.64%；固定RNJ树重拟合为12.90%/38.12%，经典NJ为12.10%/37.36%。独立测试平方电压RMSE分别约为6.52、6.69、6.53、6.47乘10⁻⁴ pu²。**电压拟合更好不等价于拓扑更好**，不能用单一验证MAE来证明结构正确。

上述wzzT结果已经在各自搜索路径内使用验证集选择；若再把候选法与RNJ固定树进行一次验证MAE比较，无冻结候选法的F1从0.9158变为0.8511，对RNJ仅3胜、33平，整树恢复10/36。收缩联合方法经过同样额外选择后，主实验F1仅从RNJ的0.8389变为0.8438；N8和N16均回到与RNJ相同的拓扑。原始搜索结果的24胜不能当作额外回退策略实施后仍有24胜。扩大one-edit候选域的12个条件中，这个验证回退全部返回RNJ，未产生拓扑收益。它是保守的运行选择，不是有效的拓扑真值判别器。

全子集wzzT在本轮2秒单次求解预算内均未得到首步最优证书，主实验得到的可用部分路径是星形，非平凡clade F1为0。该行已保留在全量表和总图，但不能把它解释为“充分时间下纯MILP精度为0”。有限池法改变了求解规模；对比证明的是本轮严格预算下RNJ候选有实际作用。只要有一步未获证书，即保留partial/失败记录，不包装成全树全局最优。

全部48条件×9方法=432计划结果：364 complete、65 partial、3 search_failed，无丢失条件。3个失败均在one-edit补充实验，保存原始错误和RNJ回退；拓扑缺失时不以删除整行的方式改善均值。one-edit虽然真值候选覆盖达到1，仍会受候选数、冻结错误和前向搜索预算限制，覆盖率不是恢复保证。

![收缩联合相对RNJ的全部配对变化，包括验证回退](../../artifacts/root_information_study_20260908/figures/paired_hybrid_gains.png)

[432行原始结果](../../artifacts/root_information_study_20260908/ablation/metrics.csv)、[独立全条件汇总](../../artifacts/root_information_study_20260908/ablation/analysis/summary.md)、[机器可读汇总及有效分母](../../artifacts/root_information_study_20260908/ablation/analysis/summary.json)。图中的派生“验证回退”行复用已有拟合结果，不计为新增实验。

### 5.4 同时准确识别根部和末端的可追溯图例

**主图：flynn16，N32，repeat0，RNJ候选+wzzT，无冻结。** RNJ整体F1由0.8提高到1，根与末端F1均为1，原RNJ的3个多余clade全部消除，无缺失clade。最终R相对误差6.90%，X为23.43%，说明拓扑全恢复仍不等于X系数已经非常准确。

该图按“全恢复且相对RNJ改善的条件中，min max(R误差,X误差)”事后选作展示，规则见[selection.json](../../artifacts/root_information_study_20260908/figures/rnj_candidates_wzzt_example/selection.json)。完整结果仍以上面的固定网格为准。

![R与X：真值、RNJ输入与无冻结RNJ候选+wzzT估计](../../artifacts/root_information_study_20260908/figures/rnj_candidates_wzzt_example/rx_heatmaps.png)

![真树、RNJ和RNJ候选+wzzT的拓扑对照](../../artifacts/root_information_study_20260908/figures/rnj_candidates_wzzt_example/topology.png)

![按真值排序的R和X元素比较](../../artifacts/root_information_study_20260908/figures/rnj_candidates_wzzt_example/rx_sorted_entries.png)

热图按真实树DFS排列只是显示规则，算法没有读取该顺序。R和X各自统一色标，不把两种物理参数共用一个误导性的范围；零位置另标灰色，并在误差图中显示假非零。每图均提供SVG、逐元素表和NPZ，见[矩阵与标签](../../artifacts/root_information_study_20260908/figures/rnj_candidates_wzzt_example/rx_heatmaps.npz)、[逐元素值](../../artifacts/root_information_study_20260908/figures/rnj_candidates_wzzt_example/rx_heatmaps.csv)。拓扑边长是示意布局，不是现场距离。

另保留两个收缩方法图例：[验证选择后最大改善](../../artifacts/root_information_study_20260908/figures/largest_validated_improvement/topology.png)，soumalas11/N32/repeat1，F1为0.8889→1；[原始联合结果最大退化](../../artifacts/root_information_study_20260908/figures/largest_raw_regression/topology.png)，flynn16/N16/repeat2/one-edit，0.8571→0，验证回退恢复到RNJ的0.8571。不能只展示前者。

### 5.5 复现与核验

从`rnj_wzzt_core`运行，使用新的输出目录避免覆盖正式结果：

```powershell
python experiments/run_root_information_study.py --workers 2 --output outputs/root_information_replay
python experiments/run_rooted_ablation.py --include-one-edit --workers 2 --output outputs/rooted_ablation_replay
```

根研究入口为[run_root_information_study.py](../experiments/run_root_information_study.py)，结构消融入口为[run_rooted_ablation.py](../experiments/run_rooted_ablation.py)，预算/收缩适配为[rooted_ablation_support.py](../experiments/rooted_ablation_support.py)，经典NJ和固定树评价为[rooted_study_baselines.py](../experiments/rooted_study_baselines.py)。估计、图算法和数据生成均复用独立核心，生产默认未切换为实验胜出配置。

所有正式消融输入、候选、选择路径、求解状态、R/X及原端重拟合均保存；运行源码另有不可混用配置的指纹和快照。首轮报告字典写入错误保留在`ablation_attempt_01_reporting_bug`，修正后按原协议重跑48条件；该批不与正式成绩混用。人工查询示例取该批未受报告错误影响的输入，来源在生成审计中明确记录。

独立复核完成45,471项检查：432计划行、JSON/CSV一致性、输入恒等式、验证选择、拓扑分区、候选覆盖、858个R/X矩阵及31个源码/快照SHA；全部通过。完整测试为**289 passed、10 subtests passed**，记录见[final_tests.xml](../../artifacts/root_information_study_20260908/final_tests.xml)。这些检查确认本批实现和数据一致性，不代替方法总体正确性证明。


## 6. 先验信息与人工核查

完整的先验语义、接口和测试见[先验与人工核查设计](prior_acquisition_design.md)。二元exact cherry可通过initial_supports固定；完整多元末端组还需禁止内部clade；同根支路不是一个pair-clade；完整根分区还需排除跨组proper clade。未知可靠度的线索优先扩充候选或软约束，不能直接冻结。

对完整候选树集给经验质量p，令L为clade、根分区pair、exact-cherry差异的加权对称差，当前工具定义

\[
D(p)=\sum_{T,T'}p(T)p(T')L(T,T'),\qquad
\text{priority}(q)=\frac{D(p)-\sum_y p(y)D(p\mid y)}{\text{cost}(q)}.
\]

它是候选结构分歧下降，不是校准后验，也不是生产求解器重新拟合后的真实准确率收益。核查成本尚未提供时默认为1；多个问题可能诱导同一候选分割，应去重，并在每个回答后重排。

现场核查流程是：显示带末端标签的问题和yes/no代表树→记录答案、证据、时间和可靠度→检查先验冲突→更新候选/约束并重新拟合→重新排序。若回答排除了全部候选，应扩池或补量测，不能对空集合继续给置信度。人工实物边必须有稳定设备ID，RNJ的隐藏负编号不能发给现场当作真实接线点。

评估先验收益时，应排除直接告知和逻辑蕴含的结构，再报未告知部分恢复率；并另做错误先验、过期分区、无法判断和回滚对照。本轮实现了核查排序与实例，没有把尚未实施的现场核查或先验闭环恢复记成实验收益。

固定paper15/N8/repeat0的真实运行输入产生100棵完整bootstrap树、96种不同最简拓扑，排序器生成182个问题。去重后的[人工核查清单](../../artifacts/root_information_study_20260908/manual_query_demo/manual_checklist.md)首问是：“终端107、113是否位于根侧首个分叉后的同一支路？”其yes经验质量为0.21，成本暂设1，结构分歧下降代理为7.901；这个数不是21%的现场正确概率或预期提高7.901个百分点。

查询生成代码及审计见[脚本](../../artifacts/root_information_study_20260908/build_manual_query_demo.py)、[候选树与权重](../../artifacts/root_information_study_20260908/manual_query_demo/candidate_trees.json)、[生成审计](../../artifacts/root_information_study_20260908/manual_query_demo/generation_audit.json)。独立排序工具和14个针对性测试位于[rank_topology_queries.py](../../artifacts/prior_acquisition_design_20260908/rank_topology_queries.py)。实际获取人的回答、注入新事实和再拟合的完整闭环尚未执行，不计为本轮拓扑恢复收益。

## 7. 推荐的研究叙事

先证明/说明各类根信息能消除哪些自由度，再用根信息实验检验估计效果；之后用分区消融回答RNJ稳定组、候选范围和wzzT各自何时有贡献；最后把人工核查定位为候选缺失、根分区冲突和低样本不确定性的纠错闭环。

可信的结果应包含成功、持平和失败条件。四个小型合成网络与三次重复可以形成方法验证和清晰图例；尚不足以宣称跨网络普遍优于现有所有方法。三相不平衡、隐藏注入、根负荷闭环相关、实测表计偏差、更大网络和完整文献复现是另行验证的范围。
由现有证据支持的下一版方法结构是：**根量测锚定的R/X估计 → RNJ候选构造 → 不自动冻结的有限域wzzT → 分歧诊断 → 必要时人工核查**。只有被独立证实的结构才考虑硬固定/收缩。这里的“下一版”是研究方向：本轮分别验证了根模型与结构消融，尚未验证整个新组合，不能把两个模块的单独收益直接相加。
