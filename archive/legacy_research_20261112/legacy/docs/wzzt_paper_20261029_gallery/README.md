# WZZT数据记录与全面绘图

基于已完成的2026-10-29实验生成；本次未重跑估计器或求解器。模型结果共15500条、可用15358条，覆盖原1204个输入条件。五类结果通过cohort区分；补充处理不增加独立输入/重复数量。

## 浏览与取图

共36张图，其中六张为固定条件的规范拓扑示例。每图提供高分辨率PNG、矢量PDF及实际绘图数据CSV。完整PDF合集为[all_figures.pdf](all_figures.pdf)，离线图册为[index.html](index.html)，机器可读图目为[figure_index.csv](figure_index.csv)。

英文图注适合论文排版；中文说明在下列各图条目中。主统计区间沿用固定seed20261029的20000次bundle均值重采样；新树按树分块，图中逐点区间是描述性逐点区间。

## 数据与统计口径

- 原实验13500条主记录保持原LP15预算及原始缺失分母；13组LP60重试独立存储。
- 自由域80输入/160模型、两步扩展160输入/480模型、LP60重试13输入/156模型、过滤保护1204条分别归档。五类记录合计15500条，参数15358条。
- F1和exact在非平凡末端簇上计算；缺失F1保留为零。MAE、R/X误差和AC误差以实际有指标的可用模型为条件，CSV记录分母。
- 平方压降MAE单位p.u.^2；AC相对电压幅值RMSE原表为比例、图中乘100显示百分数。R/X为响应矩阵误差。
- 日内设计矩阵诊断使用每日96点；N24/48/96/288是同一三天的嵌套采样，独立重复按数据bundle计算。
- case33恢复视图保留N48原LP15与N288补充LP60的来源和预算；较大网络原始失败图仍保留。
- 过滤保护和两步扩展为次要/探索性结果。完整协议、原始失败及原54项交付核验位于[原报告](../wzzt_paper_20261029/README.md)。

## 记录与查询

model_records.csv.gz为长表。records.sqlite.zip包含可恢复数据库，模型、参数、输入路径、逐日诊断、AC拟合、搜索事件及计时通过键关联。数据库原文件在本机保留，Git中使用压缩归档。data_dictionary.csv给出字段和单位；methods.csv完整记录23个方法代码；method_settings.csv按cohort/variant记录35组算法配置与预算。

```python
from research_experiments.wzzt_paper.gallery_records import query, load_record
folder = r"D:\0-github_workspace\Topo\docs\wzzt_paper_20261029_gallery"
rows = query(folder, cohort="original", panel="main", case="paper15",
             voltage_sigma=.001, N=48, bundle=0, variant="ridge_guard0")
model = load_record(folder, rows.iloc[0].record_id, with_observations=True)
R, X = model["R"], model["X"]
data_config, method_config = model["configuration"], model["settings"]
# cohort="lp60_retry" 显式读取60秒重试；cohort=None才跨批查询。
```

SQL主要表：model_results、parameters、inputs、method_settings、methods、ac_fits、daily_diagnostics、physical_regimes、greedy_events、mip_events、timing、joint_diagnostics、filter_sensitivity；case33_recovered为带来源的恢复覆盖视图。主键record_id由cohort、job_id、variant构成；算法配置按cohort/variant关联。AC字段ac_key连接584条去重AC记录，模型表中共有876条AC选择关联。

原始观测、网络、checkpoint及求解器记录保留在outputs/wzzt_paper_20261029_full，哈希与路径位于本索引及source_manifest.json。仅查询参数、绘图和指标不依赖原始观测；with_observations或重绘真拓扑示例需要本机原始路径。

## 复现

```powershell
& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_paper.gallery docs/wzzt_paper_20261029 outputs/wzzt_paper_20261029_full docs/wzzt_paper_20261029_gallery
& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_paper.gallery --verify docs/wzzt_paper_20261029_gallery
```

PDF逐页检查使用装有pypdf/Pillow的Python运行 `python -m research_experiments.wzzt_paper.gallery_pdf_qa <directory>`；脚本调用Poppler渲染。检查或补充目视记录后，使用 `gallery --verify --seal <directory>` 更新交付哈希并复核。仅修改文字字典可用 `gallery --refresh-metadata <directory>`，不会重新生成图形数据。

源码副本与环境记录在source/和environment.json。manifest.json绑定每张图、绘图数据和数据库归档；verification.json记录数值、关联、分母和归档核验。visual_review.json在实际目视检查后写入。

## 图目

### 主结果

|图号|内容|文件|
|---|---|---|
|01|主面板：完整方法比较|[PNG](figures/01_main_methods.png) · [PDF](figures/01_main_methods.pdf) · [CSV](plot_data/01_main_methods.csv)|
|02|噪声 × 数据量：六种主要方法|[PNG](figures/02_noise_sample_size.png) · [PDF](figures/02_noise_sample_size.pdf) · [CSV](plot_data/02_noise_sample_size.csv)|
|03|逐网络的噪声与数据量曲线|[PNG](figures/03_each_network.png) · [PDF](figures/03_each_network.pdf) · [CSV](plot_data/03_each_network.csv)|
|04|各方法F1全条件矩阵|[PNG](figures/04_F1_heatmaps.png) · [PDF](figures/04_F1_heatmaps.pdf) · [CSV](plot_data/04_F1_heatmaps.csv)|
|05|各方法exact全条件矩阵|[PNG](figures/05_exact_heatmaps.png) · [PDF](figures/05_exact_heatmaps.pdf) · [CSV](plot_data/05_exact_heatmaps.csv)|
|07|精确率、召回率与采样量|[PNG](figures/07_precision_recall.png) · [PDF](figures/07_precision_recall.pdf) · [CSV](plot_data/07_precision_recall.csv)|

### 组件分析

|图号|内容|文件|
|---|---|---|
|06|配对增量：多初值、扩池、删除|[PNG](figures/06_component_gain_maps.png) · [PDF](figures/06_component_gain_maps.pdf) · [CSV](plot_data/06_component_gain_maps.csv)|
|08|每个独立数据重复上的配对收益|[PNG](figures/08_paired_bundle_effects.png) · [PDF](figures/08_paired_bundle_effects.pdf) · [CSV](plot_data/08_paired_bundle_effects.csv)|
|18|预定消融：正则化、预算、删除与候选库|[PNG](figures/18_predeclared_ablations.png) · [PDF](figures/18_predeclared_ablations.pdf) · [CSV](plot_data/18_predeclared_ablations.csv)|
|19|十bundle两步扩展的收益与分布|[PNG](figures/19_joint_extension.png) · [PDF](figures/19_joint_extension.pdf) · [CSV](plot_data/19_joint_extension.csv)|
|20|最终过滤阈值的结构与预测影响|[PNG](figures/20_filter_sensitivity.png) · [PDF](figures/20_filter_sensitivity.pdf) · [CSV](plot_data/20_filter_sensitivity.csv)|
|21|验证过滤保护：预测与拓扑的不同变化|[PNG](figures/21_filter_guard_tradeoff.png) · [PDF](figures/21_filter_guard_tradeoff.pdf) · [CSV](plot_data/21_filter_guard_tradeoff.csv)|

### 预测与AC

|图号|内容|文件|
|---|---|---|
|09|独立测试日的平方压降误差|[PNG](figures/09_linear_test_error.png) · [PDF](figures/09_linear_test_error.pdf) · [CSV](plot_data/09_linear_test_error.csv)|
|10|R/X响应矩阵误差|[PNG](figures/10_response_errors.png) · [PDF](figures/10_response_errors.pdf) · [CSV](plot_data/10_response_errors.csv)|
|28|AC重估后的独立测试电压误差|[PNG](figures/28_AC_test_prediction.png) · [PDF](figures/28_AC_test_prediction.pdf) · [CSV](plot_data/28_AC_test_prediction.csv)|
|29|AC残差与拓扑F1的对应关系|[PNG](figures/29_AC_error_topology.png) · [PDF](figures/29_AC_error_topology.pdf) · [CSV](plot_data/29_AC_error_topology.csv)|

### 场景与泛化

|图号|内容|文件|
|---|---|---|
|11|十类场景 × 两档数据量|[PNG](figures/11_stress_conditions.png) · [PDF](figures/11_stress_conditions.pdf) · [CSV](plot_data/11_stress_conditions.csv)|
|12|各场景相对多初值的收益|[PNG](figures/12_stress_gains.png) · [PDF](figures/12_stress_gains.pdf) · [CSV](plot_data/12_stress_gains.csv)|
|13|新拓扑的规模、噪声与恢复性能|[PNG](figures/13_unseen_size.png) · [PDF](figures/13_unseen_size.pdf) · [CSV](plot_data/13_unseen_size.csv)|
|14|十二棵新树逐一比较|[PNG](figures/14_each_unseen_tree.png) · [PDF](figures/14_each_unseen_tree.pdf) · [CSV](plot_data/14_each_unseen_tree.csv)|
|15|case33恢复覆盖：逐行注明预算|[PNG](figures/15_case33_recovered.png) · [PDF](figures/15_case33_recovered.pdf) · [CSV](plot_data/15_case33_recovered.csv)|

### 物理数据

|图号|内容|文件|
|---|---|---|
|16|实际AC电压与压降尺度|[PNG](figures/16_physical_voltage.png) · [PDF](figures/16_physical_voltage.pdf) · [CSV](plot_data/16_physical_voltage.csv)|
|17|激励条件、信号与计量误差|[PNG](figures/17_excitation_meter_noise.png) · [PDF](figures/17_excitation_meter_noise.pdf) · [CSV](plot_data/17_excitation_meter_noise.csv)|

### 失败与求解

|图号|内容|文件|
|---|---|---|
|22|候选覆盖、错误交叉与过细分|[PNG](figures/22_failure_diagnostics.png) · [PDF](figures/22_failure_diagnostics.pdf) · [CSV](plot_data/22_failure_diagnostics.csv)|
|23|完整扫描中的一步训练目标偏好|[PNG](figures/23_greedy_objective.png) · [PDF](figures/23_greedy_objective.pdf) · [CSV](plot_data/23_greedy_objective.csv)|
|24|有限池局部停止、预算与不可用记录|[PNG](figures/24_finite_pool_stops.png) · [PDF](figures/24_finite_pool_stops.pdf) · [CSV](plot_data/24_finite_pool_stops.csv)|
|25|自由域策略与原生MILP gap|[PNG](figures/25_free_domain_and_gap.png) · [PDF](figures/25_free_domain_and_gap.pdf) · [CSV](plot_data/25_free_domain_and_gap.csv)|
|26|原共享进程缺失与LP60恢复|[PNG](figures/26_LP60_recovery.png) · [PDF](figures/26_LP60_recovery.pdf) · [CSV](plot_data/26_LP60_recovery.csv)|

### 成本

|图号|内容|文件|
|---|---|---|
|27|独立运行成本与配对精度|[PNG](figures/27_runtime_accuracy.png) · [PDF](figures/27_runtime_accuracy.pdf) · [CSV](plot_data/27_runtime_accuracy.csv)|
|30|AC计算成本和局部收敛|[PNG](figures/30_AC_cost_convergence.png) · [PDF](figures/30_AC_cost_convergence.pdf) · [CSV](plot_data/30_AC_cost_convergence.csv)|

### 拓扑示例

|图号|内容|文件|
|---|---|---|
|31|拓扑示例：paper15|[PNG](figures/31_topology_paper15.png) · [PDF](figures/31_topology_paper15.pdf) · [CSV](plot_data/31_topology_paper15.csv)|
|32|拓扑示例：soumalas11|[PNG](figures/32_topology_soumalas11.png) · [PDF](figures/32_topology_soumalas11.pdf) · [CSV](plot_data/32_topology_soumalas11.csv)|
|33|拓扑示例：flynn16|[PNG](figures/33_topology_flynn16.png) · [PDF](figures/33_topology_flynn16.pdf) · [CSV](plot_data/33_topology_flynn16.csv)|
|34|拓扑示例：pengwah18|[PNG](figures/34_topology_pengwah18.png) · [PDF](figures/34_topology_pengwah18.pdf) · [CSV](plot_data/34_topology_pengwah18.csv)|
|35|拓扑示例：random32_t1|[PNG](figures/35_topology_random32_t1.png) · [PDF](figures/35_topology_random32_t1.pdf) · [CSV](plot_data/35_topology_random32_t1.csv)|
|36|拓扑示例：case33_terminal|[PNG](figures/36_topology_case33_terminal.png) · [PDF](figures/36_topology_case33_terminal.pdf) · [CSV](plot_data/36_topology_case33_terminal.csv)|

### 01_main_methods 主面板：完整方法比较

F1与exact均针对非平凡末端簇集合。误差棒按十个bundle均值重采样，显示逐项95%区间；640条件共享四个固定网络。

![主面板：完整方法比较](figures/01_main_methods.png)

### 02_noise_sample_size 噪声 × 数据量：六种主要方法

每点四网络×十bundle；区间按bundle分块。N为三个物理日的嵌套采样总量，V=0仍保留P/Q噪声0.5%。

![噪声 × 数据量：六种主要方法](figures/02_noise_sample_size.png)

### 03_each_network 逐网络的噪声与数据量曲线

每个小图固定网络和噪声，每点十个独立数据bundle；区间逐点计算。

![逐网络的噪声与数据量曲线](figures/03_each_network.png)

### 04_F1_heatmaps 各方法F1全条件矩阵

每格40输入，按四固定网络×十数据bundle平均；原分母保留。图中数字为均值，逐点区间见绘图CSV。

![各方法F1全条件矩阵](figures/04_F1_heatmaps.png)

### 05_exact_heatmaps 各方法exact全条件矩阵

每格40输入，按四固定网络×十数据bundle平均；原分母保留。图中数字为均值，逐点区间见绘图CSV。

![各方法exact全条件矩阵](figures/05_exact_heatmaps.png)

### 06_component_gain_maps 配对增量：多初值、扩池、删除

同一输入上先相减，再按十bundle平均。各小图使用独立对称色标，数字给出实际效应量。

![配对增量：多初值、扩池、删除](figures/06_component_gain_maps.png)

### 07_precision_recall 精确率、召回率与采样量

每个点为固定噪声/N下四网络×十bundle的均值。线连接同一方法的N24→48→96→288；标记由小到大对应这四个N，用于观察漏簇与过细分。

![精确率、召回率与采样量](figures/07_precision_recall.png)

### 08_paired_bundle_effects 每个独立数据重复上的配对收益

每点一个bundle，先在该bundle内平均全部64个网络/噪声/N条件。黑色点为均值与95%区间。原三主检验的Holm结果保留于原报告。

![每个独立数据重复上的配对收益](figures/08_paired_bundle_effects.png)

### 09_linear_test_error 独立测试日的平方压降误差

固定训练参数后评价独立测试日；纵轴为平方压降MAE，单位p.u.^2。区间按十bundle；误差统计以可用模型为条件。

![独立测试日的平方压降误差](figures/09_linear_test_error.png)

### 10_response_errors R/X响应矩阵误差

相对Frobenius误差；R/X是平方压降响应矩阵，不是逐条线路阻抗误差。每点四网络×十bundle。

![R/X响应矩阵误差](figures/10_response_errors.png)

### 11_stress_conditions 十类场景 × 两档数据量

每个场景/N含四网络×五bundle；区间按五bundle。所有方法使用同一条件数据。

![十类场景 × 两档数据量](figures/11_stress_conditions.png)

### 12_stress_gains 各场景相对多初值的收益

先对相同输入配对相减；原池追加修正控制额外搜索预算。颜色统一，完整五bundle区间在CSV中。

![各场景相对多初值的收益](figures/12_stress_gains.png)

### 13_unseen_size 新拓扑的规模、噪声与恢复性能

每档三棵新树，按树均值重采样；共享三个数据bundle和N48/288。原LP15缺失保留为零分，同时CSV记录可用数。

![新拓扑的规模、噪声与恢复性能](figures/13_unseen_size.png)

### 14_each_unseen_tree 十二棵新树逐一比较

各格平均该树的三个数据bundle、两噪声与两个N；逐树展示，避免总体均值掩盖网络差异。

![十二棵新树逐一比较](figures/14_each_unseen_tree.png)

### 15_case33_recovered case33恢复覆盖：逐行注明预算

同一棵端点化case33，五个数据bundle。N48为原LP15，N288为独立LP60重试；预算不同，数据量效应与预算不作单因素因果分解。过滤保护为探索性输出。

![case33恢复覆盖：逐行注明预算](figures/15_case33_recovered.png)

### 16_physical_voltage 实际AC电压与压降尺度

无噪声物理数据，合并训练/验证/测试九日。左侧为每bundle分位数再平均的描述性范围，右侧为全体极值；均非置信区间。

![实际AC电压与压降尺度](figures/16_physical_voltage.png)

### 17_excitation_meter_noise 激励条件、信号与计量误差

每个点先汇总同一网络/bundle的三个训练日；原诊断使用每日96点，不是训练N子集。此图为描述性分布，未把日内快照当独立重复。

![激励条件、信号与计量误差](figures/17_excitation_meter_noise.png)

### 18_predeclared_ablations 预定消融：正则化、预算、删除与候选库

每行仅与自身匹配输入上的参考比较，子面板分母不同。区间按各自bundle；原单bundle两步结果保留在数据索引，十bundle扩展另图展示。

![预定消融：正则化、预算、删除与候选库](figures/18_predeclared_ablations.png)

### 19_joint_extension 十bundle两步扩展的收益与分布

次要扩展复用原主面板数据，固定父模型库与算法。每点一个bundle；并列展示稀疏收益与退化计数。完整区间、符号检验及最终来源保留在数据库/原表。

![十bundle两步扩展的收益与分布](figures/19_joint_extension.png)

### 20_filter_sensitivity 最终过滤阈值的结构与预测影响

固定同一收缩前预选模型，仅改变最终阈值；未重新拟合或搜索。F1保留全部输入分母，MAE按可用模型统计；五阈值属于探索性诊断。

![最终过滤阈值的结构与预测影响](figures/20_filter_sensitivity.png)

### 21_filter_guard_tradeoff 验证过滤保护：预测与拓扑的不同变化

同一预选模型上验证误差不增，拓扑F1仍可能下降。图只展示可用的1191个模型；原13条不可用记录仍在模型总表。新树/case33的预先冻结补充检查标记在prospective_filter_holdout字段。

![验证过滤保护：预测与拓扑的不同变化](figures/21_filter_guard_tradeoff.png)

### 22_failure_diagnostics 候选覆盖、错误交叉与过细分

前三项以可用模型为分母，unavailable以全部计划输入为分母。过细分指非平凡预测簇多于真簇；最终交叉是结构诊断，时间先后证据另见逐步事件。

![候选覆盖、错误交叉与过细分](figures/22_failure_diagnostics.png)

### 23_greedy_objective 完整扫描中的一步训练目标偏好

仅保留完整有限池扫描且测过真簇提案的错误正权簇接受事件。严格偏好阈值1e-8；多个事件可属于同一输入，不作为独立重复或长期阻挡的充分证据。

![完整扫描中的一步训练目标偏好](figures/23_greedy_objective.png)

### 24_finite_pool_stops 有限池局部停止、预算与不可用记录

局部停止针对当前有限候选邻域；不可用是共享工作进程阶段未生成记录，不直接归因于各算法不可行。

![有限池局部停止、预算与不可用记录](figures/24_finite_pool_stops.png)

### 25_free_domain_and_gap 自由域策略与原生MILP gap

左侧80匹配输入、五bundle区间。自由域100秒路径采用兼容簇追加，有限池25秒修正允许交叉替换；比较完整策略。右侧626次调用，gap属于单步模型，状态2按本轮1%容差终止。

![自由域策略与原生MILP gap](figures/25_free_domain_and_gap.png)

### 26_LP60_recovery 原共享进程缺失与LP60恢复

十三组原LP15触时输入均以独立LP60重试恢复；左侧可用性包含原checkpoint已有结果。不同预算分别保留，原触时不被覆盖。

![原共享进程缺失与LP60恢复](figures/26_LP60_recovery.png)

### 27_runtime_accuracy 独立运行成本与配对精度

每方法16个相同输入，各用新进程顺序重放。CPU/Gurobi环境见原hardware.json；不含AC后处理。右侧为固定bundle0的描述性成本/精度，不将16条件当独立重复。

![独立运行成本与配对精度](figures/27_runtime_accuracy.png)

### 28_AC_test_prediction AC重估后的独立测试电压误差

只用训练数据重估线参数，冻结后评价独立测试日。纵轴为相对电压幅值RMSE百分数；按十bundle计算逐点区间。与平方压降MAE分开解释。

![AC重估后的独立测试电压误差](figures/28_AC_test_prediction.png)

### 29_AC_error_topology AC残差与拓扑F1的对应关系

原批次580个去重的输入/参数组合；同一模型被多个方法选中时只画一次。标记区分局部参数拟合收敛与否，低残差作为预测指标。

![AC残差与拓扑F1的对应关系](figures/29_AC_error_topology.png)

### 30_AC_cost_convergence AC计算成本和局部收敛

max_nfev=12约束求解器的主函数评价计数，数值Jacobian额外潮流调用另计。图用实际seconds记录成本，580个原批去重模型均保留。

![AC计算成本和局部收敛](figures/30_AC_cost_convergence.png)

### 31_topology_paper15 拓扑示例：paper15

示例按网络、bundle0、固定噪声与N选取，不按结果优劣筛选。图为非平凡簇的规范层级，压缩根/叶的平凡结构；边长只作布局。case33为LP60补充，过滤保护为探索性方法。

![拓扑示例：paper15](figures/31_topology_paper15.png)

### 32_topology_soumalas11 拓扑示例：soumalas11

示例按网络、bundle0、固定噪声与N选取，不按结果优劣筛选。图为非平凡簇的规范层级，压缩根/叶的平凡结构；边长只作布局。case33为LP60补充，过滤保护为探索性方法。

![拓扑示例：soumalas11](figures/32_topology_soumalas11.png)

### 33_topology_flynn16 拓扑示例：flynn16

示例按网络、bundle0、固定噪声与N选取，不按结果优劣筛选。图为非平凡簇的规范层级，压缩根/叶的平凡结构；边长只作布局。case33为LP60补充，过滤保护为探索性方法。

![拓扑示例：flynn16](figures/33_topology_flynn16.png)

### 34_topology_pengwah18 拓扑示例：pengwah18

示例按网络、bundle0、固定噪声与N选取，不按结果优劣筛选。图为非平凡簇的规范层级，压缩根/叶的平凡结构；边长只作布局。case33为LP60补充，过滤保护为探索性方法。

![拓扑示例：pengwah18](figures/34_topology_pengwah18.png)

### 35_topology_random32_t1 拓扑示例：random32_t1

示例按网络、bundle0、固定噪声与N选取，不按结果优劣筛选。图为非平凡簇的规范层级，压缩根/叶的平凡结构；边长只作布局。case33为LP60补充，过滤保护为探索性方法。

![拓扑示例：random32_t1](figures/35_topology_random32_t1.png)

### 36_topology_case33_terminal 拓扑示例：case33_terminal

示例按网络、bundle0、固定噪声与N选取，不按结果优劣筛选。图为非平凡簇的规范层级，压缩根/叶的平凡结构；边长只作布局。case33为LP60补充，过滤保护为探索性方法。

![拓扑示例：case33_terminal](figures/36_topology_case33_terminal.png)
