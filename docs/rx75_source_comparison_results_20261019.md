# RX75、RNJ初值与Source：完整30项结果

2026-10-19。**找到可展示优势的算例：soumalas11、replicate_2、rank2_source、288点全部拟合。完整RNJ树普通初值加WZZT的有效拓扑F1为1，NJ/RNJ均为0.8889，纯WZZT与严格末端组版联合方法均为0.6667。** 这里的完整树初值是预先登记的独立消融，不替换原来的四方法结果。

30/30项运行结束，均为complete；18项含WZZT的方法均以认证的`no_significant_one_atom_gain`停止，无求解超时、未认证求解或扩界。独立保存证据审计 **33694/33694通过**。完整表、路径分数、失败分母和协议在[结果包](/D:/0-github_workspace/Topo/outputs/rx75_source_comparison_20261018/reports_final/README.md)；[独立审计](/D:/0-github_workspace/Topo/outputs/rx75_source_comparison_20261018/audits/independent_final_20261019.json)。complete描述本次有界贪心流程结束，单次求解证书覆盖当前单原子扩展域。

## 数据和设置

- 原始soumalas11网络及线路阻抗不变。三次重复固定replicate=0/1/2，每次采用三个reference场景，各96点，共288点；全部拟合，验证集和独立测试集均为0。
- 每次配对无源和有源，共用背景负荷、根电压及噪声抽样。P/Q相对噪声0.005，终端和根电压相对噪声0.001，主表P/Q噪声0.002。这是预先登记的电压噪声压力设置。
- 已知Source的精确P/Q时序，未知位置在三个场景中共享。物理仿真源接母线5，位于1→3→5路径下游；位置只供仿真与事后评价。
- 源平均P=19.1kW，占全网终端名牌Pd的25%；峰值P约34.78kW、Q约11.36kvar、S约36.30kVA。P/Q非比例且非负；同一96点曲线在三个场景重复。它表示可调P/Q聚合额外消费。全部18个AC场景、1728个时点收敛。
- 保留100次bootstrap、block length=4、频率阈值0.75、最大2组、最小组大小3、子簇频率上限0.1、fine tolerance ratio=0.5。seed为20260903+replicate。
- 保留RX75的R/X比例0.75/0.25、同簇尺度0.16、原改善阈值和剪枝、系数上界2/2及原扩界条件；`max_atoms=2*n-1=21`。单次求解6000秒、方法总预算30000秒，solver/BLAS各1线程。
- 两个新worker与旧pengwah18长时限批次同时运行。下表为本次非独占机器上的实测计算时间，包含相应方法的回归、bootstrap和搜索。

预登记的[设计与方法协议](/D:/0-github_workspace/Topo/docs/wzzt_source_advantage_protocol_20261018.md)在任何新数据拟合之前固定并随运行复制；[协议JSON](/D:/0-github_workspace/Topo/outputs/rx75_source_comparison_20261018/protocol.json)的SHA为`927593d0617983a723f9608ff4e3cb9a5d8b947de2571668004dd3f7e067142c`，拟合代码从其冻结source_snapshot实际导入。

## 展示算例：第3次有源重复

R/X误差均指响应矩阵相对Frobenius误差，不是逐条物理线路的平均误差。拓扑主指标为固定RX75尺度下有效非平凡簇与完整物理真簇的F1。真树未按阈值收缩。

| 方法 | 有效拓扑F1 | R误差 | X误差 | 计算秒 |
|---|---:|---:|---:|---:|
| NJ | 0.8889 | 28.16% | 88.41% | 0.074 |
| RNJ | 0.8889 | 28.16% | 88.41% | 0.390 |
| 仅WZZT | 0.6667 | 39.12% | 80.56% | 394.99 |
| RNJ+WZZT，严格末端组 | 0.6667 | 39.12% | 80.56% | 413.82 |
| **RNJ+WZZT，完整树普通初值** | **1.0000** | **18.18%** | **83.54%** | **69.56** |

该初始化有11个叶原子、6个普通非平凡簇；严格atomic组为空。联合拟合把两个假簇[101,103]、[109,111]的R/X权重置零，留下4个真实活跃簇。故family F1仍为0.8，active/resolution F1为1；零权假支撑未被隐藏，完整记录保留。

本例`path_length=1`、`selected_path_index=0`：**有效拓扑改善来自完整RNJ初始家族的固定支持、同目标L1/Source联合重拟合及零权收缩，后续扩展没有被接受。** 它验证了RNJ能提供有用初始家族，也揭示了从叶节点贪心搜索的路径依赖。此项不构成“新增支撑搜索进一步改善”的证据。

本例固定校准为`a=4.2621916396,b=0.9772173484,tau=0.0952529528`。未知Source路径准确恢复两条真祖先簇[104,105,106,107,108]和[106,107,108]；gR/gX相对误差为1.335%/3.219%，非源支路响应泄漏为0。全网X误差仍大；源路径响应准确与全网阻抗精度分别评价。

## 三次重复全部保留

以下均为有效尺度F1；严格末端组版与纯WZZT在全部六个工况中结果相同，因为严格筛选均返回空组。完整树初值将普通簇与atomic组分开传递。

| 工况 | 方法 | rep0 | rep1 | rep2 | 平均 |
|---|---|---:|---:|---:|---:|
| 无源 | NJ | 0.8889 | 0.6000 | 0.8000 | 0.7630 |
| 无源 | RNJ | 0.8000 | 0.6000 | 0.8889 | 0.7630 |
| 无源 | WZZT / 严格组联合方法 | 0.7273 | 0.7273 | 0.5455 | 0.6667 |
| 无源 | 完整树初值联合方法 | 0.7273 | 0.8889 | 0.8889 | 0.8350 |
| 有源 | NJ | 0.8889 | 0.6000 | 0.8889 | 0.7926 |
| 有源 | RNJ | 0.8000 | 0.6000 | 0.8889 | 0.7630 |
| 有源 | WZZT / 严格组联合方法 | 0.8000 | 0.8889 | 0.6667 | 0.7852 |
| 有源 | 完整树初值联合方法 | 0.8000 | 0.8889 | 1.0000 | **0.8963** |

完整树初值联合方法在有源组分别对NJ、RNJ取得2/3次严格胜出，平均F1差为+0.1037、+0.1333，达到预登记条件。第1次重复仍劣于NJ；结论限定本网络、本噪声/源设计、三次重复。纯WZZT及严格末端组版仅1/3次胜出，未达到该条件。

同一重复无源→有源，纯WZZT与严格组联合方法的F1均提高；完整树初值版为0.7273→0.8000、0.8889→0.8889、0.8889→1.0000。完整树版平均提升0.0613，额外Source在这一设计下有帮助。

| 工况 | 方法 | 平均F1 | 平均R误差 | 平均X误差 | 平均计算秒 |
|---|---|---:|---:|---:|---:|
| 无源 | NJ | 0.7630 | 34.84% | 81.82% | 0.193 |
| 无源 | RNJ | 0.7630 | 34.84% | 81.82% | 0.379 |
| 无源 | WZZT | 0.6667 | 27.52% | 78.14% | 319.02 |
| 无源 | 严格组联合方法 | 0.6667 | 27.52% | 78.14% | 334.54 |
| 无源 | 完整树初值联合方法 | 0.8350 | 21.86% | 78.42% | 79.52 |
| 有源 | NJ | 0.7926 | 32.85% | 83.38% | 0.148 |
| 有源 | RNJ | 0.7630 | 32.85% | 83.38% | 0.347 |
| 有源 | WZZT | 0.7852 | 25.33% | 71.98% | 448.15 |
| 有源 | 严格组联合方法 | 0.7852 | 25.33% | 71.98% | 465.37 |
| 有源 | 完整树初值联合方法 | **0.8963** | **18.35%** | 72.98% | **112.31** |

完整树初值版的有源平均耗时约为纯WZZT的1/4。NJ/RNJ仍快得多。完整树版的X误差也略高于纯WZZT；各指标的取舍完整保留。

NJ/RNJ的R/X来自前置ordered/FWL dense回归，WZZT来自结构化L1原子拟合，因此参数误差差别同时包含模型域与拟合目标影响。树初值消融已单列path0和最终点以检查搜索的增量；未运行独立固定NJ树L1控制，不把全部参数收益归给WZZT新增边搜索。

## 对实现与历史结果的解释

1. **RX75短边尺度应进入方法，而不只用于NJ/RNJ。** 本轮以同输入的前置FWL矩阵一次校准`tau=0.16*median(diag(aR+bX))`，整条路径共用；旧内部边可为零或达到tau，新内部边必须达到tau，叶边按原图收缩语义保留。它是归一化内边分辨率。达到tau的伪边仍可能被训练目标接受，第1次重复已出现该情况。
2. **严格选组为空时，原联合方法没有获得非平凡RNJ初值。** 本轮六组全部为空；传参和标签映射正确，保守筛选将它们排除了。完整树普通支持与不可拆分的bootstrap末端组用途不同。消融显示，提供完整普通家族能显著改变恢复和时间；普通零权支持仍参与层状相容约束，错误初始家族的影响尚需保留警惕。
3. **及时停止与正确恢复分别评价。** 本轮所有含WZZT运行均在认证的单原子域无显著改善时停止，仍有不利F1；纯WZZT的贪心路径与初始化有关。完整树初值也有path0较好、随后新增伪簇使F1下降的重复。未按物理真值挑选路径点。
4. **Source的结构信息有条件增益。** NJ/RNJ对已知U同时投影P/Q和Y，消去自由终端响应G，并未忽略源。WZZT把G约束到同树完整路径。当前FWL设计仍满秩，弱方向受输入误差影响；源直接激励两条祖先边，其他支路仍可能辨识不足。详见[机制与两叶反例](/D:/0-github_workspace/Topo/docs/source_information_diagnostic_20261019.md)。

部分保存的`regression_diagnostics.source_present=false/source_design_rank=0`描述投影后的内层ordered QP；同一记录的外层`source_rank=2`、`source_model=unconstrained_terminal_response_FWL`及保存的U说明Source已在FWL层消元。实际288行P/Q/U对齐、自由G回算和源秩由独立审计核对。字段需按调用层次读取。

全部9项结构化有源结果均包含两条真源祖先簇；rep0/rep2的6项源路径准确，rep1的3项额外延伸到106叶节点。同一重复三种WZZT得到相同g。故rep2完整树初值相对纯WZZT的额外拓扑收益发生在非Source支路，不能归给更准确的源定位。

## Pengwah18原始批次与已修复原因

原始pengwah18无源联合路径首次F1=1后，继续增加8个短伪簇，最后F1=0.6364；按固定NJ校准的既有尺度收缩并实际重拟合后，F1=1，R/X误差5.74%/18.23%→4.25%/13.65%，独立193项检查通过。它是单独的后处理实验，原始结果保留。详见[路径、上界和重拟合诊断](/D:/0-github_workspace/Topo/docs/rx75_pengwah18_diagnosis_20261018.md)。

截至本报告，原16项长时限批次已终结11项：9 complete、2 incomplete；另2项运行、3项未启动。常值源及间歇源联合方法均在6000秒扩展求解时限后未获最优认证，gap分别36.63%、50.37%，保留最后认证星形模型和未接受incumbent的raw NPZ。两项分别925项独立检查通过。它们属于超时，与无源已认证路径过细化分开统计。[当前11项快照](/D:/0-github_workspace/Topo/outputs/source_ac288_benchmark_20261018/reports_checkpoint_11/README.md)。

原批次保持运行，已接上一段只读收尾流程：收到完整16项结束标记后自动执行完整审计、汇总到`reports_final`并递归保存raw SHA；不重试估计器或修改方法。当前[收尾状态](/D:/0-github_workspace/Topo/tmp/finalize_source_ac288_state_20261019/state.json)仍为等待原批次。尚未完成的5项未算作完成结果。

## 文件和复现

- [30行逐项结果](/D:/0-github_workspace/Topo/outputs/rx75_source_comparison_20261018/reports_final/per_job.csv)、[聚合](/D:/0-github_workspace/Topo/outputs/rx75_source_comparison_20261018/reports_final/aggregate.csv)、[path0对照](/D:/0-github_workspace/Topo/outputs/rx75_source_comparison_20261018/reports_final/tree_init_path0_vs_final.csv)。
- [运行器](/D:/0-github_workspace/Topo/research_experiments/rnj/run_rx75_source_comparison.py)、[RX75域适配](/D:/0-github_workspace/Topo/research_experiments/rnj/rx75_edge_floor.py)、[独立审计器](/D:/0-github_workspace/Topo/research_experiments/rnj/audit_rx75_source_comparison.py)。实验使用的具体源码随输出source_snapshot保存，拟合时Git版本为`7f84c029`。
- [原始数据配置](/D:/0-github_workspace/Topo/outputs/source_advantage_soumalas11_20261018/run_config.json)、[源功率预算](/D:/0-github_workspace/Topo/outputs/source_advantage_soumalas11_20261018/source_budget.csv)、[数据manifest](/D:/0-github_workspace/Topo/outputs/source_advantage_soumalas11_20261018/manifest.json)。
- [Source物理路径诊断](/D:/0-github_workspace/Topo/outputs/rx75_source_comparison_20261018/audits/supplemental/rx75_final_source_path_diagnostic_20261019.json)、[1728个AC时点的独立数据审计](/D:/0-github_workspace/Topo/outputs/rx75_source_comparison_20261018/audits/supplemental/source_advantage_data_independent_audit_20261018.json)、[输入谱与AC偏差诊断](/D:/0-github_workspace/Topo/outputs/rx75_source_comparison_20261018/audits/supplemental/source_advantage_fwl_rank_diagnostic_20261019.json)。复算脚本与汇总脚本随supplemental及postprocessor_source归档。
- 全递归SHA清单：[30项原始运行](/D:/0-github_workspace/Topo/outputs/rx75_source_comparison_20261018/reports_final/raw_evidence_sha256.json)、[SourceAdvantage原数据](/D:/0-github_workspace/Topo/outputs/source_advantage_soumalas11_20261018/reports_final/raw_evidence_sha256.json)、[pengwah18收缩重拟合](/D:/0-github_workspace/Topo/outputs/source_ac288_rx75_refit_20261018/no_source_hybrid/reports_final/raw_evidence_sha256.json)。共2255文件、96,918,797字节；覆盖solver NPZ、progress、checkpoint、输入与源码/producer快照，排除各报告目录及清单自身。归档前2246份既有raw文件的SHA、大小及mtime均未改变。

运行器17项、RX75适配13项、独立审计器15项、收缩重拟合6项测试已通过；测试覆盖实际小型有/无源求解、源路径重新拟合、原始约束审计、冻结闭包、checkpoint及预算回退。正式30项原始结果和模型保持不变。
