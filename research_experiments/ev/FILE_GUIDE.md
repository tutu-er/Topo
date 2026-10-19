# 文件作用、输入与输出

以下位置相对于[封存源码根](../../archive/ev_methods_20261019/payload/research_experiments/)。`M/` 表示 `mobile_ev_mle/`，`L/` 表示 `mobile_ev_probing/`。完整逐文件记录见[源码目录CSV](../../archive/ev_methods_20261019/FILE_CATALOG.csv)，包括各文件的首行说明、主要函数和源码中可直接识别的输出文件名；这些输出提示是静态提取，具体调用参数仍以对应协议为准。

## 代码、测试与实验入口

| 文件或文件组 | 作用 / 输入 | 返回或落盘输出 |
|---|---|---|
| `M/likelihood.py` | 完整Gaussian P/Q/V及总表模型；输入候选、量测、误差模型、参数域/起点 | `FitResult`，线路、逐时状态/源、偏差、取得损失与停止信息；本模块本身不负责批次落盘 |
| `M/candidates.py` | 已到达量测的被动提案及局部修正 | 有限候选集，覆盖范围按具体批次记录 |
| `L/ac_general_v9.py` | 径向树、完整AC潮流及相关物理核对 | 候选/AC响应；同文件旧 `solve_outer` 留作误差盒历史方法 |
| `L/passive_bridge_v9.py`，`L/wzzt_adapter.py` | RNJ/WZZT及源项兼容桥 | 候选/支持集/求解诊断 |
| `M/simulation.py` | 合成物理状态及含噪计量，真值只交评价器 | `SyntheticPath`、观测、评价真值 metadata |
| `M/design.py` | 最早可行EV动作与条件历史锚定判别 | `EVAction`、可行性检查、动作得分；该版本含有限动作枚举，不能概括后来的连续MILP |
| `M/information_context.py`，`M/information_targeting.py` | 历史中心、公开量测精度和不确定结构的历史优先级 | 规划上下文、历史导数、任务优先级与白化映射 |
| `M/ev_fixed_train_fit.py` | 固定末端总P/Q输入的条件AC拟合 | 每候选取得的R/X、逐时同接口源及拟合状态 |
| `M/ev_fixed_train_design.py` | 固定每候选训练R/X，以竞争源调节做规划 | 规划上下文与候选动作，供study写记录 |
| `M/ev_systematic_refit.py` | 同一动作的固定训练线与共享线推断 | `RefitHistory`、FIX/JNT候选拟合/损失/参数误差 |
| `M/ev_mobile_service_domain.py` | kW/kWh运行约束，接入、出行、SOC及终端任务 | 线性域/弧/变量索引及原域操作检查结果 |
| `M/ev_mobile_v4_domain.py` | 四个嵌套域、计划提升、同分数父计划继承 | 可执行父/子计划、AC评分缓存、继承检查 |
| `M/ev_fixed_response.py`，`M/ev_fixed_gradient.py` | 固定线AC/LD/I²响应及源profile梯度 | V、根P/Q、响应导数、竞争对距离/梯度 |
| `M/ev_scalar_distance_milp.py` | 冻结标量源profile L2距离的分段弦MILP | 原动作值、主问题界、源子问题检查、计划 |
| `M/g_precision/coefficients.py` | 冻结候选响应及标准差 | 每候选对的常数/一次/二次系数、精确冻结分数 |
| `M/g_precision/context.py`，`study.py` | 训练响应中心；P-SQ/G-SQ/G-AQ同域规划与评价 | `trial`/`freeze`/检查点及拟合、计划、代理界、费用 |
| `M/block_joint_margin/study.py` | H与假想未来双方共享线拟合，完整AC价格提案 | 开发/评价历史记录、价格验收/回退、候选优势与费用 |
| `M/ev_mobile_v4_study.py` | aligned/guarded同目标继承，配对regular/shift | 每历史计划、实际候选损失/裕量、父子边与操作记录 |
| `M/ev_bayes_small_*` | 小型Gaussian预测风险与保守重叠代理 | 后验/风险、MILP、审计、固定协议结果 |
| `M/ev_bayes_response_ac.py`，`_truncation.py` | 单中心共享线/PQ/源局部联合不确定性 | 截断域积分、联合矩、预测均值/协方差、精度失败 |
| `M/ev_bayes_response_milp.py` | 冻结协方差下BC代理，完整模型接受 | 计划、代理界、原域操作失败/合法回退 |
| `M/ac_history_refresh/` | 不改变原计划，刷新历史AC后验中心/预测 | 原/新概率与预测差、校准、可用性及补充响应误差审计 |
| `M/full_pipeline/data.py`，`forecast.py`，`passive.py` | OPSD窗口/来源核对，因果预测，实际候选生成 | `prepared`/数据清单、预测配置与误差、原池和更新池 |
| `M/full_pipeline/study.py`，`controller.py`，`analysis.py` | 从候选生成到真实AC含噪执行和共享拟合 | `development/calibration/evaluation`逐历史记录，阈值、点判别、保留集、费用、完整失败分母 |
| `M/reference_benchmark/` | 原标准网、公开P/Q曲线与更大冻结批次 | `prepared`、网络来源、控制器/恢复状态、逐条件记录；本批封存时仅28开发记录 |
| `M/ac_block_fit/blocks.py`，`linear.py`，`nonlinear.py` | 共享线路块/局部源块、QR消元、有界阻尼GN | 线性步/KKT、完整AC接受、取得解与驻点检查 |
| `M/ac_block_fit/batched_source.py`，`fixed_source.py` | 固定线逐期二维有界源拟合，批量/串行对照 | 源状态、逐期停止/失败及完整历史回放 |
| `M/ac_block_fit/polished_fit.py`，`gpu.py` | 可选端点修整、FP64 GPU物理计算 | 微升/梯度日志、CPU/GPU响应与计时，未接替默认TRF |
| `M/case33_quality_fixed_rx/experiment.py`，`audit_saved.py` | 已知真树/RX，LD、DF-I²、AC及P/Q-EIV | 306方法槽、自身/AC回代分数、梯度、多起点与独立核查 |
| `M/case33_candidate_ranking/experiment.py`，`audit_saved.py` | 已知逐设备RX，10棵候选树评分 | 3060方法槽、胜者/逐对逆序、AC回代与未达标分母 |
| `M/*study.py`，`*controller.py` | 各代批次执行入口，读取对应protocol | `trial_*.json.gz`或`history_*.json.gz`、freeze/source snapshot、阶段检查点 |
| `M/*analysis.py`，`*collect*.py`，`*report.py` | 保存记录的统计/整理；有些诊断另做AC回放 | summary、CSV、图、失败与费用统计；是否新拟合按模块说明辨认 |
| `M/*audit.py`，`*review.py`，`*restore.py`，`*delivery.py` | 物理/统计/来源核对、证据打包或恢复 | 独立审计JSON、source/member哈希、zip包及恢复回执 |
| `M/test_*.py`及各子包`test_*.py`，`L/test_*.py` | 历史组件/数学/运行域/审计测试 | pytest结果；冻结快照中的测试属于复现材料，封存区不自动收集 |

## 文档、记录和二进制材料

| 类型 / 位置 | 作用和阅读方式 |
|---|---|
| `*_math.md`、`theory.md` | 公式、假设、代理界及证明范围；与同前缀实现/报告配套 |
| `*_protocol.md`、`*_freeze.md` | 事前样本、预算、目标和验收；不因封存而改写 |
| `*_results.md`、`report.md`、`REPORT.md` | 人工解释后的批次结果和失败；先读这些，再追原始汇总 |
| `*_checkpoint.json`、`checkpoint.json` | 当时运行/交付状态；本次停止批次以新STOP_RECEIPT为准 |
| `*_reproduce.md`、`REPRODUCE.md`、`REPLAY_*.md` | 恢复/复算命令、证据层次、环境与版本 |
| `evidence/`、各子包`evidence/` | 已提交的紧凑报告、冻结源码、输入样本、阈值、完整或精简记录、审计/恢复包 |
| `outputs/`、外部D盘执行副本 | 本机原始记录、源码/数据副本、完整轨迹、中间结果与执行环境；多数不在Git中 |
| `*.npz`、`*.npy` | 数值数组、轨迹、状态或观测；与旁边JSON解释共同使用 |
| `*.zip`、`*.tar.gz`、`source_snapshot.json.gz` | 按原receipt/manifest恢复并核对成员哈希；精简统计包是否含完整物理记录由原README说明 |
| `__pycache__/`、pytest临时物 | 可再生缓存，本次随目录保留为本机材料，不追加上传 |
| 原 `README.md`、`NEXT_PROMPT.txt`、resume/followup计划 | 历史叙述与当时续接计划；当前入口是本组封存说明 |
| `docs/mobile_ev_probing_report_v2.tex/.pdf`及其他版本 | 旧长报告留在仓库`docs/`原位；PDF存在既有未提交改动，封存未重编译、未改动 |

## 更早阶段怎么找

`adaptive_*`、`continuous_eig_*`、`information_*`、`structured_*` 对应早期信息目标/积分；`ev_profile_*`、`ev_bilinear_*`、`ev_cross_*` 对应历史支持及双线性设计；`ev_compact_*`、`ev_dynamic_*`、`ev_fixed_train_*` 对应紧凑MILP、动态负荷和固定训练线；`ev_gradient_flow_*`、`ev_response_metric_*`、`ev_systematic_*` 对应梯度、距离与系统比较；`ev_mobile_v2/v3/v4_*` 对应各代移动域/指标。相同前缀的 protocol、math、study、results、checkpoint 和 evidence 应一起读取。

`L/README_v6.txt`至`README_v9.txt`与`results_v6`至`results_v9`是较早的误差盒/外LP路线；其中AC前向、候选桥等仍被后续代码复用，所以两个源码包整体保存。更早的2,658文件本地归档在[2026-10-12归档](../../archive/mobile_ev_pre_mle_20261012/README.md)，未重复迁移。

各D盘旧执行目录的新位置、原始文件数及体积见 [EXTERNAL_LOCATIONS.csv](../../archive/ev_methods_20261019/EXTERNAL_LOCATIONS.csv)。执行目录整体迁移后，历史绝对路径字符串仍保留原字节；复现先按恢复说明还原原有布局，再运行旧命令。
