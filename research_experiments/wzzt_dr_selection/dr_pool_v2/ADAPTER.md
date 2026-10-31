# 独立需求响应观测接口

`adapter.fit_observed(training, validation, output_dir, root_label=1, search_seconds=25., lp_method=1)` 接收现有 `end_to_end.wzzt_adapter.make_observed_scenario` 的场景列表。P/Q/V/root均为标幺值，各通道索引和末端顺序必须完全一致；真拓扑不是接口输入。

## 与已核验策略的对应

保留 `multistart_original_rule` 的训练池构建、15个raw RNJ完整族联合L1拟合、主25秒路径、最多两个额外25秒路径、3%验证选模和0.16最终弱块收缩。返回 `rnj_bank`、`baseline25`、`multistart_original_rule` 三个模型。没有运行long75、AC门控、全候选收缩后重选等其他版本。

`provenance.json`记录所复用主工作区算法文件的哈希。核心图算法及有序稠密回归来自本工作树，运行时哈希写入结果；QP明确传5000次迭代，与冻结producer的预算一致，不使用本工作树500次的默认值。`constrained_least_squares.py` 与冻结producer版本相同；没有外部Source输入。

独立调用从冷的LP缓存开始，仅预先拟合本方法所需的15个RNJ家族。这与历史多版本批次复用父路径及预热缓存不同，因此同一墙钟上限下截断路径可能不同。75秒仍是三个局部路径名义预算；结果的elapsed_seconds涵盖本次完整调用。每次LP最多15秒，可能跨过局部路径截止时间。


## LP数值方法（2026-10-31新增显式配置）

fit_observed与PoolLP接收lp_method=0/1/2，默认仍为1。参数进入输入身份、结果settings及每次LP回执；增加支撑时保留原方法。0为primal simplex、1为dual simplex、2为barrier，均保持相同目标、非负权重、界、容差与15秒LP时限；该参数不改变W的接受、验证及收缩规则。

此配置用于诊断预算内参照批的首个固定族LP状态9。四个单LP对照已在原15秒内最优，但不据此承诺完整路径速度或恢复优势。非唯一最优点、数值剪枝和墙限会使不同求解方法产生不同路径，因此新配置必须另存协议与结果，不能覆盖旧失败或复用旧源码缓存。两次完整联合重放见dr_balanced_design/NUMERICAL_REPLAY_REPORT.md。SharedFitter亦接受显式lp_method并绑定cache key及方法记录；不同方法不共享结果。旧默认fake无方法记录时标为legacy_default_unrecorded，非默认缺记录或记录冲突直接报错。缓存修复见dr_balanced_design/CACHE_METHOD_REPORT.md。

## 对真实DR数据的必要接口修正

- 按DatetimeIndex的真实日期去均值，不按三等段切分。因此3个历史日＋1个行动日可以直接处理。
- 每个物理日允许不同采样点数，但至少2点。非DatetimeIndex必须传与行一一对应的day_ids。
- 每个观测通道、场景顺序和列标签先核验；不自动交集、排序、合并终端或补值。
- root_label为参数，当前NJ/RNJ实现要求整数物理标签且根不在末端集合。
- 每批使用实际训练观测重新建候选池；验证仅用于额外初值和模型选择。

## 结果与恢复

`models` 中每个模型都包含最终 supports/r/x、support_indices/support_labels、r_values/x_values、r_matrix/x_matrix及R/X。train_mae/validation_mae是最终收缩后实际模型误差；selection_validation_mae及selected_uncontracted保留选模使用的收缩前模型。

pool_support_indices/labels、完整bank、searches、逐固定族LP状态与数值约束检查、校准、输入及源码哈希均保存。status=complete表示调用结束且所有固定族LP通过数值最优与约束核验；搜索仍可能在pool_wall_limit或step_limit停止，应查看search_stop_statuses，不能解释为全域最优。

失败保留result.json及异常；已有模型保留供诊断，但不伪装成完成。checkpoint.json保存已完成阶段及接受路径；不恢复底层求解状态。resume=True只复用输入和源码哈希完全相同的complete结果。未完成断点拒绝静默重启或增加预算。

## 已完成检查

7项测试通过：四日不等样本分组、列/时间错配、训练验证重叠、显式日分组矛盾、独立一维L1中位数例、原已存paper15高噪N24的选模/收缩纯重放、3节点独立建池/拟合/多初值/只读resume。selection_fixture.json不含原始负荷，只包含保存模型和来源哈希。

该接口检查不代表DR收益通过；实际配对批次由runner另行保存。
