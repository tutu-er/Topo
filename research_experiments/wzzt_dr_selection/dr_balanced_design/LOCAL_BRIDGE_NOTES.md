# 冻结局部DR物理桥的最小参数化

- 沿用原physical_bridge/coded_physical/actual_worker/controller，不改RNJ/WZZT识别器。
- 非legacy来源在价格hash通过后显式绑定case、metadata hash、根和基准；实际producer上下文必须一致，才可生成新profile。
- active_labels_by_row按policy声明两个逐轮节点对；报价不得跑到该行配对外。同向/反向的双报价方向须一致，旧legacy全局任务保留旧行为。
- orientations_by_event可冻结四个独立整段随机方向；缺省全1保留旧结果。参与概率及部分响应未改。
- worker新增case参数，观测包根和基准来自校验后的producer；controller显式pilot_job先于旧默认筛选，并冻结history_loader_kwargs。
- controller旧八策略、双响应律和early/final框架保留；另锁final-only工程pilot时直接复用既有actual_worker及SharedFitter，不复制辨识器。
- 25项新增物理边界测试、6项新增worker/controller测试通过；加8项既有controller及10项既有coded/history回归，共49项不同测试。首次coded回归仅因旧测试需要显式module目录而在收集阶段失败，0项执行；调用路径修正后10项全过，失败回执留私有。
- 本改动为来源/执行接口准备，不构成价格创新或DR恢复收益；没有在上述纯测试中调用图、实际profile、AC或LP。
- 新物理比较必须先冻结全部价格、位置、方向、source hash、随机流和预算；真拓扑只在全部观察拟合尝试固定后评价。
