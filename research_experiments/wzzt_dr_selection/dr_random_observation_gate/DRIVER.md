# 最小随机控制候选门：识别驱动

该模块仅处理观测和数据产生的候选，不生成负荷、不设计价格、不读取真结构。导入/prepare均不实例化优化模型；只有root明确执行run才调用真实engine.PoolLP/search(repair=True)及原selection。

每份snapshot从全部已经到达的training调用一次真实build_bank，其raw calibration应用到所有方法；未手工设置有利a/b/median。旧history的池和starts仅保留为消融。原接受1e-10+.001J、3%选择、.16收缩和[0,2]连续域均保留，所有叶权允许0。n3全部8合法族作为固定支撑LP参照；n4为52族的组合函数保留，但本轮root协议只开n3，不自动扩大。

## 纯接口

- prepare_history(history_packet, settings, budget, output_json)：历史packet只有training/validation场景和observed_only元数据；真实历史densefit/build_bank与联合拟合产生旧池/旧starts，不将该旧校准强加给新snapshot。
- identify_snapshot(history_packet, future_packet, history_state, discovery, settings, budget, output_json)：future packet明确contains_history=false，驱动自动保留全部历史。先从当前全部training生成current_state，各方法共享当前raw calibration和当前bank；增pool W从current pool增加候选。七个主读出是历史RNJ库重拟合、当前重建RNJ、旧pool W、current pool W、增pool W、同pool合法树枚举、全合法族参照；另有直接control树重拟合。
- observed_training_for_discovery(history_packet,future_packet)：只用两个training分割经真实normalize生成P/Q/Y，绝不读取validation；与random_code.discover_candidates接口匹配。
- controls/random-code生成和discover_candidates由random_code.py/根负责，仿真truth只在各方法输出固定后由根评价。所有方法看到同一实际观测，历史单独校准共享。

每种读出独立PoolLP，不共享其他方法的LP缓存，避免把缓存顺序/非唯一解差异伪装成计算优势。数据不足/LP到时/未全扫描/无directfamily均保存，未决不填0；all_legal枚举是n3/n4有限域参照，不是全规模算法。

## CLI与锁定配置

prepare接收root已经生成且冻结的观测包和候选JSON（本模块不触碰数据生成随机流）：

    python -B gate_driver.py prepare --root <new-execution-root> --config <root_config.json>
    python -B gate_driver.py run --root <same-execution-root>

config必需字段：status='root_frozen_before_solver'；settings={coefficient_bound:2,max_n:3,root_label:1,calibration_snapshot:'current_observed',search_seconds:25,wall_seconds:<root固定>,max_LP_calls:<root固定>}；histories=[{id,path,sha256}]；snapshots=[{id,history_id,path,sha256,discovery:{path,sha256}}]。当前RNJ复用该snapshot已生成的bank，不在方法内重复denseQP；每种方法仍用独立PoolLP。直接control树始终允许full-root块且权重下界0，不因平衡码抵消公共块就剥夺简单基线的根块。

history/future packet是私有pickle，唯一允许顶层键training、validation、metadata。metadata.observed_only=true，future另须contains_history=false。场景沿用真实P/Q/V/root/drop_target/units契约；不要手造缺失units/drop_target的四字段截窗。prepare冻结packet和候选hash及实际依赖closure；有结果/锁的执行根拒绝重跑。真结构/物理网络/未来自然P0等不进入packet。

root最终首门为最多32个内容去重snapshot（另有51个评价case-policy映射），n3不打开n4，统一900秒/3000真实LP。每snapshot当前state准备至多8个族，七主读出各至多8个族，direct树至多1族，即保守65次LP/snapshot，加每个独立历史state至多8次。每份当前state含2个densefit(raw/demean)，current RNJ复用它；这些只是上限计算，不是已运行耗时或收益。零/弱/强、三真pair和b=0条件由root冻结，不从结果选取。

mean planning、价格—随机参与—接受后随机code、真实随机自然负荷与设备预算检查属于外部数据/合同层。此驱动没有全空间价格优化器，也不把候选生成或控制矩变化当最终恢复。

当前6项纯测试通过，其中真实观测契约和mock状态传递验证了当前calibration共用、old/current/expanded三池差异及direct根块允许。早先4/5测被本版覆盖，不叠加计数。尚未由本agent启动任何denseQP、LP、AC或算例数值实验。root协议是运行前提。
