# 接受后随机DR：恢复入口

先读REPORT.md和EXPERIMENT.md。核心结论：精确小系统存在DR正例；观测候选新增0/28，51/51与当前池/简单全族相同，噪声条件无改善。RNJ补一个可选公共根块已复现全部33条无噪W输出，未形成W专属价格方法。

入口：

- random_code.py：接受后独立整段方向、执行日志、观测矩和条件候选。
- study.py：冻结后生成同步历史的线性机制模型；truth只送选后评价。
- gate_driver.py：原build_bank、共同L1、cross repair及原3%/.16链；仅run调用求解器。
- evaluate_gate.py：独立结构与L1评价；不拟合、不设计策略。
- independent_run_audit.py、independent_structure_summary.py、audit_selected.py：独立只读复核。
- evidence/random_observation_gate_v1：协议、逐条结果、评价、源码和失败回执；没有原始P/Q/V/家庭轨迹。

私有根都在本线程scratch，已有记录禁止原地重跑：

    wzzt_dr_random_observation_generation_v1  JSON序列化失败，0优化，保留
    wzzt_dr_random_observation_generation_v2  生成完成，51映射/28snapshot
    wzzt_dr_random_observation_execution_v1   759LP，28/28，已结束
    evaluation_v1                           独立评价，0新求解
    dr_random_protocol_audit_stage           独立1458项核验

完整运行顺序仅供恢复：先纯测试，study prepare/generate新根，gate_driver prepare/run另新根，evaluate_gate读取固定输出。完整源码哈希、原种子和数据hash见证据。已有批次不应延预算或重抽行为，n4_gate_passed=false。

研究只更改本隔离目录、STATUS/RECOVERY索引及对应证据；生产算法和原paper15数据不变。新方法尚无价格优化器，也没有统计置信/真实性概率或全参数证书。
