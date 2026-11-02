# 连续异质DR概率可行化与文档

当前完整结果见 REPORT.md，数学模型与推导见 algorithm/main.tex、algorithm/main.pdf。源包algorithm/source_bundle.zip包含结果子文件和图，可直接XeLaTeX编译。17项不同测试、576/576实际记录、25,240项独立重放已通过。

本轮没有找到95%规划可行方案。paper15主响应新价恢复21/24、旧价20/24、同费轮换3/24；失配新旧均20/24。新费88.2076比旧53.3829高65.24%，不建议默认替代旧价。flynn仍0/24；pengwah零采购24/24。旧6日结果保留，本批24日不是将旧6日重复计算。95%是原验证排除事件的规划目标，不是真拓扑恢复概率。

## 实现

- tail.py：原有符号L1损失差的精确条件正部期望，只作尾部初值。
- refine.py：复用既有设备控制/条件概率/原AC/冻结选择；有界均值与尾部初值、原概率可行化、找到可行点后硬约束降费。没有新辨识器。
- policy_probability.py：对全部冻结策略使用同一4096独立积分点，只读核验、不回流选价。
- audit.py：独立重算loss、选择、错漏、报价费、CRN、设备状态和完整分母。
- test_refinement.py、test_cost_branch.py：数值积分/梯度与已知可行降费分支检查。依赖原库与本机保留的旧历史包，物理实验需要私有输入。
- export.py：公开汇总与初稿图表生成。最终文档和REPORT另作人工核对；重跑此脚本会覆盖自动生成的初稿，不要覆盖已核验文档。

## 恢复而非重跑

私有运行已结束：
C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_dr_probability_refinement_stage/run_v1

protocol/source_snapshot、三例pool/problem/solver_private/plans/price_receipt、price_freeze、actual/576逐次receipt/result、observations_frozen、scores、predictions_frozen、truth_private、summary、scored_private、audit、policy_probability_summary、ended/complete锁均保留。原run_v2一字未改；不重启本批optimize/observe。规划失败与壁钟/迭代停止均在各solver_private中。

若将来另锁新批，再依次执行prepare、optimize、observe、evaluate；后处理audit与policy_probability可以独立恢复，后者已有完成断点会复用。不改变已完成实验的hash、价格或随机流。原AC、冻结R/X、w2/common3%/.16和全库存均保留；新拟合与新W搜索为零。

## 公开范围

源码、声明协议摘要、标量汇总和文档。真实或聚合P/Q/V、完整R/X、价格/设备/AC轨迹及pickle仅留私有。公开配置中的路径/hash供本机恢复，不含输入数据本体。资源、行为、费用与3日预测的适用范围见文档。
