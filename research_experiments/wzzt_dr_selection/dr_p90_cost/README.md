# 90%概率目标的小规模测试

先读REPORT.md（结果与限制）、METHOD_REVIEW.md（MILP/概率模型/创新性判断）、PLAN.md（预锁范围）。

cost_only.py只增加一次有界降费调用，复用原Probability与全部设备控制。run.py复用原AC观察和冻结选模；没有重写RNJ/WZZT，也没有实现全空间概率MIP。原95%代码与运行记录均未改。

执行源在私有wzzt_dr_p90_stage/code，完成记录在同目录run_v1。code/run.py的prepare/optimize/qualify/observe/evaluate五阶段均已完成，勿重跑。公开run.py包含本机恢复路径，私有输入需要另行提供；原物理轨迹不随源码发布。audit_export.py按私有目录布局只读审计，不是新的物理实验。

纯测：`python -m unittest test_cost_only -v`。3项通过；完整运行360记录、36独立背景、204不同观察。只在paper找到90%规划可行报价，其余保留诊断和零收益，不删困难条件。
