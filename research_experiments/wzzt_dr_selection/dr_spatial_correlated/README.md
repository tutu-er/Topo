# 开放节点、整段相关误差DR

模型见MODEL.md；本批结果完成后写REPORT.md。复用原条件L1积分、冻结选模器和AC/量测生产器，没有重写拓扑辨识器。

Python入口runner.py依次prepare、optimize、observe、evaluate，--output指定新的私有目录。prepare保存源/输入hash、协议和种子；optimize冻结所有策略；observe才允许生成真实网络新日；evaluate先冻结选择再进行真实结构评价。已有记录按receipt复用，不覆盖重跑。现场路径在脚本REPO/PRIVATE中明确声明。

运行环境：D:/apps/miniconda3/envs/Topo/python.exe，NumPy/SciPy/Pandas/Gurobi；原参考AC源码来自outputs/wzzt_paper_20261029_full/source。源与输入变化会使已锁批次拒绝续跑。原轨迹、完整参数、控制/随机数组只存私有目录，公开仅源码和聚合汇总。

纯测试：python -m pytest research_experiments/wzzt_dr_selection/dr_spatial_correlated -q -p no:cacheprovider。实验冻结后新增的独立测试/审计不追改已执行源manifest；其字节另保存。

数值求解是有界多起点局部方法。若没有返回满足95%有限积分约束的点，保留最大概率诊断方案并明确标注未达标，不能称问题不可行、最小费用或概率已验证。独立512积分只核验，不按其结果事后挑价格；实际AC结果也不用于重新优化。

本轮原有限罚规划保留在run_v1，没有生成实际新日。一次有界约束修复在refinement/refined_runner.py：prepare/run --output 私有run_v2，随后使用原runner.py的observe/evaluate --output run_v2。constrained_refine.py新增6项解析测试，修复先达概率要求再降费，未读取新日。refinement脚本显式定位同一私有stage及原输入，不覆盖旧协议或价格冻结。
