# 本次执行环境口径

四个独立历史worker并行，BLAS/OMP/MKL环境线程值为1。主辨识MILP使用Gurobi13.0.3，显式Threads=1、Seed=0。
确定性计量质量/可行操作初值调用SciPy/HiGHS，沿用该接口默认选项，没有单独指定HiGHS线程数。
原environment.json和冻结config中的solver_threads=1字段应按主Gurobi口径读取；本说明补足其范围，不改写冻结源、配置或旧记录。
限时求解的实际墙钟包含初始化、模型形成、QP及完整AC验收；多worker耗时是当前机器的观测值，不作隔离硬件性能基准。
阈值文件保存时间与四个OS进程启动时间的只读证据见pre_evaluation_execution_order.json。
