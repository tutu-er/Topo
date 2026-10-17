# G-SQ / G-AQ：拓扑灵敏度与计量精度

统一名称见[方法表](../METHOD_NAMES.md)。G-SQ为拓扑加权平方MILP，G-AQ进一步保留背景差异及线性项；强对照P-SQ是只用公开计量精度的平方MILP。

本轮实现与独立评价完成，先读[简明结果](results.md)。完整分母、失败、求解间隙和复算证据见[evidence/README.md](evidence/README.md)。

先读math.md、protocol.md。源参数和未来预测在规划中冻结；真实计量到达后分别运行固定R/X与共享R/X联合拟合。全部候选对保留。该版本不包含滚动重规划或概率校准。

从Topo源码根目录执行：

```powershell
python -m pytest research_experiments/mobile_ev_mle/g_precision -q -o addopts='' -p no:cacheprovider
python -m research_experiments.mobile_ev_mle.g_precision.study --phase development --output <新输出目录>
python -m research_experiments.mobile_ev_mle.g_precision.study --phase evaluation --output <同一输出目录>
python -m research_experiments.mobile_ev_mle.g_precision.analysis --output <同一输出目录>
```

正式数值环境使用Topo Python 3.11和Gurobi。小型MILP单测可用SciPy；完整driver的公开基线补全使用Gurobi。设置OMP_NUM_THREADS、MKL_NUM_THREADS、OPENBLAS_NUM_THREADS为1可复现实验线程条件。新增原始记录不可覆盖，续接仅用--resume；科学源码hash变化时需新版本和新输出目录。

规划只接收已观测数据。未来模拟、真实拓扑及新增无噪声诊断均在计划封存后进入评价，失败和未决保留在完整分母。source_snapshot.zip和freeze.json记录实际执行源码；单个代理间隙只约束冻结二次目标。
