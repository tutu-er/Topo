# AC 参数增强接口核验

2026-10-29。只读检查当前任务工作树，并运行既有测试；未改生产实现、算法参数或DR数据。

结论：旧的固定树 AC 修正/排序接口和新的完整 AC P/Q/V 联合拟合接口都仍然存在且可调用。75项相关既有测试通过。它们均未接入当前 `dr_pool_v2.adapter.fit_observed`；不能把“接口可用”写成“当前DR流程已使用AC增强”。

## 1. 旧固定树 IRLS + AC 修正

- 入口：`terminal_case33.pipeline.ac_likelihood.rank_candidate_trees_ac`，第522行。
- 调用方：`terminal_case33.pipeline.topology_validation.select_and_validate_topology`，第166行；另有`experiments/run_ac_quartet_rerank_demo.py`。
- 输入：`candidates, training_scenarios, validation_scenarios, reference_net`，以及P/Q与电压相对噪声率等；每个scenario须保留`P_terminal/Q_terminal/V_terminal/root_voltage/drop_target`，候选为该模块的`RNJCandidate`。
- `_fit_eiv_tree`第140行：固定树路径矩阵，非负共同边系数；最多8次误差传播IRLS，每步新旧系数各取0.5。
- `_refine_with_ac`第341行：在代表训练点做AC，计算AC与线性平方电压降之差；作一次修正重拟合，然后依次试1/.5/.25步长，接受首个令训练AC电压RMSE严格下降的步长。这里不是无限反复AC修正。
- 最后在留出场景进行预测AC评分，并加既有复杂度/场景离散惩罚。有限候选集没有的拓扑无法恢复。
- `_candidate_network`第248-249行：原子系数转物理Ω为`.5 * coefficient * Z_base`，且有`1e-12`数值地板；零边不是精确保留。
- 证据：`tests/test_ac_likelihood.py`第82行核验留出AC排序选真候选且训练AC RMSE不升；另2项验证代表场景划分。

## 2. 当前 core 的完整 AC P/Q/V 联合拟合

入口：`rnj_wzzt.estimation.ac_measurement_fit.fit_ac_pqv`，第184行。

```python
fit_ac_pqv(net, observed, *, sigma,
           fit_impedances=True,
           r_bounds_ohm=None, x_bounds_ohm=None,
           source=None, source_bus_id=None,
           max_nfev=100, ac_max_iter=200,
           ac_tol=1e-12, tol=1e-8)
```

- 固定闭合边拓扑；拟合跨时段共享物理线路R/X、逐时段潜在真实P/Q，以及可选根电压。每次残差计算都调用径向AC潮流。
- 输入`net`须为core的`TerminalizedNetwork`，不是终端灵敏度矩阵；该类与terminal_case33的类分属不同模块。
- `observed`必须含原始P/Q/V DataFrame和root Series，标签严格对齐；可选总表仅在提供对应sigma时计入。不读取truth，不重复计入由电压派生的drop_target。
- `sigma`为`ACMeasurementSigma`，全部是正的绝对pu标准差；不支持零噪声硬约束（第104-105行）。
- `fit_impedances=True`要求有限、非负、包含初值的Ω范围（第146-154行）；允许下界为0。范围必须由调用方提供，不能为了结果随意限制。
- 可选Source的幅值与位置固定；不搜索源位置，也不估固定表计增益D。
- SciPy `least_squares`，稀疏三点差分＋TRF（第345行）；局部优化，无全局最优保证。AC失败显式报错；优化预算耗尽保留`success=False`及实际损失。
- 返回`fitted_net`副本、初始/最终损失、逐通道标准化残差、AC结果、状态、活跃界、Jacobian秩等。验证调用`fit_impedances=False`冻结训练阻抗，但仍拟合验证批潜在P/Q状态；不是纯预测误差。
- 独立demo：`rnj_wzzt_core/demo_ac_measurement_fit.py`第101、116行展示训练和独立验证。
- `rnj_wzzt_core/README.md`第143-146行明确它尚未自动接`pipeline.run()`或MILP选候选；对当前调用链的源码检查一致。

## 3. 当前 DR 适配器状态

`research_experiments/wzzt_dr_selection/dr_pool_v2/adapter.py`第39行入口，约第101行设置`final_refit=False, AC_gate=False`。实际流程为新建RNJ/NJ池、固定族L1、搜索与选模；没有调用上述两个AC接口。

当前五末端小测试的Y/V由线性无噪声模型生成。它不应原样插入完整AC拟合并宣称工程增强，尤其不能为绕过接口任意设正sigma或阻抗界。后续若比较AC增强，应另锁真实AC模拟观测、统一的量测误差和参数先验，对RNJ/WZZT候选同样处理，单独报告参数改善与结构选线。

## 4. 本次执行的核验

运行环境：`D:/apps/miniconda3/envs/Topo/python.exe`，临时PYTHONPATH为工作树根及`rnj_wzzt_core`；`-B`和`PYTHONDONTWRITEBYTECODE=1`，关闭pytest缓存。

```text
python -B -m pytest -p no:cacheprovider 
  rnj_wzzt_core/tests/test_ac_measurement_fit.py
  rnj_wzzt_core/tests/test_ac_validation.py
  tests/test_ac_likelihood.py -q
```

结果：75 passed in 13.12s，进程exit_code=0。完整输出与命令、源码哈希、输出哈希在`ac_interface_tests.txt`和`ac_interface_test_receipt.json`。包括独立二母线闭式解、错误阻抗恢复、P/Q校正、异方差残差复算、固定Source、根电压、参数边界、标签错误、AC不收敛及优化预算失败、候选AC排序等检查。通过这些接口测试，不等于已验证paper15或本轮DR的参数/拓扑效果。
