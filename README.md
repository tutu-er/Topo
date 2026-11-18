# 拓扑辨识与条件 DR

当前唯一开发入口是 [rnj_wzzt_dr](rnj_wzzt_dr/README.md)。流程为 RNJ 连接池与参考点、W 未知连接搜索、可选 T/E 重构、训练侧非末端收缩、独立验证 MSE 选择和完整候选库冻结。默认训练损失为 MSE，也提供 MAE；本次整理不改变算法和研究参数。

```powershell
$py = 'D:/apps/miniconda3/envs/Topo/python.exe'
& $py -I -B rnj_wzzt_dr/run.py doctor
& $py -I -B rnj_wzzt_dr/run.py check --solver
& $py -I -B rnj_wzzt_dr/run.py --help
& $py -B -m pytest rnj_wzzt_dr/tests -q -p no:cacheprovider
```

Python 3.10+；NumPy、SciPy 支持基础运行，结构搜索和非零报价另需 Gurobi 与许可。实测 P/Q 数据处理还使用 pandas。根 pyproject.toml 是安装入口，不再安装已退出的 rnj_wzzt_core。

| 位置 | 当前用途 |
|---|---|
| rnj_wzzt_dr/identification、dr | 辨识、冻结评分、零动作核算、设备及已有单对报价 |
| rnj_wzzt_dr/physical | 四网络 AC 模拟器和网络表 |
| rnj_wzzt_dr/experiments | 保留的研究实现和实验入口，按各自协议使用 |
| data/ | 本地私有数据与必要基准，Git 忽略 |
| data/frozen_studies/ | 最新两批的冻结候选、参数及逐例结果汇总 |
| KEEP.md、cleanup_manifest.json | 保留范围、清理结果及恢复位置 |

默认历史评估范围仍为 296 项，见 rnj_wzzt_dr/evaluation_scope.json。不同来源的数据预算按组比较。旧压力 160 已退役。

封存数据没有改写：读取时通过路径解析器找到新位置，再按原哈希核验内容。旧独立工程已拆出必要模拟器和基准文件，其余历史材料移出开发树。最新完整求解轨迹在仓库外压缩保存，当前候选库可用于后续 DR。

通用 plan 保留零动作和已有单竞争者报价；多竞争者非零采购仍按各实验的独立实现使用。
