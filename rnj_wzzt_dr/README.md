# 拓扑辨识与配网灵活性研究入口

当前辨识代码集中在`identification/`，DR读其冻结候选与R/X，评价期间不重新拟合。
近期整理恢复了W的廉价单连接QP预试，E保留一个固定参考初值；删除弃用的整树热启动字段。
[版本差异与贡献核查](experiments/identification_contribution_20261116/VERSION_DIFF.md)解释历史有效R＋W和当前实现的区别，
[历史结果](experiments/identification_contribution_20261116/HISTORICAL_RESULTS.md)与本轮同后处理测试分开报告。

本轮[完整结论](experiments/identification_contribution_20261116/RESULTS.md)：旧主240统一后处理后，101内部恢复120/240，多RNJ基线55/240；
近期长历史8输入中，单参考RNJ与101均8/8且最终矩阵相同。方法的额外收益取决于输入条件，不能只根据初值或返回来源标签归因。
[早期耗时核查](experiments/identification_contribution_20261116/EARLY_WZZT_TIMING.md)区分纯W真实搜索、强初值后的求证成本，以及E和恢复分支开销。

## 当前辨识流程

```text
训练P/Q/Y → 前置R/X拟合 → RNJ连接池及固定参考点
基础支持模型 → W：池内单连接QP初值＋未知连接MIQP → 可选T → 可选E
本方法接受路径＋单节点保护模型 → .08非末端收缩及训练QP → 验证MSE选择 → 冻结
```

| 部分 | 实际作用 |
|---|---|
| RNJ | 生成连接池；提供W连接模式与E的一个固定参考点；参考尺度仅作共同训练校准 |
| 初始结构 | 每个观测末端的单节点支持＋全节点公共支持，R开启也从这里开始 |
| W | 对池中兼容的单连接做固定族QP，取较好点初始化，再用MIQP搜索一个未知连接并联合调权 |
| T | 最多5块的局部联合重构；101关闭、111开启 |
| E | 从有限连接池联合重选并拟合R/X；可撤销原内部连接；不进行15整树预试 |
| 初值归因 | 当前→实际初值、初值→原生返回、原生→QP收尾分别记录；保留预试点须注明来源 |
| 输出收缩 | 仅训练数据，RX75相对强度低于.08的非末端支路逐条删除并QP重拟合；末端接入支持豁免 |
| 最终选择 | 原验证MSE，归一化并列容差1e-12，再按支持数、分数、原序决定 |

训练目标为`MSE_train/M_ref + .003×内部支持数`。默认仍为111，100是R＋W，101是R＋W＋E。
R关闭的000不接收RNJ结构池或参考热启动，仍保留共同数值尺度。普通方法的最终库没有独立完整RNJ整树兜底；
E原样返回参考初值时，收益归初值。W预试已找到的点也可保留，收益归预试。
W实际模型初值保留全部旧支持，即使其系数已归零；规范预试点的清零收益另记，避免计数罚项造成假的原生负收益。
完整数学和DR接口见[MODEL.md](MODEL.md)。

## 结果应该怎样使用

旧主240、当前40和16是不同的数据预算与来源，按组比较。旧压力160已退出新默认清单，见[evaluation_scope.json](evaluation_scope.json)。
旧40/40、16/16是旧联合候选库经过.08收缩的回放结果；不能当作新简化实现的重新辨识结果。
最近四例的无预试A/B版本及其失败保留在[原配对说明](experiments/rnj_pool_only_20261116/README.md)。
内部恢复检查2至N−1节点的活动集合；全部支持恢复还包括末端和公共项。R/X矩阵误差另报。
历史研究、DR正负结果和旧入口索引已移到[HISTORY.md](HISTORY.md)，原数据和源快照保持原路径。

## 使用

Python3.10+、NumPy、SciPy；结构搜索与非零采购需Gurobi及许可。

```powershell
$py='D:/apps/miniconda3/envs/Topo/python.exe'
& $py -I -B rnj_wzzt_dr/run.py doctor
& $py -I -B rnj_wzzt_dr/run.py check --solver
& $py -I -B rnj_wzzt_dr/run.py fit --train /private/train.npz --validation /private/validation.npz --method 101 --output /private/fit
# 短预算W对照：总限120秒，每次W未知连接求解最多10秒；不改变通用默认。
& $py -I -B rnj_wzzt_dr/run.py fit --train /private/train.npz --validation /private/validation.npz --method 100 --seconds 120 --step-seconds 10 --output /private/short_W
& $py -I -B rnj_wzzt_dr/run.py score --result /private/fit/result.json --history /private/validation.npz --new /private/new_day.npz --output /private/score.json
```

观测含`labels,P,Q,V,root`，或用`Y=root²−V²`替代电压列；功率为pu，节点顺序固定。
`--internal-floor 0`关闭输出收缩。输入不包含真拓扑；验证不生成候选，新日不重拟合。
`--step-seconds`只限制W的一次未知连接MIQP；T/E使用所在阶段的剩余时间。省略该选项保持原默认，不能将它理解为所有原生求解的统一硬限。
`zero/plan`接收冻结库、明确目标和事前功率预测，命令见`run.py --help`；原设备、AC桥及资产边界见[ARCHIVE.md](ARCHIVE.md)。
多竞争者非零采购的通用入口仍有待接通；各实验目录中的采购实现按其自身协议使用。

`PROVENANCE.json`核对当前运行闭包；历史源哈希保留。私人P/Q/V、拟合系数和报价轨迹保存在ignored outputs，不提交Git。
