# 简化R/W/T/E研究入口

本包实现独立研究版本：RNJ提供候选和热启动，原始WZZT负责前向扩展，随后至多执行一次末端T和一次最终E。所有结构提议先QP重拟合并清理零块，再按同一个带线路数惩罚的训练目标接收。数学定义见 [MODEL.md](../../docs/simplified_rte_20261111/MODEL.md)。

本轮是四输入pilot；性能改进尚待实验核验，生产默认未更新。

## 搜索结构

| 模块 | 本版执行方式 |
|---|---|
| R | 从父`common.initial`及实际提供的RNJ/NJ热启动支持族导出候选；初值为单点/全集基底 |
| W | 原始未知支持扩展；R池只用于热启动，不限制未知支持域 |
| T | 最大直属子块数≤5后，枚举每个父节点全部宽度≤5的完整子树切分，并在合并候选池上联合MIQP |
| E | 在R池、自身全部路径支持、T池及全部终端对的并集上，一次联合可撤销选择 |
| 输出 | 原始自身路径加单点星形基线；严格验证MAE择优，另给同库MSE视图 |

T候选池中的旧支持可以释放，其余当前骨架保留；候选集合可以包含很多终端，限制的是每次切分的子树块数。E只固定基底和显式硬先验，其余临时块均可退出。T与E共用重构选择内核。

W结束后至多调用一次T，然后至多调用一次E。T后的零清理不再返回W。前向使用结构访问记录和总预算处理数值循环与终止。

## 共同目标及变化边界

训练损失为`L = K * physical_MSE`，尺度`K=f0_MAE/f0_MSE`继承父输入并保持固定。共同目标为

```text
J = L + 0.003 * f0_MAE * proper_internal_support_count
accept iff J_old - J_QP_and_zero_closed > 1e-10
```

本版去除小种子数量/大小筛选、删除排名、相对接收阈值、单独的纯MSE接收分支，以及RNJ小块近邻修复窗。最终输出不作`.24`或`.16`结构映射；父RNJ前处理自身的历史几何及`.16`选择继续存在。`alpha=.003`沿用且未调优，宽度5是工程上限。

父`common.pool`只用作固定QP容器的特征集合。搜索R域由已提供RNJ/NJ支持族重新导出，避免继承父包的额外候选策略。完整RNJ模型可提供可行起点；成为最终自身路径点仍须通过搜索接收，输出阶段没有独立完整RNJ备用分支。

本pilot实际传入父`common.bank`中已有模型的支持族；该接口按父记录提供的内容使用，不另跑NJ或RNJ生成额外族。

## 运行

在项目根目录使用包含NumPy、SciPy、Pandas及Gurobi的Topo环境。PowerShell示例：

```powershell
$env:PYTHONPATH = "$PWD;$PWD\rnj_wzzt_core"

python -B -m research_experiments.simplified_rte_20261111.run freeze outputs/simplified_rte_20261111 --parent outputs/miqp_loss_benchmark_20261111 --workers 4
python -B -m research_experiments.simplified_rte_20261111.run pilot outputs/simplified_rte_20261111 --workers 4
python -B -m research_experiments.simplified_rte_20261111.run status outputs/simplified_rte_20261111

# 四个原生结果返回后执行独立后分析
python -B -m research_experiments.simplified_rte_20261111.analyze outputs/simplified_rte_20261111
```

`freeze`只冻结源码、配置及父输入引用，不启动拟合。研究操作入口为`freeze/pilot/status`，`worker`由driver内部调用；当前没有完整48输入批次的`all`模式。

pilot固定使用`paper15`、`soumalas11`、`flynn16`、`pengwah18`，每个网络取`seed_index=0`、电压噪声标准差`.0005`，共四个配置。原始每输入数据仍为训练1×96、验证3×96、测试1×96个连续时点。

## 输出与证据

- `protocol.json`、`catalog.json`和源码快照：冻结参数、父输入与源码SHA。
- `jobs/<job>/result.json`：自身路径、W/T/E事件、零块清理记录、主MAE与次MSE选择，以及耗时。
- `candidates.json`：完整候选模型的支持、R/X系数、终端顺序和训练来源哈希；系数按训练数据冻结。
- 独立后分析：重建预测与拓扑、核对QP/MIQP目标及状态，保存失败和未完成分母。

所有后分析才能读取真值和测试观测。训练、候选生成及主验证路径均保留数据来源边界。

## 预算和对照口径

每个pilot逻辑输入名义总预算为14,400秒，父公共预处理成本按保存记录计入，并预留600秒用于输出。剩余结构预算的前半供W/T使用，E使用后续剩余时间。W单次未知支持MIQP上限3,600秒；T/E分别使用各自阶段剩余预算。公共参考尺度及提供的热启动家族来自已核验父记录。

旧T003只复用相同四个输入的保存结果。其初始化、末端释放、E采纳及`.24`输出映射与本版不同，且耗时来自历史执行环境。因此本轮给出实现核查和描述性对照，正式公平耗时或单模块因果比较需要另建一致的评估合同。
