# RNJ / NJ / WZZT 独立研究目录

这个目录可以整体复制、改名，在另一位置单独运行。算法、四个网络定义、48组历史观测、算例生成器及消融入口都在目录内；默认运行不读取原项目代码、旧`outputs/`或网络数据。

采用当前简化平方误差模型：RNJ提供候选与热启动，原始WZZT逐步扩展，T进行小区域联合重构，E允许旧连接退出。W/T/E使用凸MIQP，固定拓扑的权重拟合使用有界凸QP。原项目和历史结果没有被覆盖。

## 快速运行

需要Python 3.10以上。完整运行安装：

```powershell
cd topology_identification_standalone
python -m pip install -r requirements.txt

# 六终端小例：RNJ、NJ、纯WZZT、RNJ+WZZT、完整RTE
python run.py demo

# 完整8格R/T/E消融，另含RNJ和NJ
python run.py ablation

# 检查数据和单元测试；完整检查另在隔离复制目录运行全部10种方法
python check.py --full-smoke

# 生成可搬走的整包，包含代码、网络和全部附带观测，排除开发outputs
python tools/package.py ../topology_identification_standalone.zip
```

结构MIQP需要已安装的Gurobi及有效许可。若只运行RNJ/NJ，安装NumPy、SciPy即可，无需Gurobi。第三方Python库和求解器通过运行环境提供，不把许可或二进制塞入目录；`requirements-tested.txt`保存本次验收使用的版本。

入口可从任意工作目录调用，也支持隔离模式，例如：

```powershell
python -I D:/my_copy/run.py demo --methods rnj nj
```

这里的路径替换成复制目录的实际位置；代码只加入自身目录，不搜索原项目。

## 方法和消融

| 方法 | 命令中的名字 |
|---|---|
| RNJ基线 | `rnj` |
| NJ对照 | `nj` |
| 纯WZZT | `pure_wzzt` 或 `000` |
| RNJ＋WZZT，关闭T/E | `rnj_wzzt` 或 `100` |
| 简化RNJ＋WZZT＋T＋E | `rte` 或 `111` |
| 完整消融 | `000 100 010 001 110 101 011 111` |

三位编号依次开关R、T、E。全部W类方法从单点/全集基底开始；关闭R会禁用RNJ结构候选和热启动，但消融仍共享训练参考尺度。RNJ/NJ各输出自身树的QP重拟合与数值清零结果；W类方法验证选择自身保存路径及单点保护模型，不加入独立完整RNJ整树备用。

共同结构目标为

$$
J(\mathcal F)=\frac{\operatorname{MSE}_{\rm train}(\mathcal F)}{M_{\rm ref}}
+\alpha\,k(\mathcal F),\qquad \alpha=0.003.
$$

每个结构先QP重拟合并在允许范围清零，再检查J是否超过数值容差下降。没有额外相对改善门槛、最近邻修复窗或最终0.24收缩。T块宽上限为5；R仍公开保留三种R/X几何与五档RNJ分组容差。全部有效参数写入每次实验的`plan.json`。

完整定义见[MODEL.md](MODEL.md)，对话中旧算法与当前代码的对应关系见[ALGORITHMS.md](ALGORITHMS.md)。旧小种子、旧L1-MILP及整树组合等探索版保留为历史谱系和结果快照，不继续堆进当前运行引擎。

## 使用附带的四网络数据

内置`paper15`、`soumalas11`、`flynn16`、`pengwah18`，共48组历史输入，包含144个原字节观测NPZ：训练96点、验证288点、测试96点。真实网络定义独立保存；算法仅接收训练/验证，测试和真树只用于事后评价。

```powershell
# 四网络固定pilot；完整预算默认每方法/输入4小时
python run.py benchmark --jobs pilot --methods rnj nj pure_wzzt rnj_wzzt rte --output outputs/pilot

# 48输入 × 10方法；需显式选择，计算量较大
python run.py ablation --dataset bundled --jobs all --output outputs/full_ablation

# 用本地径向AC生成器产生新的一天96点轨迹
python run.py demo --case pengwah18 --seed 9 --points 96 --time-limit 300 --output outputs/new_case
```

默认demo是六终端线性小例，每方法预算30秒，用于验证实现。内置历史面板逐字节保留；新AC生成器是可读的本地实现，其随机轨迹不冒充历史仿真的字节复现。网络、噪声、单位和输入哈希见[DATA.md](DATA.md)。

## 代码入口

```text
run.py                 命令入口
experiment.py          数据划分、逐方法运行、评分与完整消融统计
algorithm.py           唯一的R/W/T/E搜索流程及开关
optimization.py        固定族QP、未知支持W、共同T/E选择模型
trees.py               灵敏度估计、RNJ、NJ、层级与全部有界切面
data.py                观测读取、真实支撑提取及径向AC生成器
cases/                 四个物理网络定义
data/                  48组历史观测、种子与数据校验清单
tests/                 图算法、优化模型、信息边界与隔离测试
validation/            本次实现验收记录
reference_results/     独立保留的历史结果快照
check.py               数据、测试、复制改名后隔离运行的检查
```

代码不依赖Pandas、NetworkX、Pandapower或旧项目包。数据读取API也允许用户主动传入自己的文件；默认附带数据运行没有这些外部文件依赖。

## 实验结果与复现边界

输出保留逐输入/逐方法的计划分母、F1、完整恢复、MAE/MSE、耗时、求解状态和上下界。完整八格还能直接生成条件边际、二/三阶交互、Shapley及六种模块加入顺序表。负效应、失败、超预算和缺失格都会保留。

模型源码与数据在计划中保存哈希。同一命令可跳过已有终态记录；改变源码或参数需要使用新输出目录，失败方法不会自动获得全新预算。完整命令及输出说明见[experiments/README.md](experiments/README.md)。

总时间是计算预算而非可抢占的全过程硬保证：原生MIQP有TimeLimit；SciPy灵敏度估计和固定QP采用迭代限制，QP墙钟上限是返回后的软检查。超出总记账预算会标记`budget_exceeded`，不算有效完成。模型最优证书只覆盖每次声明的W或有限T/E候选域。

本目录是新的可读重构。它保留原始RNJ生成族的候选，不再依赖旧L1拟合后剪零的缓存；目标直接按MSE归一化。历史48/120输入的成绩不能直接标成这个新入口的运行结果。验收范围和已完成测试见[VALIDATION.md](VALIDATION.md)，来源与差异见[PROVENANCE.md](PROVENANCE.md)。
