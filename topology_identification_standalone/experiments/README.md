# 可独立运行的实验入口

本目录的实验调用共享的可读算法实现，不导入原项目包。48组历史观测已随上一级目录携带；新生成数据使用本目录自身的生成器。新运行结果属于独立重构版本，不应直接作为历史120输入或旧48输入分数的重复。

## 方法与消融

三位数字依次表示R、T、E开关。全部八格与两个树基线构成十种方法：

| ID | 含义 |
|---|---|
| `rnj` | RNJ层级及同界权重重拟合 |
| `nj` | 同灵敏度几何的经典NJ对照及重拟合 |
| `000` / `pure_wzzt` | 纯逐步WZZT，无R结构引导、T或E |
| `100` / `rnj_wzzt` | RNJ候选/热启动引导加逐步WZZT |
| `010`、`001` | 分别只开T、只开E |
| `110`、`101`、`011` | 分别开RT、RE、TE |
| `111` / `rte` | 简化RTE全部开启 |

R关闭指不使用RNJ的结构候选和热启动。全部消融共享训练数据估计的参考误差和几何前处理，故不应称`000`完全没有计算RNJ相关前处理。共同预处理只实际执行一次，成本对每种方法均单独计入。

T与E仍使用MIQP，固定支持族重拟合使用QP。所有结构目标统一为`MSE / reference_MSE + alpha * internal_count`。本重构不引入旧L1目标单位的K缩放；数值容差因此按统一归一化目标解释。当前最终按原始保存路径的严格验证MSE选模，规范化排序仅除以共同训练参考误差；MAE视图只作诊断，不参与采纳。没有输出.24收缩或完整RNJ整树备用。

## 命令

在上一级目录运行（也可使用`python -m topology_identification_standalone`的对应入口）：

```powershell
# 六终端线性小例：五种主方法，每方法30秒上限。
python run.py demo --output results/demo

# 小例的完整八格RTE消融，另含RNJ/NJ对照。
python run.py ablation --output results/ablation_toy

# 内置四网络各取seed_index=0、电压噪声.0005的固定pilot。
python run.py benchmark --jobs pilot --methods rnj nj pure_wzzt rnj_wzzt rte --output results/pilot

# 内置48组观测的全部十种方法。此命令计算量较大，须显式选择。
python run.py ablation --dataset bundled --jobs all --output results/full_ablation

# 本地生成器的新物理轨迹；与bundled历史观测不作字节一致声明。
python run.py demo --case paper15 --seed 9 --points 96 --time-limit 300 --output results/new_paper15
```

`--methods`支持空格或逗号分隔；`--jobs`支持一个或多个内置ID、`pilot`、`all`。`--alpha`默认.003，`--block-limit`默认5。默认内置算例每方法/输入14400秒，demo/小例消融30秒。短示例预算会同步缩小QP上限和输出预留，实际配置完整写入`plan.json`，不能把小例耗时当成全面基准比较。

## 输出与恢复

- `plan.json`冻结方法、输入、全部有效配置、当地源码和观测SHA。
- `runs/<job>/<method>.json`保存选中完整模型、路径、事件、状态、耗时及事后评分。
- `results.csv`每个计划方法/输入一行；失败、超预算及待完成仍占分母。
- `summary.json`分别给计划数、完成数、失败数、超预算数、有效完成样本F1均值、完整恢复数/计划数。超预算结果保留完整模型和求解记录，但不计入有效完成评分。
- 完整八格存在时，`ablation_effects.csv`给12种条件边际，`module_addition_orders.csv`给六种模块加入顺序的路径，`shapley.csv`给每模块的平均归因，`interactions.csv`给三组二阶交互在其余模块关/开时的差分及RTE三阶交互。负效应和不完整配对/单元分母均保留；六种加入顺序只是同一八格结果的读取方式，并非六套重新排序的执行流程。

所有方法先用相同训练/验证观测拟合。全部方法对一个输入拟合完毕后，才读取该输入的测试观测与物理真树评分；二者不传入算法。按同一命令再次运行会跳过已完成、失败或超预算的记录；不会自动给方法重置预算。改变参数或源码时应选择新的输出目录。

`tests/test_experiment.py`包含十种方法映射、分母保留、预算默认值、真假信息边界，以及复制并改名后以隔离Python路径、禁止父项目导入的实际RNJ/NJ运行检查；复制目录的全部本地哈希与内置观测读取也会核对。完整MIQP正确性与求解证据由共享求解器验证覆盖。

## 冻结的48输入并行完整批次

```powershell
python tools/full_benchmark.py freeze outputs/full48_mse_20261111 --workers 8
python tools/full_benchmark.py run outputs/full48_mse_20261111 --workers 8
python tools/full_benchmark.py status outputs/full48_mse_20261111

# 等480项全部终态后，独立审计、评分和绘图
python tools/analyze_full.py outputs/full48_mse_20261111
python tools/plot_full.py outputs/full48_mse_20261111
```

该入口计算48份全新公共预处理及480个全新方法配置，最终选择严格使用验证MSE。每输入每方法配置预算14400秒，最多8个单线程worker；1920方法小时是名义上限，不是实际耗时。公共计算物理上每输入一次、逻辑上每方法各记一次。详情见full48_mse/PLAN.md。

运行源与全部数据复制到新输出目录的runtime内，原始记录不可覆盖；存在没有终态的attempt时必须审计，不自动重置预算。独立后处理源码另外记录哈希，不在计算期间修改冻结模型。绘图额外需要requirements-plot.txt中的Matplotlib。
