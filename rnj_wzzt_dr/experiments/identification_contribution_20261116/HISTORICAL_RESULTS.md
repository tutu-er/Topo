# 早期 RNJ 与 ordinary 的同输入对照核查

核查日期：2026-11-16。只读取既有协议、原始模型和结果表，没有重新拟合、求解或运行潮流。

结论：早期确有“普通 RNJ＋WZZT 整流程优于独立 RNJ”的结果。主108输入中，原生 RNJ 的内部结构精确恢复为18/108，RNJ树库经联合 L1 拟合和原选模后为38/108，历史 `ordinary` 为49/108。这个优势早于近期 main240 测试，不能用当前 MSE `100` 的成绩替代。各方法的候选库、收缩及验证规则不同；其中有多少收益由 WZZT 新增连接单独产生，现有比较尚未完全分离。

## 输入范围与核验

唯一筛选清单采用 [rnj_four_methods 的 protocol.json](D:/0-github_workspace/Topo/outputs/rnj_four_methods_20261031/protocol.json) 中的 `jobs`，共120个；再用同目录 `catalog.json[job].panel == "main"` 取得108个主输入。不能直接使用整个 catalog，因为其中另有16个开发输入。

主108为4网络×3个 bundle×3档电压噪声×3档训练采样量。训练在三个物理日内均匀取合计24、48或96点，验证与测试各为另外三天、各288点。同一物理日的不同噪声和采样量不是独立网络样本。另12个早期附加输入为4网络×3条件（弱激励、根表噪声、异常值）；它们与后来的、已退出默认评估的 `pressure160` 完全不同。本说明以主108为结论依据，全120只用于核对旧报告；没有读取或采用 pressure160 的效果。

独立读取检查结果：

- 120个输入的 train/validation/test 共360个文件 SHA，均同时匹配四方法协议和原 `wzzt_paper` 数据 manifest；两批 payload 的文件路径也全部一致，训练 N 全部一致。
- 120份普通方法原始结果的 SHA 均匹配逐项 provenance；120份公共前端的 SHA 同时匹配协议与结果引用。
- 从保存的模型支持与各网络 `branches.csv` 重建内部真簇，复算240条 RNJ_native/RNJ_L1_bank 与120条 ordinary 的 F1，全部与原始 CSV 一致。
- 两个来源 CSV 的 SHA 分别匹配原 delivery_manifest 和 own-mainline manifest。此次没有重新运行旧审计程序。

## 主108结果与成对变化

F1针对非单叶、非全终端的内部下游集合；“精确恢复”也只针对该内部集合，不是完整线路或全部响应支持。

| 方法/记录口径 | 平均内部 F1 | 内部精确恢复 |
|---|---:|---:|
| RNJ_native，固定原生几何树 | 0.612022 | 18/108 |
| RNJ_L1_bank，联合 L1 拟合及原3%选模 | 0.714079 | 38/108 |
| ordinary，历史最终输出 | 0.753671 | 49/108 |
| ordinary，仅自身路径及单点候选 | 0.723577 | 45/108 |
| ordinary，未统一输出处理的最终路径点 | 0.600948 | 3/108 |

后两行是原记录已有视图。最后路径点没有经过相同的输出收缩及路径选模，不能拿它当作删除该模块后重跑的独立消融。

| ordinary 的比较对象 | F1提升/相同/下降 | 精确恢复新增/丢失 | 净增 |
|---|---:|---:|---:|
| RNJ_native | 80 / 25 / 3 | 31 / 0 | +31 |
| RNJ_L1_bank | 40 / 57 / 11 | 12 / 1 | +11 |

因此，“以前 R＋W 已经优于 RNJ”作为该历史整流程的描述有数值依据；“全部提升由 WZZT 扩展单独带来”没有被这组表证明。

全120复核值为：RNJ_native F1=0.599539、18/120；RNJ_L1_bank F1=0.700018、39/120；ordinary F1=0.746955、55/120。ordinary 相对二者的 F1胜/平/负分别89/27/4、47/61/12，精确恢复新增/丢失分别37/0、17/1。

## 当时三个方法实际做了什么

**RNJ_native**：固定 `raw_RX75_0.24` 的 RNJ几何树，保留原生结构。为比较参数和预测误差，仍对该固定树进行联合 L1 拟合；这是绝对残差的线性规划，不是当前 MSE 固定支持 QP。结构评分保留原生支持，即使某条支持的拟合系数为零也不据此改写原生拓扑。来源：[旧 pipeline.py:7](D:/0-github_workspace/Topo/outputs/wzzt_paper_20261029_full/source/research_experiments/wzzt_fine/pipeline.py:7)、[pipeline.py:32](D:/0-github_workspace/Topo/outputs/wzzt_paper_20261029_full/source/research_experiments/wzzt_fine/pipeline.py:32)。

**RNJ_L1_bank**：多个原始量测 RNJ树分别联合 L1 拟合，验证 MAE 位于最小值的3%以内时优先少支持；先选模型，再按 `.16 × median_root_depth` 收缩。该旧版本收缩时保留剩余系数、重算分数，并没有像 ordinary 那样在每个收缩家族上重新求解。来源：[旧 core.py:16](D:/0-github_workspace/Topo/outputs/wzzt_paper_20261029_full/source/research_experiments/wzzt_refine/core.py:16)。

**ordinary**：完整 RNJ家族加单叶和全终端响应基作为初始结构；每步选择一个相容的新集合，同时联合拟合所有系数，目标为平均绝对误差，无线路数惩罚。没有预联合、末端联合或E退出。结构总预算300秒、每次 MILP/LP最多30秒；训练接受门槛为 `1e-10 + .001 × 当前MAE`。最终从公共 RNJ整树库、自身路径和单点候选共同选择：每个候选统一`.24`收缩，固定家族 L1重新拟合，再按严格验证 MAE选择，`1e-12`同分时优先内部连接少者。来源：[ordinary_search:25](D:/0-github_workspace/Topo/outputs/rnj_four_methods_20261031/source/research_experiments/rnj_four_methods/run.py:25)、[最终库:47](D:/0-github_workspace/Topo/outputs/rnj_four_methods_20261031/source/research_experiments/joint_exit/run.py:47)、[收缩/选择实现:9](D:/0-github_workspace/Topo/outputs/rnj_four_methods_20261031/source/research_experiments/joint_exit/selection.py:9)。这四个运行源文件的字节 SHA 均与冻结协议一致。

ordinary 主108的最终来源标签为98个公共完整RNJ、2个自身初始点、8个W扩展点；全120分别107、2、11。该标签来自原 `choices.strict_0.24.bank` 的 index/origin 和路径事件，只描述被选中的记录位置。即使标签是公共RNJ，算法也已实际执行并计入其搜索成本；标签本身不能判定W是否产生了相同结果。

| 主108最终来源标签 | 输入数 | 平均内部F1 | 内部精确恢复 |
|---|---:|---:|---:|
| 公共完整RNJ | 98 | 0.755559 | 47/98 |
| 自身初始点 | 2 | 0.916667 | 1/2 |
| W扩展点 | 8 | 0.689789 | 1/8 |

对98个公共RNJ标签逐项比较 `choices.strict_0.24.bank.point` 与同一记录的 `path.point`：**67个point字典逐值完全相同**，包括支持、R/X系数及训练目标，验证MAE也完全相同。其中58个自身路径选择初始点，另外**9个自身路径选择的是W扩展点**。剩余31个公共输出与自身路径结构不同，公共输出的验证MAE严格更低；这31个输入中其优势不是完全重复记录造成的。

原因是最终库顺序为“公共RNJ、路径、单点”；严格选模先取验证MAE最小值加`1e-12`内的候选，再按`(内部连接数, 验证MAE, 原索引)`排序。完全相同的重复模型因索引在前，被标为公共RNJ。因此，“98个public标签＝98个W没有作用”不成立；“只有8个W来源＝只有8次W产生有用模型”也不成立。源码见上述 `joint_exit/run.py:53` 与 `selection.py:26`，本核查没有改变候选顺序或规则。

8个最终明确标为W扩展的输入，相对其自身对应的 RNJ_native 和 RNJ_L1_bank 均为F1提升7、相同0、下降1，内部精确恢复新增1、丢失0。两种基线在这个8例子集上的F1分别为0.409557和0.566865，均0/8精确恢复。这是已被最终选择的W输出子集，存在选择条件，且仍有下节所列处理差异；不能用其7/8提升率代表全部输入上的W独立成功率。

旧报告还保留普通方法20/120输入的 `no_gain_unresolved`；它们仍有输出并计入分母。前向子问题或流程完成不等于整个未知拓扑问题全局最优。普通方法原输出未采用当前 `.08` 内部收缩规则。

## 尚未具备的严格同处理对照

最直接的对照应当只允许公共 RNJ树库与同一单点候选，采用与 ordinary 完全相同的`.24`收缩、收缩后 L1重拟合和严格验证 MAE。现有已读文件没有该对照的120份完整选中结果；原 ordinary 只保存最终选中模型，LP日志没有保存全部候选重拟合系数。因此本次没有重新拟合并补出该成绩，也没有将 RNJ_L1_bank 冒充这个对照。

主要未统一因素有：单棵固定RNJ与多树候选库；`.24`原生几何、`.16`系数收缩与`.24`系数收缩；收缩前还是收缩后评分；是否收缩后重拟合；3%验证容差还是严格最小验证MAE；以及最终公共整树备用选择。普通路径包含完整 RNJ初始点，故“仅自身路径”也不等于不使用 RNJ。

此外，旧 RNJ_native 按原生支持评分，而 ordinary 按 `max(r,x)>1e-9` 的有效内部支持评分。为检查这项差异，单独从原保存系数把 RNJ_native 也按同一正权条件计分，主108变为F1=0.623483、19/108，全120为0.612422、20/120；RNJ_L1_bank计数不变。ordinary 的优势仍在，但这项算术对照仍未统一其余输出处理。

## 来源与复算

原始模型引用：

- RNJ：[wzzt_paper 输出根](D:/0-github_workspace/Topo/outputs/wzzt_paper_20261029_full)，逐项 `jobs/<job>/result.json` 的 `models.RNJ_native`、`models.RNJ_L1_bank`。
- ordinary：[四方法输出根](D:/0-github_workspace/Topo/outputs/rnj_four_methods_20261031)，逐项 `evaluation/<job>/ordinary/result.json` 的 `choices.strict_0.24.bank.point`，自身路径视图为同位置的 `path.point`。来源/输入哈希分别在相邻 `provenance.json`、协议与公共记录内。
- 原 RNJ逐项表：[per_job.csv](D:/0-github_workspace/Topo/archive/legacy_research_20261112/legacy/docs/wzzt_paper_20261029/per_job.csv)。
- 普通方法逐项表：[per_input_quality.csv](D:/0-github_workspace/Topo/archive/legacy_research_20261112/legacy/docs/own_mainline_four_methods_20261101/per_input_quality.csv)；多视图：[per_model.csv](D:/0-github_workspace/Topo/archive/legacy_research_20261112/legacy/docs/rnj_four_methods_20261031/per_model.csv)。
- 结构真值从 catalog 指定的 `truth_network/branches.csv` 求每个支路下游终端集合，仅用于上述选后评分。

| 文件（仓库相对路径） | SHA-256 |
|---|---|
| outputs/rnj_four_methods_20261031/protocol.json | `97c43d6d8076964b4e56fb2b32208988f89e02fa4c83e6e634780dca3f22c658` |
| outputs/rnj_four_methods_20261031/catalog.json | `54efbc7a575e587d69cc86f07b38c67470da3938133ff7174055a45c7e022754` |
| archive/legacy_research_20261112/legacy/docs/wzzt_paper_20261029/per_job.csv | `47c6e4d34a6eed1e49f84e6faeebbebfe55dadf9cd192bcbcd34e56d3e556a9d` |
| archive/legacy_research_20261112/legacy/docs/own_mainline_four_methods_20261101/per_input_quality.csv | `662c076fb0ea81100413c144992cde22f9496382e49246f735958627c6d26882` |
| outputs/wzzt_paper_20261029_full/manifest.json | `3e8992bac99f929503884358353aeba27a9f61539a6ddce54d831f7290fa4d7d` |

以下纯CSV/JSON聚合可复算主表和成对计数，不导入任何项目求解器：

```python
import csv, json
from pathlib import Path

root = Path("D:/0-github_workspace/Topo")
docs = root / "archive/legacy_research_20261112/legacy/docs"
load = lambda p: json.loads(p.read_text(encoding="utf-8"))
table = lambda p: list(csv.DictReader(p.open(encoding="utf-8-sig")))
protocol = load(root / "outputs/rnj_four_methods_20261031/protocol.json")
catalog = load(root / "outputs/rnj_four_methods_20261031/catalog.json")
ids = [j for j in protocol["jobs"] if catalog[j]["panel"] == "main"]
assert len(ids) == 108
rnj = {(r["job_id"], r["variant"]): float(r["clade_f1"])
       for r in table(docs / "wzzt_paper_20261029/per_job.csv")}
ordinary = {r["job"]: float(r["clade_f1"])
            for r in table(docs / "own_mainline_four_methods_20261101/per_input_quality.csv")
            if r["algorithm"] == "ordinary"}
for method in ("RNJ_native", "RNJ_L1_bank", "ordinary"):
    values = [ordinary[j] if method == "ordinary" else rnj[j, method] for j in ids]
    print(method, len(values), sum(values) / len(values),
          sum(v >= 1 - 1e-12 for v in values))
for method in ("RNJ_native", "RNJ_L1_bank"):
    pairs = [(rnj[j, method], ordinary[j]) for j in ids]
    print(method, "F1 wins/ties/losses",
          sum(b-a > 1e-10 for a, b in pairs),
          sum(abs(b-a) <= 1e-10 for a, b in pairs),
          sum(b-a < -1e-10 for a, b in pairs),
          "recovery gained/lost",
          sum(b >= 1-1e-12 and a < 1-1e-12 for a, b in pairs),
          sum(a >= 1-1e-12 and b < 1-1e-12 for a, b in pairs))
```

这些是既有合成数据上的历史对照，已用于多轮方法设计，不是新盲测，也不作为2026-11-16新简化RNJ流程的成绩。
