# 主240的固定RNJ与旧联合候选库公平后处理

2026-11-16，执行前锁定。本批只做固定结构 QP 与保存候选的后处理，不增加 MIQP、AC、观测或灵活性实验。

## 输入与四个对照

只取 `load_catalog()` 中实际名为 `large96_main` 的全部240项，排除已退出的压力160；失败和不可读项仍占计划分母。每项只使用原同一个96点文件：前64点训练，后32点验证，没有另一天评价。

四个对照为 `RNJ1`、`RNJ15`、旧 `100` 库、旧 `101` 库，共960个计划方法项。

- RNJ1：共同前端的固定 `.75/.16` 参考族，只做固定结构拟合。
- RNJ15：三种 R/X 组合尺度、五档容差产生的15族，重复支持族去重后逐族固定拟合，以验证集择优。
- 100/101：读取原 `runs/inputXXX/{100,101}/result.json`，只使用已保存 `frozen_candidates` 中的点，保持原数目与顺序，不重搜、不追加、不用训练较好的新点替换旧点。
- 四者都包含基础支持模型和单节点保护模型；RNJ基线显式加入，100/101须核实旧库原本已有，缺失则失败而不补造。

## 共同拟合、收缩和选择

共同前端只加载64个训练点，新算训练参考 MSE 及 R/X 几何一次。对照旧封存 `common.json` 核训练指纹、参考族、连接池、Mref与几何矩阵；数值不一致时保留失败，不改旧库尺度。核对通过后，四法使用旧 Mref 的确切数值，以保持验证容差单位一致。

均采用原 `0≤r,x≤2` 有界最小二乘。所有输出点使用同一训练校准的 RX75 `.08` 非单节点收缩：包括公共项，豁免单节点项；每次删掉最弱违规项，训练 QP 重拟合，直至稳定。最终按同一严格验证 MSE 和原并列规则选择，原始与收缩后的结果均保留。

RNJ15的验证选择是明确的多候选基线。它的候选数可能大于单路径库，报告保存原始15族数、去重数和最终候选数，不将其称为单棵RNJ。

## 预算、来源与失败

每个直接基线/库后处理阶段总预算120秒，输出保留12秒，QP软限300秒，最多3个单线程worker；共同准备的实际时间在每个方法记账一次。固定QP的软限是返回后核验，超额保留。新脚本在进程内禁止创建结构求解器。

100/101的原方法耗时来自原封存wrapper，另列本次共同准备＋后处理时间。二者之和可以描述复用旧结果的总计算代价；本次后处理时间不能冒称重新运行100/101的耗时，也不能与原方法14400秒预算作算法加速结论。

prepare封存原selection seal、原100/101/common/input引用、当时源码引用、新运行源码快照、必要辅助模块、环境和协议。旧来源对照当时私有快照，不要求它与今天仓库源码一致。新源码发生变化则失败。已完成方法项按哈希复用；遇已有启动标记而无结果的部分尝试，记录失败，不自动重置其时间预算。

## 评价边界

全部960项都有终结收据并写入 `fit_complete.json` 后，评价器才打开物理分支和旧真拓扑文件。逐库从保存系数独立重算验证MSE、选择索引、内部与全支持错漏、候选覆盖、收缩前后选择，以及RNJ基线相对100/101的配对新增/丢失。方法失败保留在240分母，核验错误单列。

这是已使用主240输入上的补充贡献分析，没有独立新测试日。若简单RNJ基线经过相同后处理已达到联合库效果，应如实记录；如果联合库仍有优势，优势限定于本批保存库与同口径后处理，不能等同于MIQP相对实际热启动的逐次增益。

## 入口

```powershell
# 纯逻辑接线检查，不拟合，不运行MIQP或AC
D:/apps/miniconda3/envs/Topo/python.exe -m rnj_wzzt_dr.experiments.identification_contribution_20261116.legacy_baselines --self-test

# 仅封存；正式运行由根任务统一启动
D:/apps/miniconda3/envs/Topo/python.exe -m rnj_wzzt_dr.experiments.identification_contribution_20261116.legacy_baselines --out outputs/identification_contribution_legacy240_20261116_v1 --prepare-only

# 固定QP/后处理完成后自动独立评价；也可用--run-only、--evaluate-only分阶段
D:/apps/miniconda3/envs/Topo/python.exe -m rnj_wzzt_dr.experiments.identification_contribution_20261116.legacy_baselines --out outputs/identification_contribution_legacy240_20261116_v1 --resume
```

输出包含 `cases/<id>/<method>/native.json`、逐项结果/封存、`fit_complete.json`、`evaluation/report.json`、`evaluation/REPORT.md`、`evaluation/summary.csv` 和最终 `complete.json`。原数据、参数和细节保留在 ignored outputs。
