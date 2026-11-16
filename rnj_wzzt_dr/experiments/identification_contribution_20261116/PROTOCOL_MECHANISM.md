# 当前101代码：一个已见输入的机制核验

2026-11-16，执行前锁定。本批只有一次新的101结构搜索，不增加AC、DFS、噪声路径、候选真值拟合或RNJ15基线重拟合。

## 为什么选这一项

主240同后处理分析已完成960/960项。已观察到58个输入满足：`.08`后的RNJ15完整候选库不含真内部结构，而旧101库的最终选择内部正确。按输入ID升序固定取首项，不尝试第二项来替代失败。

资格使用了已知评价结果。因此这是针对机制的事后诊断，不是随机抽样、独立确认或总体恢复率比较。RNJ15基线直接复用这项已保存的native参数和收缩后候选库。

## 输入、源码及预算

训练与验证仍是原同一96点文件的前64和后32行。prepare另存仅包含P/Q/Y/labels的两个NPZ，并核其指纹与原记录一致。没有独立新日。拟合worker的inputs仅列这两个观测文件；完整资格报告与物理真值入口放在分开的评价文件中。

复用 `outputs/identification_contribution_20261116_v1/source_snapshot/rnj_wzzt_dr` 的完整私有源码及封存哈希：W单连接QP预试、未知支持MIQP、E单固定RNJ参考、`.08`非末端收缩，方法101。调用已有 `rnj_pool_only_20261116.run.fit_one`，不改算法。

总预算300秒，配置 `per_mip_seconds=30`、输出保留30秒、QP软限300秒。原快照中该per-MIP字段用于W的单次调用；E仍使用剩余结构阶段预算，可能超过30秒，不能将它报告为每次原生求解都限30秒。固定族QP是返回后软限核验。实际耗时、原生状态、界与未闭合gap完整保留。

当前101正常共同前端仍会拟合其参考族以确定训练尺度；“不重拟合RNJ15基线”指不重做15族独立对照，也不修改保存的RNJ15候选。

## 验收与输出

所有输入和源码先封存，一次拟合返回成功或失败后才打开物理分支进行独立评价。分别记录：

1. 保存RNJ15库确实不含真内部结构；
2. 当前101最终候选库是否找到该结构；
3. 最终验证选择是否内部正确、是否全支持正确；
4. 真结构是否在原生MIQP返回阶段已经出现，还是经后续QP/收缩形成；
5. 初值、QP预试、原生返回及最终输出各自的作用。

代码/数值核验通过与辨识机制成功分开。若限时未找到，保留该结果，不把它写成搜索域中不存在真结构；也不继续挑选有利输入。原生返回初值、未闭界及费用时间都保留。

```powershell
# 根任务完成当前批次后，先封存这一项
D:/apps/miniconda3/envs/Topo/python.exe -m rnj_wzzt_dr.experiments.identification_contribution_20261116.verify_current_mechanism --out outputs/identification_current_mechanism_20261116_v1 --prepare-only

# 已封存后：一次拟合，再独立评价
D:/apps/miniconda3/envs/Topo/python.exe -m rnj_wzzt_dr.experiments.identification_contribution_20261116.verify_current_mechanism --out outputs/identification_current_mechanism_20261116_v1 --resume
```

也可使用 `--fit-only`、`--evaluate-only` 分阶段执行。已完成结果可复用，部分求解目录不自动覆盖或重置预算。详细输出在新ignored目录，不修改原主240、RNJ15或8输入记录。
