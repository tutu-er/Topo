# 紧凑交付的离线 AC 审查范围

`ev_compact_slim_review.py` 只读取 `.slim.json.gz`、交付 manifest、freeze 和源码字节快照。它复用独立支路电流回代与电压前推算术，核对保存状态；不重新拟合参数，不运行 MILP 或选择动作，不读取 manifest 所记录的原 raw 路径。

## 审查什么

1. 交付 schema、每个 slim 压缩字节的 SHA-256 和长度、freeze/snapshot 文件哈希、策略身份、72 份当前冻结源码、阶段/种子及方法分母。原 raw 的 SHA、长度和路径在 slim 与 manifest 间核对；原 raw 未随包交付时，其内容不在本次离线审查范围。
2. 历史候选拟合和保留历史解释：物理 AC、计量残差、SSE、对数似然、R/X 单位和声明参数域。
3. 每个方法实际执行的未来物理响应、成对噪声计量、联合历史与未来的候选拟合、排名、似然差及无噪响应的单独拟合。
4. 保存的初始、原始建议和最终动作的路径、接线、移动、电池能量、效率、功率、吞吐和人员成本；候选来源和确定性继承。
5. 预测部分只核对保存的拟合 SSE 摘要、对数似然归一化、任务权重、历史基准和继承分数算术。预测状态与残差已省略，故这里没有预测拟合的 AC 重放。

令保存的总标准化平方误差为 \(J\)，计量标准差为 \(\sigma_i\)。检查的高斯对数似然为

\[
\ell=-\frac12\left[J+\sum_i\log(2\pi\sigma_i^2)\right].
\]

历史与实际未来的 \(J\) 由保存物理状态独立重算；预测摘要的 \(J\) 仅按保存值校核此公式。两者在 coverage 字段中分开。

## 明确省略

预测拟合的 AC 状态/残差、非 best QP 参数/残差、完整可行见证库和 LP 证书已从 slim 中省略。best QP 参数虽保留，本工具也不重建其矩阵或证书；同见证 AC 门的残差向量未重放。因此本审查不是完整 raw 重放。完整原审查器 `ev_compact_review.py` 和完整矩阵 QP 审查器的范围保持不变。

方法、拟合和预测异常保留分母。预测 AC 异常使用其声明的 `__forecast_or_joint__` 失败记录；该参考及全部有向任务仍计入覆盖，审查不补成成功任务。部分交付另列缺失历史，`success` 表示已交付范围的算术和来源核对通过，不表示缺失样本已完成。

## 新 checkout 的用法

先检出包含工具、研究源码和紧凑包的任务提交，并安装项目所需的 Python/Numpy/Scipy 环境。以下命令从仓库根目录运行，离线审查不需要 Gurobi 求解或许可证调用：

```text
python -m research_experiments.mobile_ev_mle.ev_compact_source_restore
```

默认检查 ensemble 开发交付的 72 个冻结源。若只报告换行差异，可显式恢复原字节：

```text
python -m research_experiments.mobile_ev_mle.ev_compact_source_restore --restore-line-endings
```

源码缺失或真实内容变更会拒绝整批恢复。它是换行恢复工具，不从快照安装缺失代码。需要先检出对应任务提交。

随后执行：

```text
python -m research_experiments.mobile_ev_mle.ev_compact_slim_review --directory research_experiments/mobile_ev_mle/evidence/ev_compact/delivery_development_v2/development_ensemble_v2 --output review_ensemble_new.json
python -m research_experiments.mobile_ev_mle.ev_compact_slim_review --directory research_experiments/mobile_ev_mle/evidence/ev_compact/delivery_development_v2/development_center_v2 --output review_center_new.json
```

输出使用新文件名，工具拒绝覆盖已有报告。绝对原 raw 路径只作为来源文字保留，不影响新 checkout 的运行。

## 开发包实测

两个交付目录均在 D 的独立研究目录执行了完整已保存范围审查，4/4 历史、44/44 方法行通过：

|保存状态/账目|ensemble v2|center v2|
|---|---:|---:|
|历史候选 AC 状态|16|16|
|保留历史解释 AC 状态|47|0|
|实际含噪联合拟合状态|176|176|
|实际无噪联合拟合状态|176|176|
|以上状态合计|415|368|
|实际未来物理响应|44|44|
|初始/建议/最终操作账目（含重复计划）|158|160|
|继承分数检查|52|52|
|预测拟合摘要（未重放其 AC）|3648|608|
|预测有向任务摘要|2736|456|

缓存导致同一实际曲线可出现在多个方法行；按每个历史内部去重后，每个目录合计 37 个动作记录，含噪及无噪各自记录。不同历史的相同曲线仍对应不同计量，不合并为一次拟合。状态/任务/情景计数都不增加独立历史数。当前工具 32 项针对性测试、加原 full reviewer/source wrapper/exporter 共 68 项测试通过。没有新拟合或新策略选择。
