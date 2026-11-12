# 本批次运行与恢复索引

主输出：`outputs/large96_main_20261112_001`。240个新输入、2400个方法配置，协议SHA为`ceac935650e280105c507e26ce780191fde4340cdd3a8fd3ba29028f8a6a8fc4`。算法/生成器/运行器已在`5610f28c`提交，实际运行以runtime中逐文件SHA为准。后处理在`c97f38b5`、`5d2d40fd`提交，不改变冻结算法。

首24输入的240个方法已全部成功，1420次原生求解，无TimeLimit或界冲突。耗时证据在`sentinel_operational_review.json`；此阶段未读真树/F1。全量经理随后自动接续，当前唯一主经理的桌面exec session为46447；前一哨兵session17750已完成。实时状态以输出目录的state.json、run.lock、manager_attempts和终态文件为准，本文/ RUN_STATUS只是有时间戳的恢复索引。

## 运行中的检查

```powershell
python -I -B tools/large96.py status outputs/large96_main_20261112_001
Get-Content outputs/large96_main_20261112_001/full_manager.log -Tail 20
```

不要在已有经理或原生worker存活时再开经理；不要重跑已终态方法。未终态attempt必须审计PID、时间与已消耗预算，不通过删除lock/attempt重新发满预算。若源码或物理case发生变化，当前runtime仍保持原字节；不得用父目录新源码替换。

## 主批次全部终态、封存之后

```powershell
python -I -B tools/large96.py truth outputs/large96_main_20261112_001
python -I -B tools/analyze_large96.py outputs/large96_main_20261112_001 --bootstrap 2000
python -I -B tools/plot_large96.py outputs/large96_main_20261112_001
python -I -B tools/diagnose_large96.py outputs/large96_main_20261112_001
```

truth仅从冻结evaluation_cases生成proper clades；独立分析在pre_truth_audit通过后评分。诊断对每个可用输入只做一次真支持族QP，原64点训练、原32点评价，另记成本，不回流选择。无实际DR执行。

## 预定压力扩展

先核对父目录7份运行源码/物理case仍与主协议一致；若漂移，隔离恢复冻结副本，而非覆盖其他任务改动。压力配置按PLAN提前固定，不按主试验成绩挑选。

```powershell
python -I -B tools/large96.py freeze outputs/large96_pressure_20261112_001 --cohort pressure --workers 8
python -I -B tools/large96.py run outputs/large96_pressure_20261112_001 --workers 8
# 全部640项封存后，同样truth、analyze、plot、diagnose。
```

压力160输入×000/100/101/111四方法，replicate100..104，不使用主试验0..19日轨迹。主/压力分开报告，不将不同方法分母混合。

## 交付

保存主比较、完整八格、分噪声/网络、配对区间、失败与原生超时/界冲突、错误单输出和DR候选价值。双候选含真族仅是潜在复核价值，不是已实现的DR恢复收益。复制实际后处理源码到各run的postprocessing/source/SHA后，用publish_large96.py对两run作合法ZIP分卷及全成员SHA/CRC核验；旧记录不覆盖。

实验目录的报告/图表和归档清单可跟踪；原始运行输出由原始分卷保存。最终更新bundle_manifest、便携整包、RUN_STATUS并按任务范围commit/push。原rnj_wzzt_core中其他任务改动不得夹带。既有77项回归证据在validation/large96_20261112，若后处理发生实质修复需补对应验证。
