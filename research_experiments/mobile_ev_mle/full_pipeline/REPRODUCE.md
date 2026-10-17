# 还原、核验和复算

本批已经完成，不需要重新启动controller。先读[结果报告](report.md)、[事前方案](protocol.md)、[数学口径](math.md)和[检查点](checkpoint.json)。新结果不得写回本批原始记录或调整冻结评价。

## 证据内容

- `evidence/full_pipeline_v1_evidence.zip`：136份精简全分母记录、原始运行文件哈希、864份冻结源码/数学文件、四个预定开发完整物理回放样本、输入窗、公开来源归属、环境、校准阈值、统计、AC审计。不是只保留成功结果。
- `evidence/full_pipeline_v1_supplement.zip`：完整树F1、候选漏真原因、预测/参数/排序/成本分析、独立后处理工具、阈值冻结时间、117项测试日志及还原验证。每个成员含字节哈希，JSON大文件仅作gzip压缩。
- `evidence/package_receipt.json`、`supplement_receipt.json`：压缩包字节哈希和验证结果；`delivery_validation.json`核对报告主表与两套统计；`restore_validation.json`保存还原检验详情。

完整原始输出仍保留于执行副本`D:/0-github_workspace/Topo/outputs/ev_full_pipeline_stage_20261018/outputs/full_pipeline_v1`，未把重复的大型AC轨迹全部提交。主包的精简记录足以重算所有主统计、日期簇区间及预定空间消融；补充工具由这些记录计算F1和覆盖。物理审计可从四个保存的完整开发样本独立回放。

## 已记录环境

Python 3.11.15，NumPy 2.4.6，SciPy 1.17.1，pandas 3.0.3，Gurobi 13.0.3。完整环境和实际导入路径见主包`provenance/environment.json`。建议使用记录版本完成严格数值复现。本次跨Python版本还原已观察到约10⁻¹⁵的求和尾差；计数、种子和规则仍须逐项一致，数值容差核验应保留差异记录。

## PowerShell示例

从源码根目录执行，先将下面Python路径换为记录环境的实际可执行文件。还原目录必须尚不存在，工具拒绝覆盖。

```powershell
$evPython = 'D:/apps/miniconda3/envs/Topo/python.exe'
$evDelivery = (Resolve-Path 'research_experiments/mobile_ev_mle/full_pipeline').Path
$evArchive = Join-Path $evDelivery 'evidence/full_pipeline_v1_evidence.zip'
$evRestore = Join-Path (Get-Location).Path 'outputs/full_pipeline_restore_local'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& $evPython (Join-Path $evDelivery 'postprocessing/package_evidence.py') verify $evArchive
& $evPython (Join-Path $evDelivery 'postprocessing/package_evidence.py') restore $evArchive $evRestore
& $evPython (Join-Path $evRestore 'postprocessing/package_evidence.py') replay $evRestore --source-root (Join-Path $evRestore 'source')
```

重放只读保存记录，不重新拟合线路、不规划EV，也不生成新噪声。它重算校准阈值、三阶段全部计数、日期簇bootstrap区间和预定空间比较，与包内统计逐项比对。约需要数分钟。

补充包先核对`evidence/supplement_receipt.json`中的archive_sha256，再提取到新的目录，按`manifest.json`逐个核对成员。成员路径不应出现绝对路径或`..`。其中`postprocessing/final_clade_metrics.py`、`candidate_coverage_review.py`和`results_interpretation.py`为独立后处理；原始科学策略没有因此改变。

## 测试命令

在冻结源码根目录，设置PYTHONPATH为该根和rnj_wzzt_core，线程变量同上：

```powershell
$env:PYTHONPATH = (Get-Location).Path + ';' + (Join-Path (Get-Location).Path 'rnj_wzzt_core')
& $evPython -m pytest research_experiments/mobile_ev_mle/full_pipeline research_experiments/mobile_ev_mle/g_precision/test_coefficients.py research_experiments/mobile_ev_mle/g_precision/test_context.py -q -p no:cacheprovider
```

交付复查结果：117 passed。测试包括因果预测、候选来源、操作可行性、平方上下界、AC梯度、完整失败分母、源盲阈值和恢复校验。冻结策略身份、原CSV来源和实际窗口以证据文件为准，不使用外部未记录的最新代码替换其依赖。
