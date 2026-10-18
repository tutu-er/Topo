# 复算与证据边界

运行目录为本Git工作树根目录。已验证解释器：`D:/apps/miniconda3/envs/Topo/python.exe`；版本、依赖和19份冻结科学源码hash保存在配置与source_manifest中。设置OMP/OPENBLAS/MKL线程为1，四进程并行。

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& D:/apps/miniconda3/envs/Topo/python.exe -m pytest research_experiments/mobile_ev_mle/local_rx_identifiability -q --basetemp outputs/local_rx_tests_replay
& D:/apps/miniconda3/envs/Topo/python.exe -m research_experiments.mobile_ev_mle.local_rx_identifiability.experiment --mode main --output outputs/local_rx_replay --workers 4
& D:/apps/miniconda3/envs/Topo/python.exe -m research_experiments.mobile_ev_mle.local_rx_identifiability.experiment --mode bootstrap --output outputs/local_rx_replay --workers 4
& D:/apps/miniconda3/envs/Topo/python.exe -m research_experiments.mobile_ev_mle.local_rx_identifiability.audit --directory outputs/local_rx_replay
& D:/apps/miniconda3/envs/Topo/python.exe -m research_experiments.mobile_ev_mle.local_rx_identifiability.analysis --directory outputs/local_rx_replay
& D:/apps/miniconda3/envs/Topo/python.exe -m research_experiments.mobile_ev_mle.local_rx_identifiability.relative_bootstrap --directory outputs/local_rx_replay --workers 4
& D:/apps/miniconda3/envs/Topo/python.exe -m research_experiments.mobile_ev_mle.local_rx_identifiability.relative_audit --directory outputs/local_rx_replay
```

使用新输出目录，避免覆盖旧证据。原始主记录保存在`outputs/local_rx_main_v1/`；25MB量级完整记录本地保留，逐文件hash见`evidence/full_local_manifest.json`。Git仅保存完整统计、全部bootstrap记录及事前指定的代表性AC状态；代表记录包含两项主批未达驻点的条件。`source_snapshot.zip`保留准确科学源码和分析审计实现。

归档由以下命令创建，并实际恢复首个K/U配对条件，重新进行独立AC审计；`--destination`必须为空，`--restore`必须不存在：

```powershell
& D:/apps/miniconda3/envs/Topo/python.exe -m research_experiments.mobile_ev_mle.local_rx_identifiability.package --directory outputs/local_rx_main_v1 --destination research_experiments/mobile_ev_mle/local_rx_identifiability/evidence --restore outputs/local_rx_restore
```

`audit.json`检查完整48条件及全部保存拟合状态，`relative_audit.json`另查624个相对bootstrap统计量、噪声digest和输入hash；`restore_receipt.json`只证明选定代表状态与归档源码的实际恢复。范围分开，不将代表恢复称作全部科学计算重跑。浮点差、失败与局部优化限制均保留。归档前将最终pytest的JUnit结果保存到输出目录`unit_tests.xml`。

数据边界：拒绝器与bootstrap不读真实标签/R/X/P/Q；真值只在推断封存后用于误差、分解与信息上限诊断。K额外给定真实边界作为理想控制，U没有该量测。没有用本批真标签确定阈值，也没有输出校准后的拓扑正确概率。探索性相对bootstrap在原批结束后追加，应与事前主指标分开解释。
