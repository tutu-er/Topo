# 复算

从本工作树根目录运行，解释器为`D:/apps/miniconda3/envs/Topo/python.exe`。使用新输出目录；原主批为`outputs/local_rx_three_day_main_v1`，pilot为`outputs/local_rx_three_day_pilot_v1`。四进程、每进程单BLAS线程，配置保存实际版本与源码hash。

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& D:/apps/miniconda3/envs/Topo/python.exe -m pytest research_experiments/mobile_ev_mle/local_rx_three_day research_experiments/mobile_ev_mle/local_rx_identifiability -q --basetemp outputs/local_rx_three_day_tests_replay
& D:/apps/miniconda3/envs/Topo/python.exe -m research_experiments.mobile_ev_mle.local_rx_three_day.experiment --output outputs/local_rx_three_day_replay --workers 4
& D:/apps/miniconda3/envs/Topo/python.exe -m research_experiments.mobile_ev_mle.local_rx_three_day.audit --directory outputs/local_rx_three_day_replay
& D:/apps/miniconda3/envs/Topo/python.exe -m research_experiments.mobile_ev_mle.local_rx_three_day.analysis --directory outputs/local_rx_three_day_replay
```

U/R/K和3/7天都是配对，主批48个计算条件只对应8个独立随机族。R新增加带噪入口电压表，K为理想边界；U不得收到这些轨迹。真参数、真实P/Q及真实分叉电压只在封存后的误差评价使用，不能提供初值或选模型的阈值。

证据包保留完整汇总与审计、冻结科学源码、12个预选代表条件。所有原始记录在本地保留并逐文件hash；不覆盖上一轮证据。归档后实际恢复2个U/R配对条件并独立复查，恢复收据只覆盖这些状态，不称48条件全重跑。
