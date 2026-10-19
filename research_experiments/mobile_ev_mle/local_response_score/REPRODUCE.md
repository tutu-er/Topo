# 复算

在本工作树根目录，使用`D:/apps/miniconda3/envs/Topo/python.exe`。每次指定新输出目录，程序拒绝覆盖主批。需要旧完整AC原始输出`outputs/local_rx_three_day_main_v1`；它的22份冻结科学源码由主程序逐hash核对。

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& D:/apps/miniconda3/envs/Topo/python.exe -m pytest research_experiments/mobile_ev_mle/local_response_score research_experiments/mobile_ev_mle/local_rx_three_day research_experiments/mobile_ev_mle/local_rx_identifiability -q --basetemp outputs/local_response_tests_replay
& D:/apps/miniconda3/envs/Topo/python.exe -m research_experiments.mobile_ev_mle.local_response_score.experiment --output outputs/local_response_replay --workers 4
& D:/apps/miniconda3/envs/Topo/python.exe -m research_experiments.mobile_ev_mle.local_response_score.audit --directory outputs/local_response_replay
& D:/apps/miniconda3/envs/Topo/python.exe -m research_experiments.mobile_ev_mle.local_response_score.physics --directory outputs/local_response_replay
& D:/apps/miniconda3/envs/Topo/python.exe -m research_experiments.mobile_ev_mle.local_response_score.benchmark --directory outputs/local_response_replay --workers 4
& D:/apps/miniconda3/envs/Topo/python.exe -m research_experiments.mobile_ev_mle.local_response_score.analysis --directory outputs/local_response_replay
```

主批28份科学源码hash、旧输入hash、环境、所有停点及失败状态在配置和记录中。4方法的64个训练候选均合格；QP外域点未伪装为原物理域可行点。计时以新配对重放为主，旧缓存费用单列。

数学验收与开发修复保留：pilot_v1的QP求解间隙不合格，经活跃面KKT精修后pilot_v2/v3均合格；不改主批科学域或按真标签调目标。广回归最初有10个临时目录权限错误，保留XML；指定工作区basetemp后通过。后处理余项中心对齐错误也保留初版文件，修正版不改变主批评分或QP下界。

代表包覆盖两网种子33001的3/7天4条件，包括当期含噪输入、4法全部状态及对应旧AC记录；另实际恢复paper15的2条件，用归档源码重算原始计量、物理约束、解析消元与QP对偶。恢复不是重跑16条件优化。完整原始输出本地保留，逐文件hash可核对。
