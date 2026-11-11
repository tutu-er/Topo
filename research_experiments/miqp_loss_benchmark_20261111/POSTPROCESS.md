# 后分析与归档

这些助手在优化源冻结后单独加入，未复制回outputs/miqp_loss_benchmark_20261111/source。原生模型继续执行冻结的run.py和squared_backend.py；实际后分析代码会另存analysis_source并记录哈希。

pilot返回8个配置后，从仓库根目录运行模块入口：

D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.miqp_loss_benchmark_20261111.analyze outputs/miqp_loss_benchmark_20261111 --workers 2

pilot阶段通过意味着8个已完成新配置均verified且invalid/failed为0，不等于全96完成。完成这些检查后再启动all，其调度器保留已完成pilot、不重跑原生结果。不要在pilot管理进程尚未结束时并行再开另一个调度器。

all运行期间可启动一次性等待助手：

D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.miqp_loss_benchmark_20261111.finish outputs/miqp_loss_benchmark_20261111 --workers 2

助手不启动优化、不新增策略或循环自动化。它等待全部96新配置返回或失败，然后调用模块化独立审计；只有96新与96父L1均完整核验通过才绘图。三个PNG/SVG图组分别显示主MAE验证下的损失×结构比较、噪声分层、同库更换验证指标的变化。

人工查看三组PNG后，在docs/miqp_loss_benchmark_20261111/visual_review.json记录通过及图文件哈希，再执行：

D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.miqp_loss_benchmark_20261111.deliver outputs/miqp_loss_benchmark_20261111

归档检查源、输入、数值回执、CSV与图的对应哈希；不打包原始NPZ、密钥或许可证。父L1的16MB归档只保存SHA引用，另纳入必要的96份配对结果，不复制嵌套ZIP。新原生fresh成本、共同父预处理、新QP共同预处理分别记账；队列等待、后分析和基础设施中断另行解释。

本次主机重启发生在78/96已返回时。8个未完成原生进程的证据已保存在
`outputs/miqp_loss_benchmark_20261111/recovery/host_restart_20261111/`，其余10个配置当时尚未启动。
恢复账本以已验证的后续系统启动时间为终止上界，从各任务启动前payload时间计到该时点；
8次中断合计保守上界为38,396.952543秒，停机等待不计入。恢复运行沿用原模型、数据和冻结源码，
各配置剩余成功尝试预算为14,400秒减去自身上界扣款，尚未启动者保留原预算。
恢复的进程核查及操作记录见`docs/miqp_loss_benchmark_20261111/RECOVERY.md`；本文不替代该记录。

`deliver`只在96个新配置与96个父L1配对结果全部核验完成后执行。它会将完整`recovery/`
子树纳入`records.zip`，保留78份未变结果哈希、8个中断目录、账本及恢复记录；实际执行的
`capture_recovery.py`、`recovery_run.py`、`recovery_worker.py`随`actual_driver/`一并归档。
原优化`source/`及`protocol.json`保持原字节，后分析助手与恢复包装器分别记录来源。

`timing_components.csv/json`分开保存以下口径：

- `successful_total_seconds`：最终成功尝试的fresh时间，加一次父共同预处理和一次新QP共同预处理。
- `new_MIQP_interrupted_wall_upper_seconds`：仅本轮MIQP主机重启的单配置保守扣款。
- `operational_total_seconds`/`operational_total_upper_seconds`：成功总时间加该扣款一次，是含中断的保守上界，不是精确CPU耗时或整批日历耗时。
- `parent_L1_*`：旧L1配对结果原有的成功时间和中断代理估计，单列保留，不加进新MSE总时间。旧L1代理估计与新MIQP启动时点上界来源不同。

共同预处理已经在成功总时间内，恢复扣款不得再叠加一次共同预处理；也不得在
`logical_total_seconds`或`operational_total_seconds`上再次加扣款。归档会从不可变账本独立
核对8次扣款、成功与运行总时间恒等式、每个原14,400秒上限及全部保全文件哈希。
中断前的尝试作为独立基础设施证据保留，最终成功尝试的solver统计与这些中断记录分开解释。

当前主训练损失虽为平方误差，导出仍保留真实MAE历史评分，且actual selected anchors与portable pool selectors分开。不能把base导出的整池择优视图冒充原生.24输出库的最终选择。
