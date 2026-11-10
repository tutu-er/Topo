# 平方误差对照恢复索引

工作区D:/0-github_workspace/Topo，任务分支codex/iterative-matched120-20261101。实现提交34b2e5eb；原L1已144/144返回且独立核验通过。

新目录outputs/miqp_loss_benchmark_20261111，配置与源已冻结，禁止改写source。观测直接引用父day96目录的同48输入，父253份源码逐字保留，仅新增平方误差后端及研究入口。训练连续一天96点，验证288点，测试另一天96点。

原生内部损失为K*MSE，K=f0_MAE/f0_MSE；K仅由同一训练参考家族的两种最优拟合得到。权重界仍[0,2]，没有ridge或正阻抗下界。所有训练拟合及结构目标一致使用新损失；主输出保持MAE验证，另一MSE验证视图只重选已保存模型。

固定8配置pilot已启动：四网络、bundle1000、中噪声，每输入noT/T003。当前阶段不要同时启动all调度器。先等待pilot管理进程退出，再运行独立analyze并确认8个新配置均通过，保留总计划96分母；通过后再启动剩余88配置。各worker的attempt.json记录进程与状态，中断的未结束尝试不能无记账重跑。

每配置总预算14400秒，包括父共同预处理和新QP共同拟合；原始MIQP单步3600秒、固定族QP300秒、前向7200秒且最多半结构余量，输出保留600秒。T和E用阶段剩余预算，单线程、最多8配置并行。名义上限384方法小时，队列等待和后分析另记。

status命令查看全96配置返回数；pilot_completion说明实际pilot8的范围。新package README给出freeze/pilot/all/analyze命令。运行状态文件位于输出根目录，进度及错误在jobs下。Windows状态读取应允许FileShare.Delete，勿仅用沙箱进程视图判断宿主进程已退出。

手算与冻结源核验见backend_verification.json、backend_frozen_verification.json：固定族QP、原始前向、T与E均通过，误差非零例验证了纯平方损失而非L1+L2混合。

后分析、绘图和归档助手是冻结后新增的独立工具，实际使用版本应另存analysis_source并记录哈希，不回写冻结优化源。最终完成后在本页更新完整分母、核验、图与归档状态。
