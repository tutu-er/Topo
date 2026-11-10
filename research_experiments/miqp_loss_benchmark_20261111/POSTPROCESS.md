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

当前主训练损失虽为平方误差，导出仍保留真实MAE历史评分，且actual selected anchors与portable pool selectors分开。不能把base导出的整池择优视图冒充原生.24输出库的最终选择。
