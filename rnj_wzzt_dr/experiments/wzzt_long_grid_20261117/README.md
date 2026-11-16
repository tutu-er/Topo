# WZZT优势、训练损失和运行接口

本批回答三个问题：WZZT在哪些观测条件下比RNJ恢复更多连接；MAE/MILP与MSE/MIQP的精度和耗时差多少；
时间究竟花在找更好的连接，还是证明当前初值已经足够好。先锁[详细协议](PROTOCOL.md)，再运行；默认仍为MSE。
启动前的[执行补充](EXECUTION_ADDENDUM.md)将RNJ直接基线的外层预算与同面板101对齐，消除原120秒上限造成的完成机会差异。
当前已核验的[前四对结果](EARLY_PAIRED.md)与[启动及首例贡献记录](RUN_STATUS.md)均为阶段报告；完整批次仍按原计划运行。

## 先明确比较对象

推荐完整链为`101`：RNJ生成连接池和参考初值，W逐步增加未知连接，E在有限池内可撤销重构，
训练侧删除低于.08相对强度的非末端连接，最后按独立验证MSE选择。T关闭，避免再加一层局部重构。
RNJ1使用一个固定RNJ参考族；RNJ15使用原15档RNJ族的去重结果。三者都用相同系数界、保护模型和输出收缩。
WZZT的最终库没有独立完整RNJ整树兜底。

WZZT不会凭方法名称自动获得根部优势。[数学核查](ROOT_IDENTIFIABILITY.md)解释其可能优势及反例；
评价分别列近根内部线路、公共上游和末端，避免把内部恢复当成整树恢复。

## 固定数据与预算

四个网络，实测同源P/Q，共192份不同输入。主网格用48/96/192/384/768点拟合，192点验证，第11日96点评价；
附加P/Q误差标准差为0/0.5%/1%/2%，每格两个噪声重复。终端V及主网格根表误差均为0.05%。
另设根表误差对照和固定八天跨度取点对照。来源、网络和物理数据验收见[DATA_PLAN.md](DATA_PLAN.md)。

先执行16输入长时限对照，再自动执行160输入主网格及消融/对照，共1272个方法任务，其中440个含整数结构搜索。
长阶段101每任务总4小时，单次W最多10分钟；主网格总10分钟，单次W最多10秒。
E使用剩余结构时间，不能把W的单步上限理解为所有求解的上限。
最多3个单线程worker；控制器在96小时后停止派发，正在运行的任务按原预算结束，因此进程结束可能晚于96小时。
失败、限时、原样返回初值、未启动、进程中断都保留，不自动重置预算。

## 一个算例的运行命令

```powershell
$py='D:/apps/miniconda3/envs/Topo/python.exe'
& $py -I -B rnj_wzzt_dr/run.py fit --train /private/train.npz --validation /private/validation.npz --method 101 --loss mse --preset grid --progress text --output /private/fit_mse
& $py -I -B rnj_wzzt_dr/run.py fit --train /private/train.npz --validation /private/validation.npz --method 101 --loss mae --preset long --progress text --output /private/fit_mae
```

`grid`配好600/10/60秒总预算、W步预算、输出预留；`long`为14400/600/600秒。
输入只需训练和验证P/Q/V或Y，不含真拓扑。进度提供目标、界、差距、初值来源和是否被搜索改善；
私人系数及原始观测留输出目录。`result.json`保存整个冻结候选库，后续DR继续复用原接口。

## 全批命令与恢复

```powershell
& $py -B -m rnj_wzzt_dr.experiments.wzzt_long_grid_20261117.data prepare --protocol rnj_wzzt_dr/experiments/wzzt_long_grid_20261117/PROTOCOL.md --out outputs/wzzt_long_grid_data_20261117_v1
& $py -B -m rnj_wzzt_dr.experiments.wzzt_long_grid_20261117.run prepare --data outputs/wzzt_long_grid_data_20261117_v1 --out outputs/wzzt_long_grid_20261117_v2
& $py -B -m rnj_wzzt_dr.experiments.wzzt_long_grid_20261117.run run --out outputs/wzzt_long_grid_20261117_v2
& $py -B -m rnj_wzzt_dr.experiments.wzzt_long_grid_20261117.run status --out outputs/wzzt_long_grid_20261117_v2
& $py -B -m rnj_wzzt_dr.experiments.wzzt_long_grid_20261117.run report --out outputs/wzzt_long_grid_20261117_v2
```

已准备/已运行的路径不重建。准备步骤保存代码、输入、环境和预算哈希；运行自动转入私人代码快照，后续工作区修改不影响在跑任务。
控制器开始前先实际启动子进程检查导入；可单独运行`check-workers --out <run>`，不读观测、不拟合。
再次运行先检查进程和`running.lock`；活进程不能同时启动第二控制器。
确认为死进程后，保留锁及全部partial证据再由人工处理；恢复只执行未开始任务，不给中断项重新发完整预算。
放置输出目录下`STOP`文件可停止后续派发；现有任务正常返回。这个操作保留尚未开始项为pending。

每个阶段结束自动分析；`report`也可随时生成增量报告，未完成的格子保持空缺，不算失败或0精度。
所有计划任务终结后自动生成最终报告。各预算面板分开统计，同一输入在长短预算中重复出现不算独立重复。

## 如何决定是否恢复MAE

主结果看完整内部/整树恢复、错漏线路、R/X响应误差和总耗时，另看换验证指标后的同库交叉复评。
如果MAE在恢复上不劣且用时至少减半，建议回MAE；如果MSE净增至少5个百分点且耗时代价可接受，建议保留MSE。
若结果依赖网络、噪声或求解预算，保留两配置并说明适用条件。这个判断只形成建议，不在批次中途更改生产默认。

启动记录：v1在Windows子进程导入阶段失败，3项基础设施错误收据保留，实际数据拟合为0；旧快照未改。
修正`__mp_main__`的包上下文并通过真实私人脚本spawn测试后，以v2继续同一数据和任务计划，协议引用原失败收据。
