# 固定计划AC修正：续接

## 当前状态：A运行、B已冻结

2026-10-17追加。下文A启动历史保留；本节及checkpoint优先。B已在读取A评价内容之前冻结：outputs/ev_ac_history_refresh_B_v1，policy 1b9d8f5d22ae6d3ca0c07f57a26466bc1bf9b83a0bde7cb80ee8688285e7d316；A150+B15份源码及快照保存在该根。B统一max_iter50，GHK仍max_power18，其余原精度与物理域不变。55项新测试通过。新物理R/X切平面仅完成8开发固定点诊断及数学稿，不在B算法中。

B独立校准PID42844，日志controller.cal.resume1.stdout/stderr.log；2worker重评A的同32份校准历史和固定计划。首次PID38336在预检发现A原calibration_002/003的旧launch状态未更新而停止，未写B科学记录。完整权限确认旧PID均退出、raw哈希与原恢复收据一致后，仅收养两个已完成运行登记；原登记全文、raw哈希和操作收据见infrastructure_A_claim_adoption_v1.json。源码、计量和原A结果不改，不能重复生成或拟合A。

先看A/B controller及完整权限进程，不重复启动。A仍为PID27340自动完成32评价；A evaluation_complete后执行原后处理。B calibration_complete后核对32记录及失败，B calibration.json按自身预测NLL选择a，不能沿用A倍率；再执行`python -m research_experiments.mobile_ev_mle.ac_history_refresh.control run --root outputs/ev_ac_history_refresh_B_v1 --phase evaluation`，它先绑定B校准哈希，再重评同32份A保存评价记录。固定计划和原future noise不变，不生成新科学批次。B源码15份此后不可修改。

两批完成后运行ac_history_refresh.report配对报告，并独立核查B源码、输入与校准绑定、MAP下降/驻点、GHK与完整AC采样门槛、原计划和计量相同、失败retain-all分母及推断费用。A conditional fixed/joint引用A原记录，B没有新增该类refit。B尚未接入MILP；局部Gaussian仍漏掉曲线补偿，概率是否改善以独立评价为准。开发0的完整AC与物理R/X近似halfSSE几乎相同，但η切平面严重失真；physical_rx_next_math.md明确为后续未实现方案。

2026-10-17。源码分支 codex/ev-train-validation-design；实现6ed82322和审计6a75c211已推送。执行副本 D:/0-github_workspace/Topo/outputs/ev_rx_stage_20261015。原贝叶斯冻结批次不再运行，原PDF与pyproject不改。

## 开发完成及新冻结批次

8份开发完成，outputs/ev_ac_repair_development_v3/controller.json 为 development_complete，PID27724及worker已退出。7/8后验完整，1份积分失败；全部条件AC拟合可评价。开发结论见 ev_ac_repair_development_results.md。完整未来传播未改善本批SQ/BC分类；共享线路重拟合对激励计划有改善，恒充有退步例。保留全部失败与弱SSE间隔。

新冻结根 outputs/ev_ac_repair_frozen_v1；freeze.json和source_snapshot.json.gz已保存。policy及实际PID见 ev_ac_repair_checkpoint.json 和输出controller.json。主批32校准、32评价，2worker；每历史两未来情景各一个噪声实现，另有clean诊断。科学设置与开发v3相同；汇总改用逐份压缩读取，8份真实开发记录的量测统计完全等价，78项相关测试通过。

## 启动与续接

启动前先查完整权限CIM进程命令行、controller.json、controller_lock.json、launch_claims与orphan_workers.json。不能因沙箱Get-Process无结果重复启动。使用带日志的完整权限 Start-Process -WindowStyle Hidden；普通沙箱启动进程可能立即消失。

主批首次PID13560因Windows临时占用controller.json导致replace失败；原worker独立完成，4份raw完整保留。已核对无活跃worker和150份科学源码不变，使用evidence/ev_ac_repair_v1/scripts/controller_safe_io.py续接，PID27340；该启动包装器仅对相同atomic_json写入的PermissionError重试最多10秒，3项测试通过。详情、四份原raw哈希及包装器哈希在infrastructure_recovery_01.json，后续日志为controller.resume1.stdout/stderr.log。遇到文件占用时不要改科学源码或重跑已有样本。

命令为 Topo环境Python -u -m research_experiments.mobile_ev_mle.ev_ac_repair_control --output outputs/ev_ac_repair_frozen_v1 --phase calibration --workers 2 --draws 256 --replicates 4 --noise-repeats 1 --then-evaluation。启动器保存launcher.json、stdout/stderr。冻结Python和数学稿不再编辑；新增独立审计脚本放evidence/ev_ac_repair_v1/scripts/。

控制器先完成32校准记录，再选全局G11协方差倍率及条件AC的SSE阈值；保存全部校准输入哈希。评价必须绑定校准文件哈希且32份规格完整，之后自动开始32新评价。评价不调参、不补跑失败。任一worker超1200秒保留process_failure；控制器中断不杀活跃worker，续接先检查。

## 后处理与交付

1. evaluation_complete 后先独立核对源码/种子/校准绑定、全部失败分母、原始计量与计划。
2. AC物理回放、四组均值协方差接线、概率评分、拟合状态/SSE与费用审计。审计调用费用单列，原记录不覆盖。
3. 对同一固定计划比较G00/G10/G01/G11/G11_cal与条件AC fixed/joint。条件AC直接输入P/Q，是不同概率口径；fixed→joint才隔离共享R/X重拟合的作用。
4. 同域SQMD与BCMD用共同历史配对比较，报告缺失范围、候选误拒/未决、MAP错误、区间覆盖及费用。独立历史是统计单位，零观测错误保留非零风险上界。
5. 输出简明结果、数学和复现证据，commit并正常push授权分支。没有完成非线性历史后验、完整贝叶斯密度或新MILP梯度时明确其状态，不把未来传播替代称为整套后验已修正。

## 旧pilot

development_v1/history_000与v2/history_001继承旧max_power16，均后验积分失败，保留其条件AC反例；新版v3与main为max_power18，原质量门槛不变。所有旧数据及审计证据只读保留。
