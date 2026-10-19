# AC96后台中断与恢复记录

2026-10-19用户询问运行状态时，核对原批次尚无`execution_complete.json`或最终汇总。已有41/64项最终记录：36 complete、4 incomplete、1 failed；另有4项已启动但无final、19项尚未启动。32项复用NJ/RNJ基线均完整保留。

4项incomplete为Pengwah18常值源的严格块/不分组/纯WZZT，以及平滑源纯WZZT，原因均为`extension_not_proven_optimal`；它们没有被重新求解。唯一failed仍是先前Soumalas11无源完整RNJ初值的记录器PermissionError，其隔离I/O恢复结果另行保存。

## 中断证据与原尝试保留

旧终端会话已经不可访问。提权只读CIM进程清单确认原控制器、worker及两收尾进程均不在运行；未操作其他实验进程。旧controller锁残留，汇总最后更新07:02:26，最后检查点07:03:12；进程具体停止时间及原因尚未确认。

中断项为paper15间歇源的严格块、不分组、纯WZZT，以及paper15平滑源严格块。四项均只有初始化认证点，没有最终结果。原核心没有路径续接接口，因此采用完整归档后从声明初始支持重新执行。

原始四目录的全部日志、模型、认证检查点、输入绑定及progress，旧锁、两收尾状态目录、CIM证明和文件SHA映射，均保存在：

`outputs/rnj_blocks96_interruption_archive_20261019/`

移动前解析并核对源、目标绝对路径严格位于工作区，目标为全新目录；逐项移动前后核对全部文件SHA。旧检查点没有被覆盖，也没有当作完整方法评分。

原中断实际耗时未知，归档记录保留最近检查点elapsed作为下界，`actual_spent_seconds`及精确总耗时留空。后续主表中这四项的耗时来自新执行；伴随`recovery_attempt_accounting.json`说明额外中断尝试及时间缺失。

## 恢复过程与当前状态

新增`research_experiments/rnj/resume_rnj_blocks96.py`默认只读plan，执行模式要求新鲜且与原批次绑定的完整进程证明。它在独立新Python进程中首次加载原source_snapshot，原生prepare前后被动校验205个既有最终文件、401个固定文件及原协议SHA。求解仍调用原冻结worker，原生算法、数据、Source条件、筛选、RX75、四worker及6000秒/solve、30000秒/method预算均保持原样。

本次检查205/205既有结果文件未变，协议仍为`167eaa23d1fb9257aaefa45ae90c6805ef28c4376fae4695993e410875acea6c`。恢复监督器通过隐藏独立进程启动，自己与子进程直接写文件日志。2026-10-19 09:42已确认监督器16164、原生控制器23524及四个worker均存活，启动终端已经退出。

监督状态及直接文件日志位于：

`tmp/resume_rnj_blocks96_state_20261019/`

原生队列完成64项并关闭锁后，监督器顺序执行既有主审计/汇总与阶段诊断收尾；任一步失败即保留错误并停止后续步骤。最终输出仍使用原`reports_final/`及`audits/supplemental/`位置，原失败、超时、I/O恢复与外部中断分别记录。

恢复器6项文件级测试通过，真实只读plan及独立实现审查通过；恢复后prepare的SHA门禁已实际通过。本文记录恢复，不宣称剩余23项已经完成。
