# 公开馈线冻结批次续接

当前任务：完成本对话已经启动的reference_benchmark V1，给出标准/论文风格改造网络的实际测试结果；不启动额外科学批次，不改已冻结策略。

- 研究源码：C:/Users/23761/.codex/worktrees/36a1/Topo，分支codex/ev-train-validation-design。
- 执行副本：D:/0-github_workspace/Topo/outputs/ev_reference_stage_20261018。
- 已验证实现commit：67405807；旧mobile_v2、bayes_response、full_pipeline批次均不重跑。
- 冻结identity：abe6a12df0497d0787e86cb700b573ff53c23057216fc288dcc2ba79185b1f54，共893个源码文件，包含校准规则；所有方法、源、预测、求解预算和日期不变。
- 输入：outputs/inputs_v2/prepared.json.gz。v2只补正MATPOWER 7.1来源metadata，时序和物理数据与v1完全相同。
- 主controller：outputs/reference_v1/controller.json，启动PID29528，16个单线程worker。读状态、日志和实际PID后再决定操作，禁止重复启动。
- 当前实际进度以outputs/reference_v1/recovery.json和各阶段已保存文件为准。2026-10-18核对时开发21/32、校准0/128、评价0/96；controller.json停在10条，但原controller与16个worker仍活跃。33节点单个历史AC拟合已耗时约7500秒。
- 原后处理PID9436因状态更新时间超过300秒退出。已复现Windows读句柄阻止状态文件原子替换；这与遗留controller.json.tmp相符，原controller最终异常仍待其退出后核实。
- 恢复监测PID48356，脚本outputs/reference_tools/recovery_supervisor.py；状态为waiting_existing_processes。它等待原controller及全部worker自然退出，随后只续接同一冻结批次一次，保留全部已有结果和失败；原进程存活时不启动第二套计算、不删除lock、不终止原进程。
- 续接入口outputs/reference_tools/runtime_resume.py只为原状态写入增加PermissionError重试，并在Windows共享读时允许文件替换；893份科学源码、配置、种子及策略保持不变。运行时改动、逐PID日志及自检单独保存，不归为科学模型改进。
- controller达到statistical_complete后，恢复监测再启动finish_watch.py，依次执行冻结analysis、独立diagnostics及package_evidence；不会新建实验或按评价调参。后处理遇到心跳旧但进程仍存活时继续等待。

## 批次与数据边界

四网case33、cigre_residential、paper11、paper15，两源none/PV；开发4、校准16、评价12个不重叠日期窗，共32/128/96条件。每窗H=192、T=12，原生15分钟采样。校准阈值在评价派发前保存且不使用源标签选阈值。各方法同一表计预算、同一11kW EV和6kWh电池服务任务。

33节点保留原12.66kV、32负荷和原Ω线路；CIGRE保留大内点负荷和原变压器。paper11/15是项目自建兼容树，恢复阻抗倍率1并声明新上游变压器，不是作者原始复现。六类公开LV聚合曲线被复用，普通P/Q回归秩至多12；这不自动等于AC参数只能识别12个方向。共同源P/Q通过同一物理接口，EV单位功率因数且只加到总P一次。规划只用历史及有偏点预测；真实未来和噪声仅在封存模拟器执行后出现。

当前仅给算法共同PCC的接口角色，未给出源与case33母线2/CIGRE R1表计的共点映射。因此不固定该表计到PCC的零边。若未来要使用已知共点资产信息，应单独冻结方案，不能在本批读过评价后补加约束重算成优势。

## 检查与交付

1. 先看recovery.json、controller/postprocessor、实际进程和日志。恢复监测或任一原进程仍运行时禁止重复启动。计数以已保存条件文件为准，不用停更的controller计数作结论。旧unfrozen pilot位于outputs/pilot_v1，PID39384；它只作开发费用诊断，不进入正式分母。
2. 不改research树内py/math/protocol；它们均在893份哈希内。纯后处理工具在outputs/reference_tools，随最终证据独立保存版本。
3. 等后处理ready_for_result_review，检查summary.json、audit.json、diagnostics.json/md及outputs/reference_delivery_v1。若needs_review，保留现场、具体排查，不能重采样替换失败。
4. 分网络读严格恢复、最终F1、结构细化/零边收缩诊断、候选漏真、未决、误拒、求解界和成本。必须分清训练失败继承、H+EV本身优化失败、物理失败及指标/预测偏差。开发paper11已经出现G因一个训练拟合超预算而回退P-SQ，不能把相同计划归为G有效或无效。
5. 核对核心证据复原和完整AC回放，保留所有失败分母。完整代表raw附件过大时留执行盘并发布全hash；核心可回放包与来源许可入研究目录。
6. 只将已核对的本任务报告/证据/检查点commit，并普通push github.com/tutu-er/Topo 的codex/ev-train-validation-design。该远端/分支已获用户明确批准；保护既有PDF、pyproject和其他修改。交付时区分已完成测试与尚未完成研究。
