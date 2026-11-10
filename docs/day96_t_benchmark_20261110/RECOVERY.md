# 连续一日96点实验恢复

工作区 `D:/0-github_workspace/Topo`，任务分支 `codex/iterative-matched120-20261101`。实现/协议提交 `1aa0fbc2`。旧120输入消融和六例正则剖面完整保留。

## 已冻结

- 运行目录：`outputs/day96_t_benchmark_20261110`。
- `configuration.json`和`protocol.json`：48输入、144方法配置、同4小时总预算；`source/`不可修改。
- `catalog.json`、`physical_seeds.json`、`data/`、`networks/`：训练1日96、验证3日288、测试1日96，独立于旧随机种子。
- `source/research_experiments/full_ablation/engine.py`仅新增T系数入口，普通WZZT/E保持.003f0。
- 数据协议审查521项通过；跨项目合成预测/回选、冻结和终端顺序核验通过。

## 运行顺序

1. `run.py pilot`先运行4网络首种子、中噪声的12配置。
2. 审查其结果和失败回执，再启动`run.py all`，复用已完成pilot；剩132配置。
3. 所有配置返回后，`analyze.py`作独立LP/响应/层级及候选重放核验，保留144分母。
4. `figures.py`绘图，根代理视觉核验后归档。`finish.py`可以有限等待所有返回后自动完成第3–4步，但它不启动求解，也不执行人工视觉核查。

```powershell
D:/apps/miniconda3/envs/Topo/python.exe -B research_experiments/day96_t_benchmark_20261110/run.py status outputs/day96_t_benchmark_20261110
D:/apps/miniconda3/envs/Topo/python.exe -B research_experiments/day96_t_benchmark_20261110/run.py all outputs/day96_t_benchmark_20261110 --workers 8
D:/apps/miniconda3/envs/Topo/python.exe -B research_experiments/day96_t_benchmark_20261110/finish.py outputs/day96_t_benchmark_20261110 --workers 2
```

重启前检查原管理进程和对应worker是否已退出；不得与运行中的管理进程重复调度。`pilot_state.json`、`all_state.json`、`jobs/<job>/<method>/progress.json`及worker.log是实时状态来源。结果和错误回执不会覆盖；修复失败前先保存证据。

后分析从16个网络-种子历史修正为4个跨网络共享种子组，属于冻结后统计分组核查；不改变优化器或数据。实际分析源码保存在`analysis_source/`，不要回写source。

候选导出为`jobs/<job>/<method>/candidates.json`，包含完整历史预测器和独立.16输出映射；最终主算法仍为.24收缩重拟合/严格验证，两者分别解释。接口相容性详见COMPATIBILITY.md。

完整核验之后运行 `bridge_candidates.py outputs/day96_t_benchmark_20261110`，把已存的.24重拟合输出也加入独立桥接候选库，并锚定实际最终选中预测器。原始raw路径池不一定含这个预测器；该后处理不拟合、不搜索、不修改原算法时间或原始导出。

未完成时的RESULTS.md/CSV仅是带明确分母的部分检查，完整结论必须等待verification.json确认verified=144且passed=true。

## 2026-11-10 调度中断恢复

原all管理进程在状态文件原子替换时遇到WinError5退出。当时69条结果已完成并保存，8条活动尝试没有结果。已在`recovery/manager_interruption_20261110`保留状态临时文件、8条旧payload/前缀检查点/进度和日志；`charges.json`保留69条结果哈希。宿主机只读查询确认旧worker已退出；沙箱内进程列表有隔离，不能据此判断外部进程存活。

live `run.py`和新`recovery_worker.py`只加固记录I/O，保留冻结原生优化源。重试从四小时额度中扣除已记录的中断时间估计，先写`result.pending.json`，完成恢复记账后再发布`result.json`。未开始任务的扣费为零；69条完整结果不重跑。最终逻辑配置数仍144，含中断尝试的预期次数为152。

代理中断扣费约1862.287秒；实际worker退出时刻未被精确记录，恢复审查另给出到宿主机确认退出时刻的上界。原始扣费记录不可回写。最终`timing_uncertainty.py`需按每条“成功耗时＋中断上界≤14400”作保守预算核验。有效返回算法时间与包括中断的运营时间范围分开报告，不把调度错误作为算法效率收益。

等待辅助进程只在waiting阶段单独重启以载入同类I/O重试，没有停止任何当时存活的优化器。后续状态读取应允许Windows FileShare.Delete；记录写入已有有限重试。

## 2026-11-11 完成状态

144/144原生配置全部返回，失败0。原生管理与等待辅助进程均已结束；不再启动同一批优化。794个保存的结构MIP均达到设定gap，无超时。所有144个原始预算已通过含中断上界的保守核验。

首次完整后分析通过按文件启动入口运行，导致相对导入失败，保留144个invalid核验记录于recovery/postprocess_import_failure_20261111。仅将live finish入口改为python -m，再次核验144/144通过；未修改任何原生结果或冻结源，也未重跑优化。

后分析、三组图的视觉检查、144份完整输出桥接、选模合同对照、阶段时钟、候选覆盖与接收门槛诊断均已完成。原始结果与实际后分析源分别归档；主结论见CONCLUSIONS.md。Source中253份冻结文件保持原哈希。

按最新用户要求，另一个独立目录outputs/miqp_loss_benchmark_20261111已冻结同48输入的96配置平方误差对照；小例核验通过，固定8配置pilot正在运行。新实验不替换本目录的L1结果。后续从新实验README和PLAN恢复；不得因新目标试验而覆盖或合并本批统计口径。
