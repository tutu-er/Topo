# 固定 P15 菜单的全部位置检查

这是一项事前锁定的位置诊断，检验现有 D-opt 是否遗漏可改善最终恢复的位置。它不实现新的价格算法，不用事后最好节点替代事前规划。

## 固定问题

- 沿用 paper15/reference、原 45 户映射、2021-02-09 行动和 02-10 验证。
- 规划只读取原 plan.json；P15 在 15/15 末端均预测可行。预算 1000 NOK/日，其他合同、功率和日电量限制不变。
- 15 个单节点位置加无奖励，N24/N96、原 reference/high 两档电压噪声和两个既有整日响应模式：128 格，124 个不同拟合输入。
- 无奖励与节点 101/115 的 24 格、20 个唯一拟合在精确观测、源码和配置身份匹配时只读复用；最多新增104个拟合。每次返回完整RNJ、单初值和多初值三个模型；模型数不增加独立样本。
- 新增位置102–114；两个行动模式、一个验证日、两档噪声，共78个日记录、1872模拟小时。历史直接复用。达到拟合批次上限前可能已经准备下一任务的物理记录，独立审计也计入这些记录。
- 算法、25秒每起点的停止规则、所有连续权重范围、3%验证选模和0.16收缩不变。实际约束失败、求解失败或未决均保留原分母，不重抽或增大预算。

先对每个固定位置平均两个经验模式，再比较无奖励、原D-opt位置115、均匀随机位置和事后最好固定位置。不同噪声与采样条件分别报告；不同模式来自同一事件日的不同用户组，不称独立日期或总体概率。当前验证日用于模型选择，不称未接触的预测测试。

## 文件与复现

`location_probe.py` 是隔离控制器，复用已验证的观察接口、物理生成器和有限候选池辨识器。`audit_location_probe.py` 只做独立算术核验，不调用优化或AC。原始家庭负荷、派生P/Q/V和完整缓存留仓库外。

在工作树根目录运行：

```powershell
& 'D:/apps/miniconda3/envs/Topo/python.exe' -B -m research_experiments.wzzt_dr_selection.dr_location_probe.location_probe `
  --source 'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_data_calibration/flow_runs/paper15_iflex_v1' `
  --old 'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_dr_pool_v2' `
  --output 'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_dr_location_probe_v1' `
  --max-new-fits 8
```

同命令按冻结协议续跑，已记录任务不重复计算；目标目录有运行锁时不得启动第二个进程。失败是终止记录，不能通过续跑隐式覆盖。

私有输出包括：protocol.json、records/逐格结果、physical/及哈希回执、fit_cache/新拟合、requests/缓存索引和summary.json。审计必须核验全部生成物理文件的回执哈希，以及协议外补记的matrix_runner.py和end_to_end/runner.py依赖哈希；首批以这些检查补足控制器原始源码清单。冻结协议不事后改写。

```powershell
& 'D:/apps/miniconda3/envs/Topo/python.exe' -B research_experiments/wzzt_dr_selection/dr_location_probe/audit_location_probe.py `
  --run-root 'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_dr_location_probe_v1' `
  --repo-root 'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo' `
  --receipt 'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_dr_location_probe_v1/audit_final.json'
```

## 判定范围

如果所有位置都不改善完整有效族、内部F1或根部块恢复，停止这张固定P15位置表的优化扩展；该结论不覆盖其他价格档位、多节点组合、其他日期或未测试算法。

若存在有效位置，先定位变化发生于建池、搜索、共同调权还是验证选模。事后好位置只证明动作空间有局部价值；利用历史信息找到它是另一个尚需独立验证的问题。后续价格规则不能读取本批真实拓扑或未来响应；本批不能再冒充该新规则的未见测试集。


## 已完成结果与聚合导出

128/128格、124个唯一拟合已完成并通过独立算术核验，详见REPORT.md；不再续跑此矩阵。
完整原始结果保留在上述私有运行根，聚合证据位于上一级evidence/dr_location_probe_v1。
`export_evidence.py --run <私有运行根> --output <聚合目录>`仅在summary complete且audit_final all_passed时导出，不包含家庭/P/Q/V轨迹。

`prior_selection_diagnosis.py`和`new_location_mechanisms_102_103.py`保存原诊断源码及其哈希。复现时先复制到私有运行根，再从那里执行；前者读取本工作树已存dr_pool_v2证据，后者读取该运行根records及结果引用。两者只做结果读取与算术/结构比较，不调用优化或AC。它们使用真值作事后机制诊断，不能移作价格规划输入。
