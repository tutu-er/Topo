# 本轮冻结实验的续接

本轮尚未完成正式评价。只完成已经冻结的一批，不以新评价结果调整算法。

## 当前范围

- 执行目录：`D:/0-github_workspace/Topo/outputs/ev_rx_stage_20261015`。
- Python：`D:/apps/miniconda3/envs/Topo/python.exe`；数值库线程均为1。
- 源码目录：`C:/Users/23761/.codex/worktrees/36a1/Topo`。
- 分支：`codex/ev-train-validation-design`；用户已明确批准普通push到`https://github.com/tutu-er/Topo.git`的该分支。
- 冻结策略：`35e7a11ba65a5366930888272f516c3165355f5dbe20de69919734dfd45e23e3`。115份科学源码不改；核心版本`0706e99a`。
- 正式计划：n3、n4分别96校准、48评价，共288条历史，11方法、3推断口径、5校准规则。
- 本线程续接：`topo-ev`，每20分钟检查；完成交付后暂停。自动化配置在`C:/Users/23761/.codex/automations/topo-ev/automation.toml`。

## 正在运行的计算

先读`outputs/mobile_v2_orchestration_checkpoint.json`及新调度日志，核对PID是否活跃。校准worker原会话为36961/42087；它们分别依次运行n3、n4的前48与后48条。它们向`outputs/mobile_v2_formal_cal_worker0.log`、`worker1.log`写入。

旧controller会话41992/PID44904因Windows检查点替换错误退出，失败日志`outputs/mobile_v2_orchestrator.log`保留。恢复后的PID和日志写在研究检查点中。控制程序完成合并校准、保存两库全部阈值后，才自动启动2个评价worker，最后输出统计与元数据。

恢复controller为会话82693/PID15048，日志`outputs/mobile_v2_orchestrator_v3.log`。05:03 UTC已确认恢复并保留现有记录。检查点写入增加有界重试，仅改运行管理；14项controller测试通过。第一次隐藏启动未形成持续进度（v2空日志/PID30092已退出），因此改用`exec_command`运行前台Python并取得长会话。本环境启动后处理优先采用相同长会话方式；若使用`Start-Process`，仍须用Hidden并复查存活。

跨执行会话的沙箱进程查询可能看不到正在运行的PID：本轮15048在普通`Get-Process`中不可见，但长会话仍运行、日志持续更新，主机只读查询也确认存活。**不要因一次普通PID查询失败便重启。** 先检查会话返回与日志推进；有疑问时用有权限的只读进程查询核实，再决定是否恢复。

原临时JavaScript监控748已终止。**目前没有自动启动物理审计的监控，正式计算结束后按下节启动一次。** 不要重复启动仍活跃的controller或worker。仅在确认进程已退出时，从既有不可变记录续跑缺失部分。

## 计算结束后的顺序

1. 等待controller状态`statistical_complete`且评价worker全部退出；审计不能与动作规划同时运行。
2. 对以下三组完整raw并行做物理审计，保存PID或长会话、日志和退出码。每个进程设置数值线程1；已有同名输出先检查，禁止覆盖。

|组|目录（均在outputs下）|分母|输出|
|---|---|---:|---|
|a|formal_cal_n3_part0、formal_cal_n4_part0、guard_dev_n3_part0|106|`mobile_v2_fullreview_a.json`|
|b|formal_cal_n3_part1、formal_cal_n4_part1、guard_dev_n3_part1|106|`mobile_v2_fullreview_b.json`|
|c|formal_eval_n3_part0/part1、formal_eval_n4_part0/part1、guard_dev_n4|104|`mobile_v2_fullreview_c.json`|

上表目录均有`mobile_v2_`前缀。CLI为`python -m research_experiments.mobile_ev_mle.ev_mobile_v2_review --directory <各完整目录> --expected-count <分母> --progress --output <新输出>`。原4pilot和28开发槽已独立回算；上述316槽完成后，另对`mobile_v2_guard60_cost_n3`、`mobile_v2_guard60_cost_n4`共4槽输出`mobile_v2_fullreview_cost.json`。成本槽复用了开发seed，须单独审计以免被判重复。共352实验槽；开发同seed不同条件不是独立历史。

3. 同时可运行以下独立后处理：

```powershell
python -m research_experiments.mobile_ev_mle.ev_mobile_v2_cache_owner_audit --directory outputs/mobile_v2_formal_cal_n3 outputs/mobile_v2_formal_cal_n4 outputs/mobile_v2_formal_eval_n3 outputs/mobile_v2_formal_eval_n4 --expected-count 288 --progress --output outputs/mobile_v2_formal_source_cache_owners_v1.json.gz
python -m research_experiments.mobile_ev_mle.ev_mobile_v2_spatial_audit --n3-directory outputs/mobile_v2_formal_eval_n3 --n4-directory outputs/mobile_v2_formal_eval_n4 --output outputs/mobile_v2_spatial_v1
```

空间消融事前选每库index0/12/24/36，8历史×4移动方法。把同一条实际净功率放回site0，仅用于拆分电气空间作用；由于少耗驾驶电量，该反事实不作为已校准的可执行固定点策略。

缓存审计只核对保存路径的费用所有权。`cap_fixed`可能免费复用仅由`scalar_mobile`生产的缓存；正式记录是否发生，由实际审计决定。缺失profile/counters按下界或未知处理，补计费用不能当作重新独立运行耗时。

4. 两库分别运行不确定性分析（n3示例，n4改对应库）：

```powershell
python -m research_experiments.mobile_ev_mle.ev_mobile_v2_uncertainty --evaluation-report outputs/mobile_v2_formal_n3_analysis/evaluation_reliability_v2.json --records outputs/mobile_v2_formal_eval_n3 --bank n3 --output outputs/mobile_v2_formal_n3_analysis/uncertainty
```

5. 等全部规划结束后，汇总运行`test_ev_mobile_v2_*.py`加`test_ev_mobile_cases_v2.py`、`test_ev_reliability_v2.py`，输出全新XML和basetemp。现有单项测试均通过，最终汇总尚待运行；不要把旧测试数当最终汇总。若修改后处理工具，保留失败输出、添加有意义的回归检查，保存新工具哈希；不改115科学源码。

6. 用`ev_mobile_v2_pack --plan`检查声明，随后`python -m research_experiments.mobile_ev_mle.ev_mobile_v2_pack --raw-root outputs --output outputs/mobile_v2_evidence_v1`。目标及其`_work`须尚不存在。14逻辑目录、352槽全部统计投影，预选完整代表及n3/n4评价0的完整raw，单包至多20MiB。中间work及其余GB级raw保留D，交付包复制C研究目录的`evidence/ev_mobile_v2/delivery`。

7. 将最终分析、审计、空间消融、测试及新工具字节快照按原字节压缩、记录SHA，加入交付证据。原stage0证据及旧记录保持不变。
8. 写简明`ev_mobile_v2_results.md`，更新研究检查点、复算入口和README状态标记；不改冗长原LaTeX/PDF。核对全部计数、失败分母、运行范围、费用口径及界对象。
9. scoped commit并普通push。核对diff、暂存区、remote/upstream；不要`git add .`。C中用户已有PDF与pyproject修改及其他无关文件必须保留。
10. 完整回答用户的三个问题，再将`topo-ev`续接设为PAUSED。没有完成所有验收时继续保存检查点，不写研究已完成。

## 报告时必须分清

- 旧69/72：曲线仅25/72相同，第一名及保留集72/72相同；三共同失败为短训练star被多2个参数的clade12替代。旧108/72已用于新规则开发，不再算独立验证。
- 主预声明规则比较：同动作`label_AIC`相对`H_R`。其余3规则解释性报告；误拒减少与未决增加分别列。
- 移动：报告真实移动次数、实际AC裕量、校准决策及驾驶/接线成本；区分继承固定点、直接梯度步骤、只因阈值改变的决策差异。空间反事实仅8个事前选定历史。
- 训练每拓扑仍一组参考参数；B竞争者与最终联合推断允许一组线路同时解释既有与新增数据。结果是局部取得拟合，不称全局MLE。
- B保护可能挡住有效移动，且局部线性保护会锁住远处可行解；已有开发反例在`ev_mobile_v2_guard_reachability.md`。新策略是否改善，由正式独立结果判断。
- 场景为公共LV线路资产参数下的模拟，3/4末端有限候选库，普通AR负荷；没有完整33节点扩网、现场数据或随机功率追踪误差验证。
- Wilson区间每库48条：1例为2.08个百分点，0/48区间仍0–7.41%。不把模拟事件率、似然或设计距离称为单个拓扑的正确概率。

## 无关修改保护

C中的`docs/mobile_ev_probing_report_v2.pdf`原SHA256为`fe89b241285fd81f570a921fe99a49bfdb1d6bca6e0ced0d498a45ef81d5dbfa`；`pyproject.toml`为`67d21b6a3da4629c5cba5c445404fc5ecde154bf6b472ee4fa06f75ceb760e1c`。D主目录其他研究修改不碰。用户本轮已明确批准向指定remote/branch普通push，不再重复询问。
