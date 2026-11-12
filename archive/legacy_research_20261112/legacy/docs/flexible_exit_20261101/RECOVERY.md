# 恢复与文件索引

工作区D:/0-github_workspace/Topo，分支codex/flexible-exit-20261101。dd92d55f保存首轮试验代码/协议/机制检查。small3_original_exit仍为研究主线；未修改生产默认或原四方法/联合方法记录。

- 先读CONCLUSIONS.md、PLAN.md。
- outputs/flexible_exit_20261101/completion.json：首轮96/96，24×4分支，不是96独立输入。
- docs/flexible_exit_20261101/summary.csv：all24及可实施13输入，raw/proposal/guarded三种结果。
- diagnostics.csv：删块、重启前后L/J、重新出现、原先被挡的真连接数及保护来源。
- block_truth.csv：真/假目标块的事后诊断，未用于选块。
- cost_per_input.csv、cost_summary.csv、batch_budget.json：逐分支账及共享成本，component之和非并行等待时间。
- paired.csv：相对旧主线与等搜索时间continue对照；cases.csv为四网络分解。
- earlier_stage_diagnostic.csv只检查更早前缀的目标块，不代表已执行另一干预。实际追加时机试验另在outputs/docs flexible_exit_earlier_20261101。
- verification.json、post_checks.json、implementation_check.json为独立证据。

本地outputs/.../source保存真正执行的冻结优化器，parents保存仅训练起点与分开存放的验证保护树；jobs保留payload/progress/checkpoint/result和worker.log。训练worker不接收test或truth路径，验证保护的旧选择记录在训练结束后读取。

records.zip包含以上文本证据和源，观测NPZ、网络CSV不重复打包。catalog/protocol保存位置与哈希；原数据继续原位保留。

冻结snapshot中的analyze.py是执行前版本；实际后处理修复了JSON列表转元组问题，未改变数值优化器或重跑分支。actual_analysis/analyze.py保存产生该结果的分析脚本，其SHA与verification一致；actual_driver包含交付时分析/归档驱动。不要把冻结的未被worker调用的分析入口当作实际执行审计版本。

Python为D:/apps/miniconda3/envs/Topo/python.exe，PYTHONPATH为D:/0-github_workspace/Topo/rnj_wzzt_core。可查看已完成结果，无需重跑求解器。

```powershell
$env:PYTHONPATH='D:\0-github_workspace\Topo\rnj_wzzt_core'
D:\apps\miniconda3\envs\Topo\python.exe -B -m research_experiments.flexible_exit.analyze analyze outputs/flexible_exit_20261101
D:\apps\miniconda3\envs\Topo\python.exe -B -m research_experiments.flexible_exit.diagnose outputs/flexible_exit_20261101
D:\apps\miniconda3\envs\Topo\python.exe -B -m research_experiments.flexible_exit.analyze package outputs/flexible_exit_20261101
```

重新分析会重算独立LP并更新分析脚本哈希；如已冻结追加试验引用旧verification，请先保留被引用文件，避免破坏追加协议。改参数或输入须使用新目录，不覆盖本次。
