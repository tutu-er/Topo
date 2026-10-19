# 小规模贝叶斯 EV 测试复算

## 已交付内容

先读 [简明报告](ev_bayes_small_results.md) 和 [数学说明](ev_bayes_small_math.md)。本轮是三末端、四个预定候选的合成测试，原长 LaTeX/PDF 未修改。

科学实现阶段提交为 fd261049；最终分析、证据和报告在后续同分支提交。执行根目录为 D:/0-github_workspace/Topo/outputs/ev_rx_stage_20261015，源码工作树为 C:/Users/23761/.codex/worktrees/36a1/Topo，分支 codex/ev-train-validation-design。

evidence/ev_bayes_small/ 保存：

- freeze.json、source_snapshot.json.gz：冻结配置、精确种子、133份科学文件哈希及原字节；base_commit 是原基线，实际新增科学代码提交见上文。
- pre_evaluation_gate.json：正式评价启动前开发通过的记录，不能由评价结果追改。
- development_assembly.json：开发2的两条原始记录逐字节复制进入统一证据集的来源与SHA；不是新实验。
- archives/{phase}_history_*.npz.zip：完整10条记录的无损数值打包，每文件保留所有数据、矩阵、失败、优化日志和条件计量。ZIP清单给出原gzip SHA、原JSON字节SHA及规范JSON内容SHA；集合路径由collection_manifest.json固定。
- independent_audit_development.json、independent_audit_evaluation.json：分别审计2条开发及8条评价；分析工具不改变策略。
- final_tests.xml、report/：68项科学/运行回归及9项无损归档测试、逐历史/方法/配对统计、诊断和预定两图。
- initial_development_failure/：开发1序列化失败、当时配置及科学源码快照；不混入正式分母。
- delivery_manifest.json、restore_audit.json：交付文件哈希和解包后的全10条独立重放核查。
- git_source_byte_audit.json：运行树133份科学源码均匹配快照；Git中90份原字节一致、43份仅有既有的CRLF/LF差异，没有其他差异或旧科学文件改写。

原始gzip记录完整保留在执行根目录的 outputs/ev_bayes_small_frozen_v1/{development,evaluation}/，未删除或修改。无损归档恢复的数据内容与它们相同；重新生成的gzip头和文本排版可以不同，因此恢复核对规范JSON内容SHA，而不是要求新gzip具有原gzip的文件SHA。

## 环境和信息顺序

本机 Python 为 D:/apps/miniconda3/envs/Topo/python.exe；冻结记录给出 Python、NumPy、SciPy 版本。MILP 主试验使用 Gurobi 13.0.3、单线程；复算概率和审计不启动优化求解。

PowerShell 中先设置：

~~~powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:PYTHONIOENCODING='utf-8'
$py='D:/apps/miniconda3/envs/Topo/python.exe'
~~~

每历史先训练80点，随后处理20点证据，规划未来6点。规划器只接收观测、固定训练参数和预测分布。真实未来输入、真实线路和32次配对未来计量噪声均在规划完成后由评价代码使用。开发/评价种子分别为2026102600–2601和2026102700–2707；两类未来共用同一历史计划。

## 直接检查已保存证据

从项目根目录执行，无需重新训练或求解MILP：

~~~powershell
& $py -m pytest research_experiments/mobile_ev_mle/test_ev_bayes_small_evidence.py -q
& $py -m research_experiments.mobile_ev_mle.ev_bayes_small_report --self-test
& $py -m research_experiments.mobile_ev_mle.ev_bayes_small_diagnostics --self-test
~~~

单个归档验证及解包：

~~~powershell
& $py -m research_experiments.mobile_ev_mle.ev_bayes_small_evidence verify research_experiments/mobile_ev_mle/evidence/ev_bayes_small/archives/evaluation_history_000.npz.zip
& $py -m research_experiments.mobile_ev_mle.ev_bayes_small_evidence unpack research_experiments/mobile_ev_mle/evidence/ev_bayes_small/archives/evaluation_history_000.npz.zip --output-dir outputs/bayes_one_fresh
~~~

解包目录必须是新目录，工具默认拒绝覆盖。完整恢复：

~~~powershell
& $py -m research_experiments.mobile_ev_mle.ev_bayes_small_evidence unpack-set research_experiments/mobile_ev_mle/evidence/ev_bayes_small --output-dir outputs/bayes_restored
Copy-Item -LiteralPath research_experiments/mobile_ev_mle/evidence/ev_bayes_small/freeze.json -Destination outputs/bayes_restored/freeze.json
Copy-Item -LiteralPath research_experiments/mobile_ev_mle/evidence/ev_bayes_small/source_snapshot.json.gz -Destination outputs/bayes_restored/source_snapshot.json.gz
~~~

集合恢复只写入development/、evaluation/两个数据目录和恢复清单，不提取科学源码。下面的<restored>应替换为上述outputs/bayes_restored。

在恢复根目录 <restored> 上运行：

~~~powershell
& $py -m research_experiments.mobile_ev_mle.ev_bayes_small_audit --input <restored> --output outputs/bayes_reaudit.json --phase development evaluation
& $py -m research_experiments.mobile_ev_mle.ev_bayes_small_report --input <restored> --output outputs/bayes_rereport
& $py -m research_experiments.mobile_ev_mle.ev_bayes_small_diagnostics --input <restored> --audit outputs/bayes_reaudit.json --output outputs/bayes_rereport
~~~

审计使用完整 AC 独立重算保存状态、证据密度、过程条件化、操作路径、未来后验及积分。它不训练新线路、不提出新动作；若当前科学源码与133文件快照不符，会报告差异。旧128份科学源码也来自冻结快照，避免新版本悄然改变回放。

Git换行规范化会影响严格字节哈希。新克隆的副本若出现该差异，应在**独立复算副本**的项目根目录恢复科学快照，再审计；不要用此步骤覆盖其他任务的工作树：

~~~powershell
@'
from pathlib import Path
import base64,gzip,json,hashlib
root=Path.cwd().resolve()
here=root/'research_experiments/mobile_ev_mle'
snap=json.load(gzip.open(here/'evidence/ev_bayes_small/source_snapshot.json.gz','rt',encoding='utf-8'))
pending=[]
for name,encoded in snap['source_base64'].items():
    path=(root/name if name.startswith('rnj_wzzt_core/') else here/name).resolve()
    path.relative_to(root)
    payload=base64.b64decode(encoded,validate=True)
    assert hashlib.sha256(payload).hexdigest()==snap['source_sha256'][name]
    pending.append((path,payload))
for path,payload in pending:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(payload)
print('restored exact source files',len(pending))
'@ | & $py -
~~~

## 测试和重新计算

最终77项测试的实际命令：

~~~powershell
& $py -m pytest research_experiments/mobile_ev_mle/test_ev_bayes_small_probability.py research_experiments/mobile_ev_mle/test_ev_bayes_small_milp.py research_experiments/mobile_ev_mle/test_ev_bayes_small_ac.py research_experiments/mobile_ev_mle/test_ev_mobile_v4_domain.py research_experiments/mobile_ev_mle/test_ev_rx_precision.py research_experiments/mobile_ev_mle/test_ev_bayes_small_evidence.py -q --basetemp=outputs/bayes_pytest_fresh --junitxml=outputs/bayes_tests.xml
~~~

basetemp须使用尚不存在的工作区目录。本环境的系统临时目录曾导致8项fixture建立失败；原失败XML另存为pytest_sandbox_temp_failure.xml，最终77项通过记录在final_tests.xml。

若需从零重新计算，应使用新的输出目录，保留本轮原结果：

~~~powershell
& $py -m research_experiments.mobile_ev_mle.ev_bayes_small_study --output outputs/bayes_new_replication --phase development --indices 0 1
& $py -m research_experiments.mobile_ev_mle.ev_bayes_small_study --output outputs/bayes_new_replication --phase evaluation --indices 0 1 2 3 4 5 6 7
~~~

这些是复算说明，本轮交付后没有自动启动新批次。MILP使用秒级时间上限，不同机器或并发负载可能产生不同可行提案；原计划、incumbent、best-bound、接受记录和所有失败都已保存，数值审计应首先回放原计划。

## 结论核对口径

八条独立评价历史是主要统计单位。每历史的regular/shift和32次噪声用于条件表现，不能当作512条独立历史。模型内Gaussian积分检查与真实AC模拟的预测校准分别报告；90%后验集合的质量和实际覆盖不同。旧完整AC联合重拟合每条件只取一个计量实现，属于另外的预定诊断，不能替换本轮概率评价。
