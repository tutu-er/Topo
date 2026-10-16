# v3 精简证据包：使用与保留范围

模块：`research_experiments/mobile_ev_mle/ev_mobile_v3_evidence.py`。

精简文件完整保留训练计量和训练拟合、验证计量、预测、历史预拟合，以及各实际动作的全部最终 `evaluation.comparisons`。方法保留实际功率、操作、失败、规划成本、来源、选择的操作向量、分数和全部有向任务、求解界、PWL范围及support信息。相同action_key的最终evaluation只存一次。

只删除planning中的重复`references`、标量`table.background`和`table.tables`。每处删除列出原路径、内容哈希、未压缩字节数和原因；被删除子树中显式失败的节点先完整另存`failure_evidence`。原始raw继续保留在D-stage，精简文件记录其路径与原gz的SHA256。

`restore`返回可供既有summary和物理审计器读取的记录。它恢复每个方法的最终evaluation引用；已经删除的规划缓存保留删除声明，因此返回值不称为完整raw。

开发2的4例已实测：32个方法的`summary.project_method`整项相同；原记录与恢复记录的独立物理审计状态、计数、误差、状态清单和失败清单均相同。原gz总计26,317,800字节，精简gz估计9,525,542字节（36.2%）；本检查未生成最终包。证据为`mobile_v3_evidence_dev2_roundtrip_v1.json`，5项单元测试记录为`mobile_v3_evidence_tests.xml`。

在最终16例及所需summary/audit/log落盘后，由根代理执行：

```powershell
& 'D:/apps/miniconda3/envs/Topo/python.exe' -m research_experiments.mobile_ev_mle.ev_mobile_v3_evidence `
  --directory outputs/mobile_v3_pilot2 `
  --output outputs/mobile_v3_compact_evidence_v1 `
  --phase development --phase evaluation `
  --attachment outputs/mobile_v3_evidence_dev2_roundtrip_v1.json `
  --attachment outputs/mobile_v3_evidence_usage_v1.md `
  --attachment research_experiments/mobile_ev_mle/ev_mobile_v3_evidence.py `
  --attachment research_experiments/mobile_ev_mle/ev_mobile_v3_physics_audit.py `
  --attachment research_experiments/mobile_ev_mle/ev_mobile_v3_summary.py
```

工具先检查所选phase的全部预声明文件齐全，核对spec/policy和source_snapshot的全部原字节哈希。之后保留根目录的freeze、source_snapshot、environment、summary、audit、日志与exitcode；附件另外存档。包清单包含每条原/精简文件哈希、排除字段、保存失败数与体积。已有目标目录会拒绝；失败留下`pack_status.json`，不补跑或删除原试验。

内存复核接口（在保存的项目源码环境运行）：

```python
from research_experiments.mobile_ev_mle import ev_mobile_v3_evidence as ev
from research_experiments.mobile_ev_mle import ev_mobile_v3_summary as sm
from research_experiments.mobile_ev_mle import ev_mobile_v3_physics_audit as ph

compact = ev._read("outputs/mobile_v3_compact_evidence_v1/evaluation/000.json.gz")
record = ev.restore(compact)
method = sm.project_method(record, "projection_mobile")
physical = ph.audit_record(record)
```

物理回放复用已有独立branch-current实现，不重新估计线路或源、不重新选择动作。规划代理的全局最优性、每次价格的完整重拟合差分，以及被删除缓存的逐点重放均不在这份精简恢复检查中。
