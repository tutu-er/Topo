# 大规模单日96点实验复现

在本独立目录根运行。Python、NumPy、SciPy以及MIQP所需的Gurobi和许可证必须可用；绘图另装requirements-plot.txt。算法和生成器不导入父项目，不依赖旧研究目录或网络下载。

## 生成并运行一个新批次

使用新的空输出目录。下列两个freeze固定各自的输入、源码与全部参数。主批240输入×10方法；压力160输入×4方法。压力条件应在查看主批真树成绩前固定。

```powershell
python -I -B tools/large96.py freeze outputs/new_main --cohort main --workers 8
python -I -B tools/large96.py freeze outputs/new_pressure --cohort pressure --workers 8
python -I -B outputs/new_main/runtime/tools/large96.py run outputs/new_main --workers 8
python -I -B outputs/new_pressure/runtime/tools/large96.py run outputs/new_pressure --workers 8
```

每个输入只有一个96点文件。前64点生成几何、参考尺度、R/W/T/E提议和全部系数；后32点只作冻结路径选择和一/双候选判别。不会生成额外日期或全96点重拟合。主replicate0..19、压力100..104。相同网络-日的观测条件配对，四网络日流独立。该协议的随机种子固定为20261112，若需要新独立试验需显式新建和审查新协议，而不是在已有输出中替换输入。

每方法预算14400秒、8个单线程worker。两个经理顺序运行，避免超出统一并发数。只用status查询已有批次；不要在经理存活时启动另一个经理。

```powershell
python -I -B tools/large96.py status outputs/new_main
```

已有终态不会重跑。若有失联attempt，先保留并审计PID、返回码、耗时与预算，不删除attempt或重置预算。生成失败也不抽新种子。失败、超时与budget_exceeded均保留计划分母。

## 全部终态并封存后才评价

等待经理返回成功，确认completion.json和selection_seal.json已经写完。不能仅凭state里的pending=0就提前truth。对每个批次依次执行：

```powershell
python -I -B outputs/new_main/runtime/tools/large96.py truth outputs/new_main
python -I -B tools/analyze_large96.py outputs/new_main --bootstrap 2000
python -I -B tools/plot_large96.py outputs/new_main
# 可选：事后真族诊断；不会修改正式选择，每输入只拟合真族一次
python -I -B tools/diagnose_large96.py outputs/new_main
```

将new_main换为new_pressure可复现压力分析。真族QP使用该输入原64/32点；它不是新算法候选。其成本另列。分析器先核对所有冻结文件并重算选择/目标、候选和界必要条件，再调用冻结物理case评分。失败与超额独立汇总，未完成或无效格不会偷偷从完整恢复分母中去掉。

## 只重建已归档结果

把records下每个records-*.zip解压到同一空目录；每卷本身是合法ZIP，各卷需要共同保留。manifest.json含每个文件SHA、所在卷及卷SHA；receipt.json给出全部CRC/SHA检查结果。解压后的两个run有独立原目录名前缀。归档不含旧多日NPZ。

便携算法ZIP排除嵌套ZIP，包含代码、本地网络、生成器及派生报告，可独立重新生成/运行。若要直接重放这400份已经封存的观测及原始输出，请另取records两卷及其清单；它们不会被重复塞进算法ZIP。

分析、绘图和诊断实际源码按SHA保存在各run中。为保护原分析，重新审计时使用新destination：

```powershell
python -I -B tools/analyze_large96.py extracted/large96_main_20261112_001 --destination analysis_replay --bootstrap 2000
```

审计器核对I/O守卫原收据的runtime/inputs/job.npz后缀并映射到当前解压根，不读取旧机器绝对路径。独立目录仍携带相同工具；需要严格复现旧字节时使用归档SHA目录中的实际源文件。不要在已封存run上重新启动拟合经理。

解释性review是第二路只读复核，使用会话内Python stdin聚合，没有另存.py脚本。其JSON保留原表SHA、公式、分母、重采样设置和逐例证据；正式指标、区间及CSV由已归档analyzer重建，见REVIEW_PROVENANCE.json。压力原版分层图及布局修正版分别存figures、figures_reviewed，各自保留实际绘图源SHA；交付图使用后者，原始数值不变。

重新发布原始记录的命令为：

```powershell
python -I -B tools/publish_large96.py --run outputs/large96_main_20261112_001 --run outputs/large96_pressure_20261112_001 --output experiments/new_records
```

目的目录必须不存在。发布前停止对源run的写入，把实际后处理工具按SHA归档到run。发布期间的console日志写在两个run之外；发布器核对归档前后清单和字节，源文件变化会拒绝完成并保存publication_failure.json。

双候选曲线只回放同一个冻结库。κ变化不重新拟合、不改变第一名，不等于实际DR后的恢复收益。若后续执行DR，需要另外记录干预次数、观测点数、响应预算及最终纠错率。
