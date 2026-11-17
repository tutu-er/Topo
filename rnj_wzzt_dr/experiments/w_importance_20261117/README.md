# 验证W是否值得保留

本批比较删去W之后的实际恢复与用时，包含负面和无效结果。详细口径见[事前协议](PROTOCOL.md)。

| 模式 | 实际执行 | 回答的问题 |
|---|---|---|
| full | 池内QP逐步组合＋未知连接MIQP＋E | 当前完整流程 |
| pool_only | 池内QP逐步组合＋E | 原生未知连接搜索是否有额外价值 |
| skip | 直接做E | 整段W是否有额外价值 |

每个模式都保留同一训练目标、RNJ原始池、E参考初值规则、.08输出收缩及验证MSE。
long16输入、grid160输入，各跑三个模式，共528项全新MSE任务。16个long输入属于160网格，不增加独立数据数。
原先已有结果只作版本参考，本批fresh full用于配对。三模式顺序循环轮换，保存完整源/输入/预算哈希。

long为14400秒总额、W单步600秒、E上限6900秒；grid为600／10／270秒。E不会获得删去W后省下的额外时间。
E的初值和候选池会随W路径变化；报告同时解释结构来源、候选覆盖与选择变化。

```powershell
$py='D:/apps/miniconda3/envs/Topo/python.exe'
& $py -B -m rnj_wzzt_dr.experiments.w_importance_20261117.run prepare --parent outputs/wzzt_long_grid_20261117_v2 --out outputs/w_importance_20261117_v1
& $py -B -m rnj_wzzt_dr.experiments.w_importance_20261117.run run --out outputs/w_importance_20261117_v1
& $py -B -m rnj_wzzt_dr.experiments.w_importance_20261117.run status --out outputs/w_importance_20261117_v1
& $py -B -m rnj_wzzt_dr.experiments.w_importance_20261117.run report --out outputs/w_importance_20261117_v1
```

run从封存的私人代码副本启动。原批次控制器退出、全部任务终结后，本批才开始派发；等待期间不占用求解进程或消耗新批预算。
`controller.json`显示waiting_parent或running及活跃worker PID。最多3个单线程worker，每任务独立进程。
已有输出目录不重建；已完成或失败任务均保留。中断不自动重给预算。
`STOP`文件停止后续派发，当前任务按原预算正常结束；控制器异常时若尚有子进程，保留锁及其PID，避免重复启动。

每阶段结束自动做独立复算，全部任务终结且报告通过后才写execution.status=complete。
错误／遗漏、完整恢复新增／丢失、候选覆盖、近根召回、时间、失败、原生和partial限时均保留。
原生训练目标改进、所选路径来源及输出收缩贡献分开说明。轨迹中的W前缀复评不冒充另跑的完整预算消融。

单个输入也可用通用入口：

```powershell
& $py -I -B rnj_wzzt_dr/run.py fit --train train.npz --validation validation.npz --method 101 --loss mse --w-mode pool_only --preset grid --e-seconds 270 --output outputs/pool_only_fit
```

`--w-mode skip`直接进入E；默认仍为full，省略`--e-seconds`保持原E预算规则。新开关不改变在跑旧批的私人快照。
