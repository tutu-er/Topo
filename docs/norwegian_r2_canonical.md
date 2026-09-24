# Norwegian radial 2 可辨识规范算例

## 数据合理性与边界

数据来自 Norwegian industrial distribution grid dataset，DOI
`10.5281/zenodo.10361330`。原始适配器保留公开径向拓扑、线路公共标幺 `R/X`
以及五个末端表计的真实小时有功电量记录。小时 `Load_kWh` 被解释为该小时的
平均 kW，再除以 1 MVA 基准得到标幺有功。

该数据适合作为“真实负荷激励 + 公开物理网络”的半合成拓扑测试，但不是现场
完整 PQV 测试：

- `P` 使用公开的小时表计记录；
- `Q` 按节点生成时变 `Q/P=0.20--0.36`，对应功率因数约 `0.981--0.941`；
- 根电压为均值 `1.02 p.u.`、标准差 `0.0008 p.u.` 的合成已知参考；
- 末端 `V` 和线路潮流由平衡单相径向 AC backward-forward sweep 生成；
- `P/Q` 噪声默认是瞬时真值的 `0.5%`，`V` 噪声默认是瞬时幅值的 `0.02%`。

原始网络中的理想变压器和连接支路使用零阻抗，且包含多个 degree-2 hidden
nodes。这些节点表达设备层级和电压等级连接，但 terminal-terminal additive
distance 只能辨识最小加性树，不能唯一确定零长度隐藏边或 degree-2 串联点。
因此不能直接以 17 节点设备图作为 latent-tree 真值。

## 规范化规则

`canonicalize_norwegian_terminal_radial` 执行：

1. 收缩隐藏节点之间的零阻抗边；
2. 将 degree-2 hidden chain 替换为串联阻抗之和；
3. 保留每个 terminal meter 的独立节点身份；
4. 对 terminal 的零阻抗连接加入 `0.0005+j0.0004 p.u.` 的短服务线。

在 11 kV、1 MVA 的公共等值基准上，该服务线为
`0.0605+j0.0484 Ohm`。它是为保持表计节点可区分而设置的合成等值服务阻抗，
不应解释为原数据集给出的 0.415 kV 电缆参数。

radial 2 的变化为：

| item | raw retained graph | canonical graph |
|---|---:|---:|
| buses | 17 | 8 |
| terminal meters | 5 | 5 |
| hidden nodes | 11 | 2 |
| degree-2 hidden nodes | 8 | 0 |
| zero-impedance edges | 12 | 0 |
| nontrivial rooted clades | 2 | 1 |

规范树的两个隐藏节点度均为 4。五个真实表计文件和时间戳映射不变。三组 168
小时场景中的 AC 真电压范围为 `0.99627--1.01993 p.u.`，未出现潮流不收敛。

## 当前辨识入口

旧候选池生成的历史性能表已经移除，避免与当前精简后的统一流程混用。当前版本
使用 Ordered RNJ、GTLS、NJ/RG 软聚合、固定树 EIV/AC 排序和 quartet 验证。

```powershell
python -m experiments.run_additional_case_validation `
  --cases norwegian_r2 --scenario-count 3 --norwegian-hours 168
```

应同时报告 final clade F1、exact recovery、candidate-oracle F1 和规范化前后的
节点统计。若 candidate-oracle F1 小于 1，失败发生在候选覆盖阶段，不能归因于
AC 排序器。
