# H-only 710009：旧与共享功率平方 upper 的单轮比较

同一重新构造的 24 pair、24 个 constant QP 可行见证、EV 配置、segments=2 与 3 秒 solver 预算下，共享模型把 upper master 从 5,914 变量/4,932 行降到 322 变量/276 行，并获得数值全局 best bound；旧 master 超时，无 incumbent 和 bound。这个单 H、同机器结果支持该构造可运行及本例计算效果，不是跨硬件耗时保证或整体策略收益结论。

| 记录 | 原 upper_master | 共享 upper_master_shared |
|---|---:|---:|
| 变量 / 行 | 5,914 / 4,932 | 322 / 276 |
| HiGHS status | 1，time limit | 0，optimal |
| master incumbent | unavailable | 52.90255752681766 |
| 最大化 global best bound | unavailable | 52.90255752681766 |
| mip_gap | unavailable | 0 |
| 建模至 solve 入口（秒） | 0.7241723 | 0.1284010 |
| solver call（秒） | 3.0363091 | 0.0141548 |
| upper call 总耗时（秒） | 3.7605040 | 0.1426124 |
| 公共 scale / 初始 qbar | 52.90265047803005 | 52.90265047803005 |

新建模时段包含可序列化 snapshot 构造；旧 snapshot 在其调用后保存，表中为实际函数调用边界，不把这些数字当一般建模速度定理。solver 的墙钟可略超过设置预算。新模型共享 graph 变量为 168=24×(3×2+1)，加共同运行模型的 153 变量与 1 个目标变量共 322；旧模型已跳过 B=0 的常数行，本次节省来自未来功率平方图共享。

读取仅限 sanitized H_710009.json/.npz。NPZ 强制恰为 history_p/q/v/master_p/master_q/root_v 六数组；JSON 只使用历史标签、候选名字与白名单声明。未用旧中心作 initializer，未读原 trial、真拓扑或未来观测、未遍历 11 动作库。原 measurement_model(3)、Observations 与 build_history_context 以 fresh seed=28101301 重建，max_centers=2、max_attempts=3、multistart=2、nfev=300、β trust=30%、source±0.05、mean forecast。各候选保留两中心：一个 optimizer_fit 和一个未优化的 feasible_low_loss_seed；其有限中心域不是置信域或全局 MLE。所有尝试及种类原样保留，失败数为 0。

每个 B 的实际精确非零行支撑均仅属一个时段，24 pair 全部通过；没有删除 near-zero 系数。双方使用同一个 pair 对象集与 constant_site_0（8 时段、3kW）QP z 见证库，完整矩阵与见证保存为 NPZ，24 个 pair fingerprint 冻结。仅另评价 none 初始化动作。旧源代码未改，只在本进程包装原 builder/solve 捕获快照与耗时，finally 恢复；前后 15 个冻结源码 SHA256 相同。

所选新动作每时段精确仅一个站点非零，EV 运行检查最大违约 3.55e−15；固定见证直接残差平方与共享分解最大差 9.09e−13。保存 master 行的独立重放最大违约 7.51e−8（scaled rows），属于数值容差证据而非 interval arithmetic。此处 EV 运行可行性不代表实测未来电网电压保证。

## QP 区间仍限制收益结论

| 动作 | 原内层 QP q_lower | 原内层 QP q_upper |
|---|---:|---:|
| constant_site_0 | 52.793863015768395 | 52.9025322694318 |
| none | 50.21451370965805 | 50.21451370975675 |
| shared 所选动作 | 52.81466416391098 | 52.902550413141086 |

新动作相对 constant 的保守代理增益区间为 [-0.08786810552081903, 0.10868739737269095]，跨 0。shared 的 global best bound 减动作 q_lower 约 0.08789336；master 在固定见证/PWL 上最优，不代表 nuisance profile 已被精确求出。selected QP 最大 raw primal−tangent gap=0.08788625，constant 为 0.10866925，均完整保留。q_upper 贴近 master best bound也不能把 upper incumbent 改称收益 lower bound。

离线 verifier 不调用优化器，从保存矩阵与 QP z 独立重算 72 条 primal/tangent 记录，最大差均为 0；核对全部 pair/snapshot SHA256、时段支撑、master x 与稀疏约束通过。证据保存在 config/protocol/source_hashes_before/after、reconstructed_pairs_and_witnesses.npz、双方 model_snapshot/result、三个 point_QP 文件、selected_point_checks、additional_diagnostic_numbers 与 offline_verification。

没有运行整体新 selector、第二 H、actual future 或现场评价；下一阶段固定预算 selector pilot 仍需主任务另行授权。本例只给冻结仿射有限 pair 上的数值 upper 证据，不给完整非线性 AC 设计最优性、拓扑正确概率或长期性能结论。
