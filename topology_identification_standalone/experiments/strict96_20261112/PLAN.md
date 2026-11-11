# 严格96点：相对J门槛与双候选开发回放

每个输入全部可用观测只有原train_observed.npz中的96个时点；每时点包括所有观测终端的P/Q/V与同一根表。保持原时间顺序，前64点训练（前16小时），后32点用于选择（后8小时），不随机打散，不额外读取旧validation/test观测文件。

RNJ几何、RNJ候选、参考MSE、热启动、W/E结构搜索、所有权重和数值清零均只使用前64点。后32点只评价已冻结模型的预测，既不回流搜索，也不参与参数重拟合。选模后不再用全96重新拟合。没有独立预测测试集；只报告train64/selection32 MSE，不能写成独立测试MSE。

只使用现有catalog中预先固定的pilot4：四网络各seed_index=0、voltage_sigma=.0005。每个输入只复制唯一train96观测，工作进程只接opaque ID；不复制旧cases、diagnostics、旧候选、结果或完整data目录。Python I/O守卫阻止误读，不声称它是OS安全沙箱。原48输入的truth已被研究过，本次标development_replay，非新增盲测。

主结构固定101（R＋原始W＋E），不引入T或重启分支。alpha=.003、响应界[0,2]不变。比较kappa=0、.001、.003，其他设置相同：每输入每kappa总预算14400s，W单次3600s、QP300s软检查、输出预留600s，单求解器/BLAS线程，四输入并行，每输入依次运行三种kappa。名义上限12×4=48方法小时，不是实测成本。

正常结构提议满足：

\[
J_0-J_1>\max(10^{-10},\kappa J_0),\qquad
J=M_{\rm train64}/M_{\rm ref64}+.003k.
\]

默认kappa=0保持原接收规则。纯数值零支持的删除若没有新增、删除前权重均≤1e−9、前后活动内部族相同、归一化训练拟合不劣于1e−10，则仅用数值门槛，防止相对门槛重新锁住零块。非零块删除仍服从同一相对J门槛。kappa不是置信水平，也不表示训练MSE固定相对下降率。

每个拟合保留实际selected作为第一候选；从训练生成库取活动内部族不同、selection32 MSE最好的第二候选。排除零权重辅助块造成的伪差异；返回完整原始模型索引、差异连接、损失差。单个32点留出块不足以评估跨日稳定性，因此statistical_ambiguity=not_calibrated、global_identification=not_established，不自动宣称模糊概率或唯一辨识。

所有计划项终态后，每个输入只按其自身32点的MSE选择kappa，数值tie优先较小kappa，并写入selection seal。其他输入的观测不影响本输入的选择；跨输入均值只作描述，不选择共同kappa。此后单独grade才可读取物理真树评分。truth不作为观测，不进入模型或参数选择；grade也不得读额外validation/test NPZ。所有失败、未完成和预算超额保留计划分母；不自动重置attempt重跑。

报告所有kappa的第一名F1、完整恢复率、top2包含真族比例、selection32 MSE与耗时。top2 coverage是候选覆盖指标，不能替代实际第一名成绩；不根据F1改选kappa。先完成信息边界、严格门槛、双候选去重单元检查，再运行12个方法配置。

QP保留作为统一固定族权重拟合和数值核查。已发现历史4个T记录的reported lower bound与同域QP点冲突；新代码在清零改变计数之前记录bound一致性，保留原reported U/L，冲突时撤销最优证据解释而保留已验证可行模型。此次先不把全局NumericFocus/Presolve改动混入kappa对照。
