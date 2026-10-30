# 参考噪声消融恢复入口

先读REPORT.md、EXPERIMENT.md和METER_AND_IMPEDANCE.md。三组仅V/仅P-Q/共同误差全部完成；DR有条件地改善最终错漏，但未证明W独有价值。正式RNJ＋可选公共根是本轮事前锁定、独立实际拟合的第9方法。

实现：reference_study.py冻结并生成新误差组，直接复用旧study负荷/DR函数；reference_noise_driver.py只扩一个独立补根基线，原driver及规则不改。所有拟合从观测进入，不从真块产生候选。

私有根位于本线程scratch：

    wzzt_dr_reference_noise_generation_v1   36命名/30不同模型-噪声，72映射/33snapshot
    wzzt_dr_reference_noise_execution_v1    33/33，1222LP，204.904s，无待续锁
    dr_reference_noise_audit_stage         独立5536检查、逐块归因及汇总
    dr_reference_noise_decomposition_audit 同预测/不同contract的单例代数反例，0求解
    dr_reference_noise_stage               原始8项纯测/只读根审计/导出脚本

原始观测pickle留私有；公开evidence/reference_noise_v1保留协议、hash、逐条拟合、状态、控制声明与审计。已完成根拒绝重跑，不以剩余预算继续加种子/加网络。旧无噪/高噪记录只作既有证据，不混成新独立样本。

恢复顺序：先核hash与README，再读取已完成结果；仅确有新问题和新协议时使用新根prepare/generate/prepare/run。主比较阻抗倍率1、预算1.6、原[0,2]/gain/3%/.16不变，不能静默改变这些条件。
