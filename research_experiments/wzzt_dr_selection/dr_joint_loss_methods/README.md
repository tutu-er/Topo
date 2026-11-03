# 联合残差概率方法诊断

先读REPORT.md（结果与边界）、MODEL.md（完整推导）、LITERATURE.md（原文和阅读范围）。这是独立研究目录，没有替换生产辨识器、报价器或设备控制器。

kernel.py实现联合Gaussian残差、Hermite矩界、成对指数矩/Holder、球径积分和名义射线MILP；quadratic.py实现二阶L2近似及一维特征函数反演。run.py与quadratic_run.py只读取已冻结私有实验，按协议保存逐次结果。主数据来自dr_probability_refinement的冻结报价。

私有恢复根：C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_dr_joint_loss_methods_stage。run_v1与quadratic_v1均已结束，不要重启。原code为执行快照来源；delivery/run.py在完成后修复+0/-0缓存去重，必须与原protocol区分。audit_export.py为原私有目录布局的只读算术导出器，不是可从公开文件单独重建私有轨迹的工具。

公开仅源码、协议hash和标量结果。不上传P/Q/V、协方差、候选RX、响应、u或报价轨迹。外部复现需要自行提供同格式的数据和冻结候选；源码中的本地路径仅为当前恢复入口。

测试：在本目录执行 `python -m unittest test_kernel test_quadratic -v`。正式批前18个不同数学核测试通过；交付新增1个去重回归通过，共19项。当前概率上界的数学有效性与数值积分认证不同，未称19项测试证明95%可行。
