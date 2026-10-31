# 全天模块式DR/EV：当前阶段入口

保留flynn16/b5、pengwah18/b5两份完整历史零收益对照。在不重写RNJ＋WZZT的前提下，扩充可控负荷的时窗、形状与幅值，检查是否存在值得购买的最终选线收益。

- [ACTUAL_TEST_REPORT.md](ACTUAL_TEST_REPORT.md)：已完成48请求/41完整拟合，正式参照zero，条件Gamma为0；收缩前后均正确。
- [ACTUAL_TEST_PLAN.md](ACTUAL_TEST_PLAN.md)：实际小试的事前信息、预算和统计口径。
- [MODEL.md](MODEL.md)：资源、整段响应、价格/费用、Gamma及优化层次。
- [HISTORY_REPORT.md](HISTORY_REPORT.md)：40个历史来源的输入审计；未找到严格近比例的全天P/Q对，不等于没有其他补偿歧义。
- [RESOURCE_REPORT.md](RESOURCE_REPORT.md)：96时段模块/幅值运行LP与29项检查，含三项反例修正。
- [ACN_DATA_REPORT.md](ACN_DATA_REPORT.md)：真实EV五会话的字段资格与信息时序，未校准SOC/容量/价格参与率。
- [AMPLITUDE_NOTE.md](AMPLITUDE_NOTE.md)：同能量不同幅值的条件性凸性结果，以及固定含噪历史反例。
- [NEXT.md](NEXT.md)：下一有界物理比较的执行顺序和验收。

已完成新全天模块的实际AC及最终拓扑小试，未发现付费结构恢复收益；当前仍无已验证的部署价格优化器。既有识别入口仍dr_pool_v2.adapter.fit_observed，全部历史/完整族与零起权重域保持。
