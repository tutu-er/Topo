# 本阶段证据

先读 ../../ev_rx_results.md，恢复和续接见 ../../ev_rx_reproduce.md。

四份原始记录归档保留开发box/cone及独立36校准/12评价的实际JSON字节，附源码快照。恢复报告逐项核对原SHA；阈值与评价统计从恢复文件重算后也逐字节相同。development_summary与formal_summary为纯读取汇总；statistics_audit为从原始拟合损失独立重建统计的36749项检查。压缩JSON可用Python gzip读取，未改写其原始内容。

precision_proof_check核对统一精度下平方基线的解析上界；development_failure保存992101的预测/实际排序反转。unit_tests.xml为126项针对性回归。完整AC物理回放报告在四个原始包内；校准/评价分别14160/4800项检查通过。

统计单位是基础历史：4开发（两个配对训练条件）、36校准、12评价，总52份。19方法、12有向任务及相同曲线不增加独立历史数。实际评分完成与代理最优间隙达标分别保存。以上范围为声明平衡三末端电缆族模拟。
