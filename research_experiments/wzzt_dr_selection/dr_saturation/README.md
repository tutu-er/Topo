# 冻结报价的设备饱和配对复算

本批只把原实际DR增量U截断至[-a,a]，等价于柔性负荷功率a+U投影到[0,2a]。主入口runner.py依次prepare、observe、evaluate，--output指定新私有目录。原144条无截断结果作为对照，新增144条截断AC记录，所有随机路径、报价、容量、完整历史、R/X和选模规则保持不变。

实现复用dr_spatial_correlated/ac_bridge.py的可选delivery_rule='device_clip'；默认'unbounded'不变。原AC求解、P/Q/V测量函数和冻结选模器直接调用。无优化、重拟合、候选搜索或新网络。费用仍为原预付合同价，不改为实际交付量结算。

完整数学结构、概率积分及设备合理性见MODEL.md；结果见REPORT.md。post/audit.py --run 私有run目录用于独立算术/守恒重放，不运行AC或优化。post/export_and_plot.py --run ... --output ...发布显式标量白名单。

私有批次：C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_saturation_stage/run_v1。完整路径及报价、P/Q/V/RX和随机数组不上传。

运行环境D:/apps/miniconda3/envs/Topo/python.exe。测试：python -m pytest research_experiments/wzzt_dr_selection/dr_spatial_correlated/test_ac_bridge.py research_experiments/wzzt_dr_selection/dr_spatial_correlated/test_saturation.py -q -p no:cacheprovider。13项桥接纯测中5项新增饱和检查。

记录保留：第一次evaluate在观测清单尚未冻结前调用，缺少observations_frozen.json而退出；日志evaluation.log保留。144次AC完成后仅重试评分，成功日志evaluation_completed.log；未因此重跑AC。
