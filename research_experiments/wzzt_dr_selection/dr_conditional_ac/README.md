# 冻结连续DR的AC评价

先读REPORT.md。现有物理producer与表计函数直接复用，原冻结候选/参数/3%/w2不变。本目录没有新增价格优化器或拓扑辨识器。

- runner.py：prepare（锁输入、报价、源码、12背景身份与种子）→observe（60AC日）→evaluate（240个四噪声选择，之后读取真值标签）。
- ac_bridge.py：原AC与原测量调用；记录容量越界而不中止、不裁剪、不重抽。
- test_ac_bridge.py：无真实网络的6项纯测试。
- load_audit.py：仅历史完整数据与已有报价的总量核算。
- post_analysis.py：仅已完成AC的量与负荷比例、配对增量误差；不重新仿真。
- audit.py：已完成原始结果的独立复算；不启动AC。
- artifacts.py：公开标量证据与图片，逐时轨迹不公开。

运行要求先核对恢复索引和ended；已有run_v1全部完成，不重复observe。私有源快照及hash为执行证据，仓库源码可用于新的显式协议，不能覆盖原冻结记录。

私有根：C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_conditional_ac_stage。公开证据：../evidence/conditional_ac_v1。
