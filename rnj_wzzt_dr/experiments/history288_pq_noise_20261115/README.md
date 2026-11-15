# 288点历史与P/Q噪声网格

用户确认三天历史按192点拟合＋96点验证使用，第4天96点评价。固定Soumalas、Pengwah，P/Q附加噪声0/.25/.5/1/2%，每档4个历史重复，共40次重新辨识；V/root误差固定.05%，四类设备和预测固定。

执行前方案见 [PROTOCOL.md](PROTOCOL.md)，模型口径见 [MODEL.md](MODEL.md)。本轮是两个已知物理背景上的配对敏感性试验，不能把噪声重复称为独立实网。

本批已完成40次重新辨识、120份报价和2560条策略记录，独立审计全部通过。主要结果和失败分析见 [RESULTS.md](RESULTS.md)：三种采购方法均执行零动作，不能据本批报告非零采购提高了准确率。全报告、设备明细和效果图保存在下列ignored输出目录：

- `outputs/history288_pq_noise_report_20261115_v1`
- `outputs/history288_pq_noise_plot_20261115_v1`

复用既有384点物理缓存、当前辨识器和四设备域，不修改生产默认。旧缓存及依赖模块仍由源码快照/哈希列出；本目录是本轮通用入口，并非无外部依赖的独立软件包。

实际阶段入口如下。输出均有完成标记与锁，不可覆盖已完成目录；复现时应更换目录名并同步修改后续引用。

```powershell
python -m pytest rnj_wzzt_dr/experiments/history288_pq_noise_20261115 -q
python -m rnj_wzzt_dr.experiments.history288_pq_noise_20261115.data --out outputs/history288_pq_noise_data_20261115_v1 --workers 3 --prepare-only
python -m rnj_wzzt_dr.experiments.history288_pq_noise_20261115.data --out outputs/history288_pq_noise_data_20261115_v1 --resume
python -m rnj_wzzt_dr.experiments.history288_pq_noise_20261115.audit --data-index outputs/history288_pq_noise_data_20261115_v1/data_index.json --out outputs/history288_pq_noise_data_audit_20261115_v1
python -m rnj_wzzt_dr.experiments.history288_pq_noise_20261115.run --index outputs/history288_pq_noise_data_20261115_v1/data_index.json --out outputs/history288_pq_noise_smoke_20261115_v1 --smoke
python -m rnj_wzzt_dr.experiments.history288_pq_noise_20261115.audit --run outputs/history288_pq_noise_smoke_20261115_v1 --out outputs/history288_pq_noise_smoke_audit_20261115_v1
python -m rnj_wzzt_dr.experiments.history288_pq_noise_20261115.run --index outputs/history288_pq_noise_data_20261115_v1/data_index.json --out outputs/history288_pq_noise_execution_20261115_v1 --preflight outputs/history288_pq_noise_smoke_audit_20261115_v1/report.json --seconds 1200 --workers 3 --prepare-only
python -m rnj_wzzt_dr.experiments.history288_pq_noise_20261115.run --out outputs/history288_pq_noise_execution_20261115_v1 --resume
python -m rnj_wzzt_dr.experiments.history288_pq_noise_20261115.audit --run outputs/history288_pq_noise_execution_20261115_v1 --out outputs/history288_pq_noise_audit_20261115_v1
python -m rnj_wzzt_dr.experiments.history288_pq_noise_20261115.report --audit outputs/history288_pq_noise_audit_20261115_v1/report.json --out outputs/history288_pq_noise_report_20261115_v1
python -m rnj_wzzt_dr.experiments.history288_pq_noise_20261115.plot --report outputs/history288_pq_noise_report_20261115_v1/report.json --out outputs/history288_pq_noise_plot_20261115_v1
```

原始数据、拟合矩阵和逐路径轨迹保存在ignored outputs，仓库仅提交代码、协议与汇总。
