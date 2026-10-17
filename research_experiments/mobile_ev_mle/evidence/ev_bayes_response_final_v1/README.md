# 冻结批次完成包

先读研究目录的 `ev_bayes_response_final_report.md`。本包对应16开发、64校准、128评价；失败分母全部保留。279项回归通过，统计、物理响应、历史支持及费用已完成核对。最终结果没有建立新贝叶斯MILP优于同域平方MILP的证据。

- `postprocess_evidence.zip`：全量统计JSON/CSV、AC响应诊断、历史支持、XML、运行恢复证据、140份冻结源码快照及其逐字节源码、配套模块、原始记录索引及哈希。共有875个成员，内含 `archive_index.json`。
- `evaluation_000.json.gz`、`evaluation_002.json.gz`：评价中索引最小的成功与失败记录，分别用于完整恢复核查，不作为效果对照或优胜样本。
- `delivery_manifest.json`：ZIP、样本、环境和主报告的SHA256，以及实际恢复检查。
- `environment_record.json`：Python、NumPy、SciPy、Gurobi及pytest版本。原科学环境另见ZIP中的 `run/freeze.json`。

原始208份gzip共8,166,735,249 bytes，保留在：

`D:/0-github_workspace/Topo/outputs/ev_rx_stage_20261015/outputs/ev_bayes_response_frozen_v1`

ZIP包含全部原始哈希和种子，未上传其余大型原始记录。全量统计重算需要这些记录，或重新运行原冻结协议；有时限的MILP重算不保证得到逐字节相同解。

## 查看与恢复

1. 按外部manifest中的SHA256核对文件，再将ZIP解压到新的空目录。`repository/`是源码，`postprocess/`是已完成结果，`run/`是冻结配置与阶段审计。
2. 如需查看代表性原始输入，在解压目录新建 `run/evaluation/`，把两份外部样本复制为 `history_000.json.gz`、`history_002.json.gz`。
3. `archive_index.json`列出除自身外的874个成员的大小和SHA256，可核对解压文件。不要用缺少其余206份原始记录的样本目录冒充完整208份输入。

研究工具 `ev_bayes_response_delivery.py`已经在全新目录执行了实际恢复核查：875个归档成员及2份样本均逐字节一致，140份科学源码与原freeze/snapshot一致，恢复源码没有执行。另一个独立核查保存在研究目录 `ev_bayes_response_final_bundle_verification.json`。

研究源码分支为 `codex/ev-train-validation-design`。包内记录打包时的源码提交和原运行绝对路径，路径是来源信息；在其他机器查看结果不需要该路径存在。原长LaTeX/PDF及pyproject修改均未纳入。
