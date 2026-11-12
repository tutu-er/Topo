# 退出当前主线的代码与文档：2026-11-12

当前研究入口为根目录的 `rnj_wzzt_dr/`。本档案收纳早期拓扑理论／文献、旧 L1/MILP 算例工程、历代 RNJ/WZZT 实验、旧偷电工程、旧仓库入口及编译成品；原始内容按 `legacy/<原路径>` 保存。

`manifest.json` 记录逐项旧路径、新路径、字节数及 SHA-256。对原已跟踪文件保存公开明细；本地 `LOCAL_MANIFEST.jsonl` 还覆盖原未跟踪／忽略文件。归档不会把这些本地资料自动上传，`legacy/` 默认忽略，只有此次明确列出的原已跟踪文件进入 Git。

没有移动新主线、正在运行的独立 96 点目录、`rnj_wzzt_core` 的未提交研究、两个未跟踪研究目录、Git 工作树，以及 `outputs/artifacts/data_external` 中的原始数据、冻结物理源和恢复证据。具体保留范围见根目录 `KEEP.md`。

## 核验与恢复

```powershell
python -B archive/legacy_research_20261112/verify.py
python -B archive/legacy_research_20261112/verify.py --include-local

# 默认只核验和预览；-Only 选择一个旧路径前缀。
./archive/legacy_research_20261112/restore.ps1 -Only research_experiments/rnj
# 加 -Apply 后复制回原路径，不覆盖当前已有的不同文件。
./archive/legacy_research_20261112/restore.ps1 -Only research_experiments/rnj -Apply
```

恢复本地未跟踪内容时再加 `-IncludeLocal`。恢复完整旧工程宜使用独立旧版本检出；当前根 README/pyproject 已更新，恢复脚本会拒绝覆盖它们。

历史脚本和协议中原有绝对路径、源码哈希及相对链接保持原文。归档区作为历史材料浏览；需要重放时先恢复相应代码与数据路径，再按旧协议核验。当前主线不导入本目录，未将旧结果改称新算法结果。
