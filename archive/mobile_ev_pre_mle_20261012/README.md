# 2026-10-12 EV 实验整理

这是研究路线变更后的可恢复归档，不表示所有旧实验在其原假设下无效。
原始数据和负结果保留；新入口为 [MLE 与 EV 设计](../../research_experiments/mobile_ev_mle/README.md)。

## 已做的整理

- 早期未纳入 Git 的结果目录移入本目录 `payload/`，每个文件移动前后核对 SHA-256。
- 同样归档旧主 README 和会覆盖现有 LaTeX 文件的两个脚本 `build_report_v2.py`、`update_open_report_v4.py`。
- 删除可重建的测试临时目录、Python 缓存及 v9 两批过期 PDF 渲染；保留最后一批渲染和对应校验记录。
- `results_v6` 至 `results_v9` 的正式数据、源码、反例和审计路径不变。
- 当前 LaTeX 源文件及 PDF 未改动。Git 已提交的实验文件未删除或搬移。

具体路径、文件数、字节数和清理理由见 [cleanup_summary.json](cleanup_summary.json)。
完整逐文件清单是 `payload/<目录或文件名>.inventory.json`，其哈希也记录在摘要中。
`payload/` 仅在本机保存并被 Git 忽略，避免上传大量原始数据；新克隆不含该归档。
移动文件不会释放磁盘空间；只有明确记录为 removed 的临时产物被删除。

## 恢复

先查摘要中对应的 `from`、`to`，确认两者解析后的绝对路径都在本工作区内。
源路径应不存在；如果已存在，不要覆盖。核对对应 inventory 的文件尺寸和 SHA-256 后，使用 PowerShell 的 `Move-Item -LiteralPath` 将 `to` 移回 `from`。

恢复旧 README 时新 README 已存在，应先保存新入口，避免覆盖。
运行早期脚本前恢复其所需结果目录；脚本中的路径仍指向原位置。递归发现历史测试时设置 `TOPO_RUN_LEGACY_EV_TESTS=1`；显式测试文件路径仍可直接运行。
恢复旧报告生成器后也不要直接运行，以免把目前报告覆盖为早期内容。
