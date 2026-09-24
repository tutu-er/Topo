# 当前核心模块：功能与源码对应

审查日期：2026-09-19。范围为 `rnj_wzzt_core/run.py`、`rnj_wzzt_core/rnj_wzzt/` 及依赖声明。

- `main.tex`：中文 LaTeX 主文档。
- `source_snapshot/`：本次审查的原始源码快照，摘录行号以此为准。
- `source_manifest.json`：逐文件 SHA-256、行数及 AST 符号位置。
- `generated/`：自动生成的源码索引、引用和核验结果。
- `build.py`：刷新快照、生成索引并验证引用；使用 `--check` 检查当前源码是否漂移。
- `evidence/`：本次测试与运行记录。

在本目录执行（conda 环境 Topo）：

```powershell
python build.py --check
xelatex '-interaction=nonstopmode' '-halt-on-error' '-output-directory=../../tmp/pdfs/core_module_code_20260919' main.tex
xelatex '-interaction=nonstopmode' '-halt-on-error' '-output-directory=../../tmp/pdfs/core_module_code_20260919' main.tex
```

在仓库中重新生成快照/索引可运行 `python build.py`。它不会修改核心源码；更新快照后须重新人工核对正文及代码摘录范围。现有 PDF 位于 `output/pdf/core_module_code_20260919.pdf`。

随附源码包可以脱离仓库编译：保留此目录结构，在本目录直接运行两遍 `xelatex main.tex` 即可。此时不需运行依赖仓库位置的 `build.py`。

## 本次验证

最终 PDF 为 39 页。核心现有测试 695 项通过，相关父项目测试 46 项通过；39 页均已渲染并完成视觉检查，最终编译无超宽、缺字或未解析引用警告。小样本运行只用于验证完整流程，不代表默认配置的恢复性能，详见正文。

编译依赖：XeLaTeX、ctex、fvextra、TikZ，以及 Windows 中文字体和 Consolas。其他操作系统请相应调整 `fontset` 与等宽字体。

相关父项目测试文件：`test_mainline_entrypoint.py`、`test_rx75_rnj_milp_hybrid_ac.py`、`test_laminar_l1_milp.py`、`test_rooted_neighbor_joining.py`、`test_rooted_hierarchy.py`、`test_sensitivity_geometry.py`、`test_ac_powerflow.py`、`test_matrix_constraints.py`。
