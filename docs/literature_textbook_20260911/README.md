# 配电网文献算法：教材式解析与完整性审查

2026-09-11 教材式修订。保留旧版 `docs/literature_detailed_20260910/`，本版不修改拓扑研究实现。

## 阅读文件

- `topology_literature_textbook_20260911.tex`：可独立编译的完整单文件，已嵌入全部章节和参考文献。
- `main.tex` + `chapters/` + `refs_*.bib`：便于逐章修改的模块化工程。
- 最终 PDF：仓库 `output/pdf/topology_literature_textbook_20260911.pdf`；源码 ZIP 也包含一份。
- `sources_*.md`：原始文献链接、版本、节号/式号及访问限制。
- `audit_*.md`：逐专题解释完整性、数学核验与本次修订说明。

## 内容

94 页，22 项文献，25 段正式算法伪代码、24 个编号教学例、8 道带参考解的练习。按共同模型、被动隐藏树、数据质量、主动探测、统计推断、运行包络、综合比较及审查附录组织。

旧稿适合研究综述和边界审查，算法教材仍缺统一入口、状态更新、停止分支和完整数值轨迹。本版围绕这些缺口重写，明确区分原文流程、等价代数整理与教学变体。

S5 本轮补到出版者全文；S2、S4、S6 仍未取得全文，对应章节保留来源缺口。不能将本书理解为 22 项论文全部实验已复现。

## 编译

使用包含中文 Fandol 字体的 TeX Live，或在 Overleaf 选择 XeLaTeX。

模块化工程：

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

也可以运行 `build.ps1`，其会检查溢出、缺字和未解析引用。模块化工程由 latexmk 自动调用 BibTeX。

单文件版：

```powershell
latexmk -xelatex topology_literature_textbook_20260911.tex
```

单文件已嵌入参考文献，不依赖外部章节、图片或 `.bib` 文件。重复 XeLaTeX 直至交叉引用稳定也可。

改动模块化源码后，先成功编译，再运行 `python assemble_standalone.py` 更新单文件快照。

## 验证范围

- 两种入口均编译成功；经 pdfplumber 提取的 94 页文本逐页一致。
- 22 个文献键全部有引用；无重复标签、未解析交叉引用、缺字或溢出告警。
- 全部页面已渲染并视觉检查，最终变更页面重新检查，未改变的正文区域与已审查页面作像素比较。
- `verify_math_examples.py` 检查公共模型、树距离、RG、EM 等教学算例。
- `verify_document.py` 检查引用、页边界和单文件一致性，需要 pdfplumber。
- 各专题代理交叉核对核心公式和数值例；发现并修正 S15 指数值、S3 失败分支及若干排版问题。
- 这不是原论文代码或实验的独立复现，不提供新增原论文成绩。

验证结果在 `build/qa_report.json` 与 `build/math_checks.json`；源码包中保留这些结果。公式与来源冲突在正文末尾及审查附录明确记录。
