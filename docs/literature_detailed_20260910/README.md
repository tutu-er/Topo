# 配电网拓扑辨识文献逐篇解析

2026-09-10。共22篇文献、45页，正文约29,100个汉字，111组编号公式环境。原论文陈述、本文补充推导和访问不足分别标记；没有声称重新运行原论文实验。

## 直接使用

- `topology_literature_detailed_20260910.tex`：独立单文件，已内嵌参考文献，不需要其他`.tex`或`.bib`即可编译。推荐将此文件上传至Overleaf并选择XeLaTeX。
- `main.tex`、`sections/`、`refs_*.bib`：方便继续修改的模块化源码。
- `sources_*.md`：逐篇访问URL、版本、关键公式和证据边界。
- 仓库中的最终PDF：`output/pdf/topology_literature_detailed_20260910.pdf`；源码ZIP中也附同名PDF。

单文件编译：

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error topology_literature_detailed_20260910.tex
```

如果只使用`xelatex`，应重复编译直到目录和交叉引用稳定（本次需要3遍）。单文件已经内嵌BibTeX生成的书目，不需再运行BibTeX。

模块化编译：

```powershell
.\build.ps1
```

或：

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

模块化编译结果为`build/main.pdf`。修改后可运行`python assemble_standalone.py`重建单文件快照；该小脚本只用Python标准库。字体使用TeX Live自带Fandol，避免依赖本机商业中文字体。

## 文献范围

沿用此前调研的S1--S17，并增加P1--P5（逆变器探测、电流注入、高频传播、缺失FSR、脉冲压缩）；S12不重复计数。

全文访问仍不足的条目：S2、S4、S6。S5依据出版社可检索的正文片段；S10和S16等按明确注明的作者公开版本审读，不假定版本细节完全一致。S2的简短方法概述沿用先前出版预览记录，本次重新核实的是出版元数据。所有受限条目中的教学推导均明确标注，不冒充论文原式。

内容包括：问题与量测配置、数学模型、关键机制推导、算法流程、可辨识条件、实验和证明的适用范围、与当前Topo代码的对应，以及可以检验的后续研究问题。

## 验证

- 模块化与独立单文件均由XeLaTeX/latexmk编译成功，均45页。
- 22篇独立章节与22个书目条目齐全；无重复标签、未解析引用、缺字或溢出警告。
- 全45页已渲染，程序检查无页外字符；排版另行逐页视觉检查。
- 对四点判别间隔、S3三相距离适用域、S4示意EM、S7证明接口、S16几何/MH核验进行了交叉审查。
- `build/qa_report.json`为本次自动检查结果；源码包附`qa_report.json`。

原始论文PDF和关键页截图仅在工作区`sources/`及临时审查目录中留存；源代码ZIP不重新打包这些原始论文。报告编译不依赖这些快照。
