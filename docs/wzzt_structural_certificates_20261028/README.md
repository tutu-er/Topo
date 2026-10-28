# WZZT 结构、对偶筛选与块必要性

`main.tex` 是中文主文件，`sections/` 是正文，`references.bib` 是独立书目。范围为结构核对、数学推导及已有算式的精确算术复核，不含算法实现或新 LP/MILP 实验。

构建（XeLaTeX + BibTeX，当前使用 Windows 的宋体、黑体和楷体；未将字体文件打包）：

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

只复核本文的有理数运算：

```powershell
python verify_derivations.py
```

源码包解压后可独立运行算术复核；只有位于本仓库目录且相关模型文件齐全时，脚本才刷新源码哈希，否则保留随包提供的静态核对记录。

最终 PDF 位于仓库 `output/pdf/wzzt_structural_certificates_20261028.pdf`。构建中间文件与逐页 QA 图像在本目录忽略的 `build/`、`qa/` 中；`verification/` 保存算术结果、源码哈希及最终核验记录。
