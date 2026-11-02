# 文档构建与核验

主文件 main.tex，结果子文件 results.tex，图位于 figures/。使用 XeLaTeX（TeX Live 2026），不启用 shell escape，连续编译至交叉引用收敛。

```powershell
xelatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
xelatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
```

15页PDF已解析并逐页目视核验。最终日志无未定义引用、缺字或overfull box。检查了符号、公式、候选排除阈值、表格、图与完整失败分母。字体使用Windows中文配置；其他系统可将ctex字体切为已有Fandol配置。

结果为576条配对记录、72独立背景；所有物理观察与候选/报价轨迹留私有。本源包只含文档与汇总图，不含原始数据。
