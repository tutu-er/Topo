# 损失口径、后验门槛与二次规划

本次交付只整理理论、核对现有代码并构建文档，没有修改辨识或报价代码，没有新实验。

- `main.tex`：完整中文推导，所有符号在使用前定义。
- `references.bib`：四份核对过的公开背景来源。
- `main.pdf`：17页阅读版，与源文件一并保存。
- `BUILD.json`：源、书目、PDF指纹与编译/页面核验。

推荐口径：保留原RNJ＋WZZT的MAE作为主评价；Gaussian平方损失是明确命名的替代规划/评分版本。若要求后验与原绝对损失同源，使用Laplace似然并接受绝对值的求解结构。不能仅为使用QCQP而将两者说成等价。

文档包含：

1. L1/L2的统计含义、Gaussian/Laplace密度与最大似然，总和/平均/单位的区别。
2. 当前DR适配器0.1%训练接受、3%验证带、输出收缩及完整历史门槛，区分通用WZZT核心。
3. 固定协方差的Bayes平方差，及MAE与SSE排序相反的具体例子。
4. 反Gamma定义、换元积分证明、未知共同尺度的Gaussian和Laplace相对门槛。
5. 共同误差、均值、相关协方差和交叉项，误差底如何由相对系数产生。
6. 相关仿射随机响应的完整二次系数、设备/费用约束、QCQP/MIQCP和全局界。
7. 后验阈值、执行达成率与期望裕量的区别，二次Gaussian变量的矩及名义概率特殊情形。
8. 现有结果、代码行索引、已推导/已实现边界和公平对照安排。

新的未知尺度和一般相关因素结果目前仅为理论，原实现仍是固定对角Gaussian似然。反Gamma混合造成统计依赖，不能自动将边际分布继续当成Gaussian；文档专门区分了这一点。

构建：

```text
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

需要支持中文的XeLaTeX、CTeX/Fandol字体、natbib与BibTeX。编译过程中不启用shell escape。主文件可在Codex的LaTeX编辑器中打开，内置编译已核验。

本地XeLaTeX/BibTeX与内置编译均成功；51个交叉引用标签和4条书目核对通过，最终日志无未定义引用、缺字或公式溢出。17页全部渲染并检查，正文、公式、代码索引和文献表无裁切或重叠。

背景来源阅读范围为公开全文的相关章节/页，未做完整文献综述：Boyd–Vandenberghe第6章、Murphy反Gamma参数化及共轭分析、Kass–Raftery的模型证据定义与先验讨论、Gurobi官方NonConvex文档。WZZT应用公式及代码对应由本次自行推导/核对。
