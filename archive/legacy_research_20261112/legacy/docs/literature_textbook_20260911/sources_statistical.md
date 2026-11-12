# 教材统计章 S14--S17 的来源

核验日期：2026-09-11。复用上一版已经阅读的 primary 全文，并补核 S16 Algorithm 1 的实际更新步骤。参考文献键保持 S14、S15、S16、S17，与全书编号一致。

## S14

- [期刊与 PMC 元数据](https://pmc.ncbi.nlm.nih.gov/articles/PMC5082587/)
- [arXiv v2 全文 PDF](https://arxiv.org/pdf/1306.6430v2)，32 页，2016-01-28。
- 对应位置：第 1.1 节式 (7)--(8)，第 3 节。
- 新稿身份：变分目标与指数更新来自原文；有限树参数格点、logsumexp 伪代码、两组数值例子是等价计算与教学变体。未把有限候选构造归因给作者。

## S15

- [PNAS DOI](https://doi.org/10.1073/pnas.1922664117)
- [arXiv v4 全文 PDF](https://arxiv.org/pdf/1912.11436v4)，28 页，2022-10-19。
- 对应位置：第 2 节 Theorem 1、Theorem 3；第 6 节条件似然讨论；第 8 节序贯扩展。
- 新稿身份：split LR 核心及覆盖机制来自原文；逐树 profile 和上下界状态机是为 Topo 提供的教学整理；三个高斯区间均值模型是独立构造，不代表三棵实际馈线。

## S16

- [arXiv 条目](https://arxiv.org/abs/2401.11515)
- [作者网站主文 PDF](https://zhenkewu.com/assets/pdfs/papers/ultrametric_inference_main.pdf)，29 页。
- [arXiv v1 HTML 补充材料](https://arxiv.org/html/2401.11515v1)。该 HTML 不是主文，不能据其缺少 Algorithm 1 推断全文没有算法。
- 新核查重点：主文印刷 p.17 Algorithm 1。第 4 步明确使新分裂继承被替换边的长度；拓扑更新之后对当前边集合逐条作截断正态长度提议。第 5.1 节给两个邻域及接受比。
- 已有证据保留在旧报告 sources/S16_author_main.pdf、S16_page7.png、S16_page15.png；详见本目录 audit_statistical.md 的校验值及版本批注。
- 新稿身份：状态数据结构、四分支 NNI 枚举说明、对数接受步骤、协方差缓存和数值例是教学实现整理。正半轴高斯的完整归一化推导为本文推导。
- 本次没有读取或运行作者 GitHub 实现，因此只确认论文流程，不声称“作者代码已复现”。

## S17

- [固定 arXiv v6 PDF](https://arxiv.org/pdf/2107.07511v6)，51 页，提交日期 2022-12-07，标题页 2022-12-08。
- 对应位置：第 1.1 节标准 split conformal；附录 D 的 Theorem D.1、D.2。
- 新稿身份：校准分位数与预测集合是标准算法的显式整理；电压残差、向量区域和整网标签例子为独立构造。
- 用固定 PDF 版本给书目信息，不把 HTML 的动态生成日期当发表时间。

## 版本及引用原则

正文每节均声明原算法、等价整理或教学变体。本文推導的实例、成本分析和模型迁移不等于原论文已验证的应用。此前原文访问限制和公式版本问题继续保留，新稿没有把它们静默删除。

