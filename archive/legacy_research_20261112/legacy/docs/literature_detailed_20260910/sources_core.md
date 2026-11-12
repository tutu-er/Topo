# S1--S4 来源审计（2026-09-10）

- S1: https://arxiv.org/html/2206.10837v2 全文。II-B 式(4)--(8)：共同路径和平方电压因子；III-C：部分节点观测和Kron；IV：智能表配置；式(24)：电压协方差。最终报告只简短归纳教程，其余是独立模型推导。公式校验 a0=-A1 直接由完整关联矩阵行和为零得到。
- S2: 官方 https://api.elsevier.com/content/article/PII:S2352467726002675?httpAccept=text/xml 返回metadata-only XML；https://api.crossref.org/works/10.1016/j.segan.2026.102385 核实7作者、题名、DOI、2026-09卷期。分别保存 `sources/S2_publisher.xml` 和 `sources/S2_crossref.json`。ScienceDirect网页和abs均403，未获取全文。方法概述只沿用先前本地报告记载的出版预览，当前未复核预览。不得把示意QP、分组和Gauss-Newton推导归于论文原式。第三方摘要/ResearchGate仅用于查找，不作方法证据。
- S3: Monash 作者机构记录 https://research.monash.edu/en/publications/three-phase-voltage-sensitivity-estimation-and-its-application-to/；新取得作者机构全文 https://minerva-access.unimelb.edu.au/server/api/core/bitstreams/26bfcb36-fcc3-4676-9b6d-9099b38c3e13/content，保存PDF与pypdf抽取文本。PDF共19页（含首张机构封面），论文18页。核查 §2.1式(1)--(12)，§2.2式(14)--(21)，§3.1--3.4及Algorithm1，§4.3，§5.1.1，§5.2与§6。特别边界：current sensitivity不同于power sensitivity；投影的电压侧cos误差二阶但整个SI近似不能一概称二阶；仅有限候选；same-phase halving额外假设相线和中性线阻抗相等；作者结论承认决策规则heuristic。评分取模反例为本报告分析，不是原文实验。
- S4: https://ieeexplore.ieee.org/document/9696306/ 搜索缓存的出版摘要（直接访问JS challenge）；作者名用IEEE Communications Society官方出版目录 https://www.comsoc.org/system/files/2022-06/Publications_Contents_Digest_2022_May.pdf 互核。未获全文。仅确认潜树、候选搜索、BIC、EM和末端智能表范围；高斯分块条件分布、EM Jensen证明、BIC边界均是明确标记的示意推导。

本报告没有独立重跑任何原论文代码或声称重现实验。网页访问限制与数学断言本身的可靠性分开记录。
