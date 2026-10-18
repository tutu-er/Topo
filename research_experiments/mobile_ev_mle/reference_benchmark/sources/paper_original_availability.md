# 既有论文原始算例的可获得性与本轮复现边界

2026-10-18；约十分钟的定向补查。未改动 `networks.py`，未新增科学算例或运行拟合。继续复用本地全文，并阅读作者公开原文和网页；没有新增论文PDF下载或订阅认证。

## 结论

在本次有限检索范围内，**没有找到能够逐边核对的 Bhela IEEE34、Cavraro/Taheri IEEE13/37、Kapoor IEEE37 作者最终单相等值数值文件**。这些文章确实使用相应标准网，但“IEEE原始三相数据公开”与“作者采用的单相改造参数已核验”是两件事。不能把自行计算的正序或均值阻抗命名为作者原始算例。

当前完整case33与CIGRE居民LV组有可核对官方数值表；paper11/paper15仅为旧项目合成兼容组。它们可满足“更真实的标准网络测试”，但不能把论文原始实例严格复现这一项写成完成。

## 检索范围和实际读取证据

1. 本地 `papers/mobile_ev_probing/manifest.json`、`research_experiments/mobile_ev_mle/evidence/ev_related_cases/provenance.json`，以及既有 `ev_related_case_review.md`。
2. Bhela Part II本地9页arXiv v2 PDF，重点PDF第5–9页的§IV、原始负荷来源和参考文献；命令行pdftotext读取，不是本轮视觉复核。
3. Cavraro图算法 arXiv v2 §VII；拓扑处理 arXiv v3 §V及参考文献；Taheri arXiv v2 §V。以上实际公开正文已读。
4. Kapoor用户提供的正式9页PDF，重点§IV–V、所有代码/补充材料相关文本命中及参考文献；并读其另一篇2025年开放文章的实验说明。读取程序为本地pdftotext。
5. [Kekatos作者官方论文目录](https://engineering.purdue.edu/~kekatos/publications.html)：这些论文项只有PDF/arXiv链接，未列数据/代码链接；同页确实给若干其他论文单列code，故本次检查了实际作者页面，而不是仅凭搜索无结果判断。
6. 公开GitHub仓库元数据精确题名短语查询：`Smart Inverter Grid Probing`、`Inverter Probing`、`Graph Algorithms for Topology`、`Maximum Likelihood`且`Line Parameters`，返回仓库数均0。它只排查标题/描述索引，未检索所有仓库内部文件，不能由此证明作者从未公开代码。

## 逐篇结果

|论文与当前实际读取范围|原网络口径|本次已核实内容|精确复现仍缺少什么|
|---|---|---|---|
|Bhela等2019 Part II，[arXiv1806.08836v2](https://arxiv.org/pdf/1806.08836v2)，本地第5–9页|IEEE34，多相改为单相|§IV明确如此描述并引用Gan等2012；P采用Pecan Street指定日期，Q按0.9滞后PF生成|作者转换后的逐线R/X、基准、调压器/电容/变压器处理及节点映射；确切客户曲线身份与合法数据副本|
|Cavraro–Kekatos2018图算法，[arXiv1803.04506v2 §VII](https://arxiv.org/html/1803.04506v2#S7)|IEEE37单相等值|§VII引用其Voltage Analytics旧文；秒级探测、全部/部分测量设置可确认|作者数值R/X和标准母线对应表，不能只凭Fig.1重建线路参数|
|Cavraro–Kekatos2019拓扑处理，[arXiv1802.06027v3 §V](https://arxiv.org/html/1802.06027v3#S5)|IEEE13/37单相改造，额外连线后7/21径向状态|原文给网络示意和候选状态族，引用Voltage Analytics作为单相转换依据|原闭合线数值、附加线R/X、精确候选开关状态与单相转换结果|
|Taheri等2019，[arXiv2004.02370v2 §V](https://arxiv.org/html/2004.02370v2#S5)|IEEE13 modified single-phase|原文为12真线加12附加候选线，所有叶点有足额逆变器，完整AC生成电压|数值单相线路表、附加12线和变压器/调压器参数；原文正文没有逐线转换公式|
|Kapoor等2025 MPCE，[DOI10.35833/MPCE.2024.000910](https://doi.org/10.35833/MPCE.2024.000910)，本地正式PDF§IV–V/refs|IEEE37已知拓扑，单相DistFlow模型|原文引用Kersting标准测试网；NextGen负荷另送入AC生成计量；读到Supplementary Material A/B/C用于数学证明|作者最终单相逐边真R/X、变压器处理和源代码；所读正文未列代码地址或完整真参数表|

Kapoor另一篇 [Estimation of distribution grid line parameters using smart meter data with missing measurements](https://doi.org/10.1016/j.ijepes.2025.110634) 的公开§4也明确采用single-phase IEEE37，但该句并没有补齐作者单相参数表；本次不将另一篇的设置替换成MPCE原设置。

### 引用链也没有自动补齐参数

Cavraro图算法 → Voltage Analytics → Gan2012；Bhela Part II直接引用Gan2012。

[Voltage Analytics公开正文§V](https://arxiv.org/html/1707.06671#S5) 明确说按引用文献转换为单相，但所读正文未列逐相到单相的计算式。Gan2012作者目录指向 [Caltech作者稿](https://smart.caltech.edu/papers/ExactRelaxation.pdf)，当前该文件题名是《Branch Flow Model for Radial Networks: Convex Relaxation》，而目录项名是《On the exactness…》，存在版本题名差异。本次读取其首页、§IV和Table II及`phase`全文查找：其数值表是SCE真实47母线馈线，未读到IEEE13/34/37转换过程。不能据此补写某个“作者采用正序平均”的转换规则。Harvard另一同题PDF公共抓取返回403，未用受限地址绕行。

## 本轮可诚实采用的两种方案

### A. 本轮先交付已核验标准网，论文原算例留明确缺口

完整case33与CIGRE居民网的优势是物理Ω/kW数据已逐条核对，能马上进行完整AC和内节点负荷测量测试。旧paper-style组用于延续历史比较，但表格标题写“项目合成兼容组”。报告单列：“Bhela/Cavraro/Taheri/Kapoor作者单相原数据尚未取得，本轮没有声称重现其数值结果。”

这一步可以继续当前计算，不依赖等待作者数据。若后续用户要求严格复现其一，应取得其单相数据或逐项转换说明；联系作者需要用户明确授权发送邮件，目前未发送。

### B. 新建有完整变换记录的IEEE13/37公开标准派生组

若确需补一组相同标准名称的小网，可另行预声明派生规则；名称必须包含 `declared_single_phase_derivative`。建议优先13母线降低计算量，原标准数据可沿 [EPRI官方说明](https://opendss.epri.com/ExamplesDocsandTestCases.html) 的IEEE测试馈线资源获取。

导出步骤：

1. 保存原三相数据版本、文件SHA、母线ID、逐相线路矩阵、变压器和调压器；先独立运行原模型AC验证。
2. 在看到辨识结果前确定单相策略：明确选择某相完整子树，或明确采用给定的正序/平衡等值；原来缺相的支路不能凭空补齐相后称原网。
3. 计算单相参数并保存每条边的变换公式和原始行。对完整矩阵，正序等值应由明确的对称分量基向量投影得到，而不是不记录地取对角线均值。变压器折算、电容、调压状态同样记录。
4. 保留所有负荷所在的原物理节点，P/Q不搬移；隐藏degree2点只做有证明的串联等值，零设备连接只做精确收缩或等压观测。
5. 以原三相AC作为外部误差对照，分别报告单相模型的幅值/响应误差；如果误差高于计量噪声，就把它归入模型失配，不将其计作EV策略失败。
6. 导出沿用当前schema：`buses/branches/root_bus/base_kv/base_kw/observation_bus_ids/source_bus/provenance`，再增加 `source_phase`、`phase_reduction_method`、`physical_edge_map`、`transformer_map` 和 `three_phase_reference_error`。源P/Q和EV服务接入只作为声明改造，不混入原论文参数。

方案B可提供“相同标准网、透明单相派生”的可复现新实验；它仍与“作者原始数值例复现”分开。当前没有改代码或运行该派生组。
