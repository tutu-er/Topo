# S1–S4 来源与版本（教材修订，2026-09-11）

- S1：Deka/Kekatos/Cavraro，公开 v2 (2023-04-27)，https://arxiv.org/html/2206.10837v2 。第 II 节式(1)–(8)为相量、幅值/平方电压及共同路径；III 为同步相量量测；IV 为智能表；V 为候选基础设施的状态检测。本版公共计算例为独立教学推导，非新增 S1 单一算法。
- S2：DOI https://doi.org/10.1016/j.segan.2026.102385 ，官方元数据 https://api.elsevier.com/content/article/PII:S2352467726002675?httpAccept=text/xml ，以及 https://api.crossref.org/works/10.1016/j.segan.2026.102385 。2026-09-11再次以精确 DOI/题名寻找合法正文，未得到全文。上一版 sources/S2_publisher.xml、S2_crossref.json 保存元数据。方法路线来自先前调研预览记录，此次未重获该预览；不能作为详细算法证据。新稿所有 QP/GN 内容明确标作背景模块。
- S3：机构全文 https://minerva-access.unimelb.edu.au/server/api/core/bitstreams/26bfcb36-fcc3-4676-9b6d-9099b38c3e13/content ，作者机构记录 https://research.monash.edu/en/publications/three-phase-voltage-sensitivity-estimation-and-its-application-to/ 。上一版已保存 sources/S3_fang2024.pdf 和逐页抽取 txt；本版重新读取 §2.1–2.2、§3.1–3.4/Algorithm 1、§5.2。PDF 第7页（印刷页6）已于本轮渲染核对父子方向文字和式(22)/(24)/(26)。新稿明确平均外部集合、父子差值方向、零值规则未规定、三点教学收尾非原始控制流，保留原始 Algorithm 1 竞争分支的范围。公式重述不冒称作者代码复现。
- S4：出版页 https://ieeexplore.ieee.org/document/9696306/ 及 DOI https://doi.org/10.1109/TSG.2022.3146205 。作者身份先前经 IEEE 学会出版目录交叉核实；2026-09-11仍未取得全文。出版摘要足以确认潜树、EM、BIC及候选搜索路线，不足以确认具体分布、隐藏节点语义和搜索算子。本版高斯潜星形完整算例是独立教学模型。

所有方法细节以实际读取版本为范围。访问障碍不是论文本身没有相应内容或证明的证据。
