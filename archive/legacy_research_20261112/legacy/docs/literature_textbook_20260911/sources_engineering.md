# S5—S11 来源、版本与解释层次

核查日期：2026-09-11。继承 2026-09-10 已读 primary 全文，新增核查 S5 完整出版者全文与 S11 扩张/去冗余公式。旧稿目录保持不变。正文用 S5—S11 引文键；数学统一记号、教学求解器与自造数据均有明确标记。以下链接用于定位原文，不把第三方文献摘要或 ResearchGate 页面尾部的其他论文当作证据。

## S5：Ca-TI；本轮由片段级提高为全文级

- 作者：Siliang Liu, Can Deng, Zenan Zheng, Ying Zhu, Hongxin Lu, Wenze Liu。
- 出版：Energies 2026, 19(6), 1503；2026-03-18；DOI [10.3390/en19061503](https://doi.org/10.3390/en19061503)。[MDPI 官方页](https://www.mdpi.com/1996-1073/19/6/1503)。
- 本轮取得 [ResearchGate 的出版者全文](https://www.researchgate.net/publication/402695820_Confidence-Aware_Topology_Identification_in_Low-Voltage_Distribution_Networks_A_Multi-Source_Fusion_Method_Based_on_Weakly_Supervised_Learning)。页面明确标为 Publisher Full-text，并注明由 MDPI 提供访问；正文有作者版权及 CC BY 4.0 声明。使用的是 29 页出版文章正文，不是页面生成的文献概述。直接 MDPI 页面访问曾受限，ResearchGate PDF 下载链接返回 404，但可读正文包含完整算法与公式。
- 核实定位：第 2 节标签/弃权与离散归属；第 3 节式 (16) E 步、(17) M 步、(18) 初始化、算法 1 循环及收敛后的逐源责任度；式 (21)—(24) 熵与缺失折扣；(25)—(29) 冲突、Dempster/Yager、sigmoid 门控；(30) Jousselme 一致性目标；(31) BetP；(32) confidence。
- 新章不再仅写通用 EM/DS 教程：保留真实串联顺序，尤其区分联合 EM 责任度与最终逐源责任度。
- 公式待核处：缺失点数的文字与打印权重公式不协调，教材输入契约为 w∈[0,1]；(W−d)/W 是教材提出的有效比例约定，不声称已经验证作者代码。置信度正文用 product 描述，但显示式 (32) 为线性加权惩罚；教材按显示式。
- 教学补充：log-sum-exp、零软计数行、近完全冲突回退、并列标签、有限四点门控网格、三电表例子及逐项成本。不是作者代码或作者实验。

## S6：只有出版记录；正文完整性仍未通过

- 作者：Ziyue Bian, Huan Long, Huihuang Cai。
- 题名：Distribution Network Topology Identification Based on Co-Training of ResNet and Conformal Prediction。
- CEEPE 2025，303—308 页，2025-04-25；[DOI](https://doi.org/10.1109/CEEPE64987.2025.11033908)。
- 已核 primary 元数据：[Crossref](https://api.crossref.org/works/10.1109/CEEPE64987.2025.11033908)。[IEEE 官方出版页](https://ieeexplore.ieee.org/document/11033908) 需要 JS/机器人检查。
- 本轮额外尝试：再次访问 IEEE，按准确题名和作者机构域名搜索合法作者全文，未找到可核正文。没有购买、请求私有文献或绕过访问控制。
- 未验证：量测输入、ResNet 架构、协同训练顺序、非一致性分数、校准拆分、停止标准、实验与定理。
- 新章的九个分数例子仅为通用 split conformal 的独立教学接口，明示不归属于 S6。正文不将其称为 S6 伪代码或 S6 实验。

## S7：量化误差界；v1 全文

- 作者：Samuel Talkington, Aditya Rangarajan, Pedro A. de Alcântara, Line Roald, Daniel K. Molzahn, Daniel R. Fuhrmann。
- [arXiv 记录](https://arxiv.org/abs/2508.05620)、[v1 HTML](https://arxiv.org/html/2508.05620v1)、[v1 PDF](https://arxiv.org/pdf/2508.05620v1)，2025-08-07，CC BY 4.0。
- 第 II-A—II-C 节：抖动中点量化、约束 generalized LASSO、固定功率因数、R+κX 模型、完全图边权参数化。第 III-B 节：Lemma 1、式 (11) 真值 l1 半径、Theorem 1 及式 (12)—(13)。第 III-C 节：高斯电压设计和拟合常数。
- 原文估计器已明确说明；投影梯度、l1 投影排序、KKT 核算、两节点例子、支持阈值证明均为教材补充，不冒充作者求解代码。
- 已在 HTML 与 PDF 文本核查的数学接口问题：固定真值 l1 球包含所有树权重的文字断言缺少归一化条件；任意 iid 次高斯行还需要非退化设计条件；真实快照构成的多行相关性需对接理论假设。
- 原文仿真常数约 13/10 是实验拟合，不是现场可直接使用的已知常数；全节点已知量测与 Topo 隐藏节点接口不同。

## S8：参考电压相别与馈线修订；作者全文

- 作者：Thibaut Théate, Laurine Duchesne, Adrien Leerschool, Alireza Bahmanyar, Simon Gérard, Thomas Wehenkel, Damien Ernst。
- CIRED 2025 Paper 831，六页。[ORBi 记录](https://orbi.uliege.be/handle/2268/327542)、[作者 PDF](https://orbi.uliege.be/bitstream/2268/327542/1/831.pdf)。
- 第 2 节明确输入归属与电压序列，以及缺乏可靠全量真值。第 3 节：变压器三相参考优先、三相电表累计相关参考、三相簇加离群组、Pearson、max(Cmin,Cmax−αL−βN)、邻近馈线匹配。第 4 节：缺测/异常处理与 RESA 现场案例。覆盖不足 20%。
- 教材补充：有效共同样本与零方差出口、并列保持未决、参考相关和的具体教学约定、四时刻正交向量与两次阈值核算。
- 原文没有充分确定所有相排列、累计方式及缺失掩码的代码细节；教材不假装这些教学约定已经是原代码。

## S9：状态估计残差与人工核验；作者全文

- 作者：Martin Castin, Alireza Bahmanyar, Adrien Leerschool, Laurine Duchesne, Adrien Bolland, Thomas Wehenkel, Simon Gérard, Damien Ernst。
- CIRED 2026 Paper 1423，五页。[ORBi 记录](https://orbi.uliege.be/handle/2268/342287)、[作者 PDF](https://orbi.uliege.be/bitstream/2268/342287/1/Fullpaper_Theme2_1423_Castin.pdf)。
- 第 2 节式 (1)—(2)：r=z−h(xhat)，按 measurement sigma 标准化，阈值 3；第 2.1 节残差地理模式。第 3 节 RESA/pandapower、晴朗夏季中午工况，相别修正、数据问题、馈线重分配。第 4 节把时序自动化列为未来方向。
- 教学补充：WLS Gauss—Newton 迭代、QR/可观性、三表共同均值例、相同量测比较与留出时刻、投影残差协方差和不可检偏移。未声称作者明确采用教材伪代码中的回溯或留出流程。
- 未将 r/sigma 偷换成已做杠杆修正的 residual；删除量测后的低目标不算独立真值证据。

## S10：四线回归与分配；作者全文

- 作者：Achala Kumarawadu, M. Imran Azim, Mohsen Khorasany, Reza Razzaghi, Rahmat Heidari。
- Applied Energy 384 (2025), 125469，2025-04-15；[DOI](https://doi.org/10.1016/j.apenergy.2025.125469)、[Monash 机构记录](https://research.monash.edu/en/publications/smart-meter-data-driven-dynamic-operating-envelopes-for-ders/)、[出版页](https://www.sciencedirect.com/science/article/pii/S0306261925001990)。
- 方法来源：[作者全文](https://www.researchgate.net/publication/388960553_Smart_Meter_Data-Driven_Dynamic_Operating_Envelopes_for_DERs)，页面明确是共同作者 Reza Razzaghi 于 2025-02-13 上传，CC BY 4.0。32 页作者稿与最终 11 页期刊排版可能不同。
- 第 3.1 节式 (1)—(5)：四线 KVL、中性线电流消元、相对中性点电压；第 3.2 节式 (20)—(23)：相别相关灵敏度；第 4 节式 (24)—(25)：回归；第 5 节算法 1、式 (26)—(38)：逐时段 DOE、三类分配目标、运行约束；第 6.2 节说明未处理预测不确定性；第 6.5 节噪声和缺失。
- 等价整理：T_V Z^(4) B_I 以及复导数到 SP/SQ 的统一注入正号。
- 教学补充：四次独立扰动的秩充分例、端口 SP 矩阵、简化电压 LP 的对偶最优性、三类削减数字、联合点转独立盒的负跨相反例。均不是作者原实验。没有把原文约束扩写为已验证所有内部线路热容量。

## S11：RDOE；v4 全文及期刊元数据

- 作者：Bin Liu, Julio H. Braslavsky。
- IEEE Transactions on Power Systems 39(2), 3921—3936，2024-03；[DOI](https://doi.org/10.1109/TPWRS.2023.3308104)、[Crossref](https://api.crossref.org/works/10.1109/TPWRS.2023.3308104)。区别于后续同名一页会议记录。
- [arXiv 记录](https://arxiv.org/abs/2212.03976)、[v4 HTML](https://arxiv.org/html/2212.03976v4)、[v4 PDF](https://arxiv.org/pdf/2212.03976v4)。v4 为 2023-08-28，首版 2022-12；HTML 标 CC BY-SA 4.0。本文是独立教学分析，未复刻论文图像和长段措辞。
- 第 II 节式 (1)—(3)：线性可行域；III-B1 式 (7)—(11)：正对角椭球、内盒及方向惩罚；附录 A-A：最大内矩形；III-B2 算法 1、式 (12)：放松一行 1 的冗余 LP；III-B3 式 (13)：双线性包含扩张，原文明确无全局最优保证；III-C：比例公平与扩张 caveat；第 IV 节：2 节点、33 节点、132 节点案例及 AC 对比。
- 本轮 HTML/PDF 文本双核：式 (13a) 印为相邻基础上下界项相减，而统一 ±e_i 的 Ep≤f 表示中应以 u−l=f_odd+f_even 计算真实宽度；教材明确固定成对半空间记号，使用 u−l 的教学重写，未默默声称逐字原式正确。f 维度亦按行数统一。未因 OCR 换行而推断新的定理反例。
- 教学补充：顺序去冗余避免重复行同时删光；固定显式半空间的精确整盒支持函数；可直接执行的凸扩张变体；二维解析最优椭球与 TikZ 图；可行方向的一阶比例公平证明；线性证书/非凸 AC/模型集合接口区分。

## 范围声明

S5、S7—S11 的教材段落提供可复算的算法主线；S6 仍不具备论文方法级证据。教材伪代码使求解与异常出口具体化，但不等同于论文作者开源实现。所有教学例子使用自造小数据，不代表原文准确率、样本复杂度常数或部署性能。正文末尾的批评均与主体算法分离，便于先学习方法再检查适用条件。
