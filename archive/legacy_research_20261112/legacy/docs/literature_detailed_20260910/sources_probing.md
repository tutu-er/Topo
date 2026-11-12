# 主动探测文献核验记录

访问日期：2026-09-10。正文中文解释、反例、误差推导均重新组织；没有复制原图或长段落。

|编号|实际核验的 primary source|核验位置与使用范围|
|---|---|---|
|S12|https://arxiv.org/html/1803.04506 （v2, 2018-06-10）|II 式(2)-(3)；III 式(4)-(5)与满行秩；IV Assumption 1、Propositions 1-2、Algorithm 1；V Definitions、Proposition 3、Algorithm 2；VI 式(11)-(19)及CLT/union bound；VII实验。|
|S13|https://arxiv.org/html/2004.02370 和 https://arxiv.org/pdf/2004.02370 （v2, 2020-04-07）|II 式(1)-(5)、Theorem 1；III 式(6)-(16)及两条Lemma；IV 式(17)-(21)、盒界经验性说明；V IEEE13实验、alpha及90%表述。|
|P1|https://arxiv.org/html/1802.06027 （v3, 2019-02-20）|III-A Laplacian模型、III-B Theorem 1、III-C 式(22)、III-D ADMM、IV拓扑状态检测；全节点电压与启发式树投影条件。|
|P2|https://journals.pan.pl/Content/119954/art04.pdf|期刊pp.300-302 §3，等效分流式(1)-(4)及接收矩阵；pp.302-305 §4：220Hz 5A仿真、240Hz方波实验与空载支路设置。|
|P3|https://journals.pan.pl/Content/121560/PDF/art04.pdf|pp.794-796 §3-4：分布参数模型、5-15MHz、1mH电感、延时/反射、固定RL负载近似；§5 IEEE13改造仿真。|
|P4|https://mdpi-res.com/d_attachment/energies/energies-17-02060/article_deploy/energies-17-02060.pdf|出版社15页终版，Published 26 April 2024；§2设备与数据链；§3.3-3.6 自记录、包容/传递、垂直/水平补全，§3.5方向不确定说明（第8页已渲染视觉核验）；§4现场27 TID与无法定位情况。先读作者预印本 https://www.preprints.org/manuscript/202404.0181/v1/download，再核对终版所用结论。|
|P5|https://arxiv.org/pdf/2305.19465 （v2, 2023-06-02）|§II 式(1)-(5)相关/周期化响应；§III PRBPT参数、Hankel/ERA与等效电路；§IV IEEE13相A单端口仿真、Table1负R1及fictitious模型说明。|

## 访问与公式边界

- P4 web工具读取期刊HTML/PDF时失败，后通过PowerShell下载出版社mdpi-res终版成功，已核查终版第3.5节、第8页原始图像及第4节。终版章节编号较预印本改变，所用水平补全方向边界仍保留。
- S13 HTML与PDF文字层均在Theorem 1显示V=G I_C Delta，但正文式(3)-(4)为G^{-1}；报告采用后者并说明内在一致性。式(14)后C^{-1}中的单位阵维数按候选边数Lbar修正；该纠正是维度推导，不视为作者额外定理。
- web PDF screenshot功能不可用，改用临时下载与Poppler渲染，已视觉核验S13第2、3页的定理1逆符号与C的单位阵维度。任意定向关联矩阵的F=-A^{-1}二值性不作无条件推广。
- S12的概率讨论使用CLT近似而非任意噪声非渐近保证；报告另行推导确定性r_min/4间隔条件与高斯假设下的union bound。
- P2的“下游不可见”由小量分流近似得到，不能写为严格零；实验220Hz和240Hz不可混同。
- P3的方向性依赖外加电感；负载固定RL跨频率外推属于建模假设，5MHz波速/线长并非不需校准的普适常数。
- P5单端口等效模型不等于内部树结构；负R1是作者明确承认的拟合等效参数，不能解释为物理线路负电阻。

## 原创解释的归属

正文标“本文推导/辨析”的内容包括：主动实验的秩与条件数分离；terminal响应的电阻距离与degree-2细分不辨识反例；Schur补与树约化区分；二值乘积精确线性化证明；边数加无孤点不保连通反例；测量自变量噪声；信号检测与祖先偏序的传递约简；FSR水平补全方向反例；传输线相位模糊与反射定位；动态端口模型和内部拓扑的非唯一性。其作用是解释条件与边界，不冒充原作者新结论。

## 正式发表元数据补核

2026-09-10通过Crossref官方API核实：
- S12：IEEE Control Systems Letters 2(4):689-694, October 2018, DOI 10.1109/LCSYS.2018.2846801。API：https://api.crossref.org/works/10.1109%2FLCSYS.2018.2846801
- P1：IEEE Transactions on Control of Network Systems 6(3):980-992, September 2019, DOI 10.1109/TCNS.2019.2901714。API：https://api.crossref.org/works/10.1109%2FTCNS.2019.2901714
- S13：2019 IEEE Milan PowerTech, pp.1-6, June 2019, DOI 10.1109/PTC.2019.8810900。API：https://api.crossref.org/works/10.1109%2FPTC.2019.8810900

参考文献使用正式发表信息；note保留实际核验的arXiv版本与修订日期，避免把上传日期当发表日期。