# 主动探测教材章节：来源与复核位置

整理日期：2026-09-11。沿用2026-09-10已读取的primary全文；本次重读P1完整ADMM/PGD部分、S13满秩引理及P4正式版附录A1。新稿主体为独立中文教学解释、等价推导和自制小例子，不复制原图或连续长段落。

|编号|实际阅读的primary全文|定位与本次用途|
|---|---|---|
|S12|https://arxiv.org/html/1803.04506 ; https://arxiv.org/pdf/1803.04506 ，v2，2018-06-10|II式(2)–(3)共同路径；III式(4)–(5)回归；IV层集合、Propositions 1–2、Algorithm 1；V约化树与Algorithm 2；VI误差与CLT近似。教材分别展开全测压交集递归和部分测压隐藏节点创建。|
|S13|https://arxiv.org/html/2004.02370 ; https://arxiv.org/pdf/2004.02370 ，v2，2020-04-07|II式(1)–(5)、Theorem 1；III-A Lemma 1及式(6)–(9)，III-B选列与求逆引理；IV式(17)–(21)、经验盒界、显式树编码；V实验。2026-09-11重读Lemma 1“任意候选树”的范围。|
|P1|https://arxiv.org/html/1802.06027 ; https://arxiv.org/pdf/1802.06027 ，v3，2019-02-20|III-A约化拉普拉斯与先验Γ；III-C式(22)凸估计；III-D式(24)–(28)四副本ADMM；IV-C已知电阻PGD与后处理。本次重新读取完整推导，按一致乘子下标展开全部更新。|
|P2|https://journals.pan.pl/Content/119954/art04.pdf|期刊pp.300–302第3节，分流电路与接收矩阵；pp.302–305第4节仿真与实验。220Hz、5A属于仿真；240Hz方波属于实验，未混同。|
|P3|https://journals.pan.pl/Content/121560/PDF/art04.pdf|pp.794–796第3–4节传输线、1mH上游电感、5–15MHz、时延/反射；4.3操作扫描；第5节修改IEEE13的仿真。教材记录组织没有越界为任意隐藏图算法。|
|P4|https://mdpi-res.com/d_attachment/energies/energies-17-02060/article_deploy/energies-17-02060.pdf|出版社15页正式版，2024-04-26。3.1/Table1四个分量，3.3存在/包含/传递，3.4纵向，3.5横向与方向不确定，3.6及附录Algorithm A1包含检测；第4节27设备与未定位节点。本次补核附录A1。|
|P5|https://arxiv.org/pdf/2305.19465 ，v2，2023-06-02|II式(1)–(5)相关压缩与周期化；III序列参数、Hankel/ERA与电路参数；IV IEEE13端口仿真、Table1负R1与虚拟电路说明。|

## 版本及公式核对

1. S12正式刊载IEEE Control Systems Letters 2(4):689–694 (2018)，DOI 10.1109/LCSYS.2018.2846801；P1正式刊载IEEE TCNS6(3):980–992 (2019)，DOI 10.1109/TCNS.2019.2901714；S13为2019 IEEE Milan PowerTech pp.1–6，DOI 10.1109/PTC.2019.8810900。正式元数据由Crossref API核实，bib的note保留实际读取的作者版本。
   - https://api.crossref.org/works/10.1109%2FLCSYS.2018.2846801
   - https://api.crossref.org/works/10.1109%2FTCNS.2019.2901714
   - https://api.crossref.org/works/10.1109%2FPTC.2019.8810900
2. S13的Theorem 1逆符号及C中单位阵维度在HTML/PDF文字层不一致，2026-09-10已将PDF第2、3页渲染核验。教材使用与电压方程和候选边维度一致的G^{-1}、I_m，并标为一致性修正。
3. P1原文局部副本更新中的乘子编号与最终对偶更新配对交叉。本教材固定U_j对应X=Z_j，由统一增广目标重新推导；这不等于已经检查或否定作者代码。
4. P4最先读取预印本 https://www.preprints.org/manuscript/202404.0181/v1/download ，后已成功下载mdpi-res正式PDF并视觉检查第8页；当前教材全部按正式版编号。web工具访问出版社HTML失败不代表未读终版，PowerShell获取终版成功。
5. P5的离散到连续转换选项没有在本次核实文本中提供足够细节，章节明确保留该复现缺项，没有指定某一软件命令为原算法。

## 数学整理和例子的来源声明

- 共同路径矩阵证明、S13秩证明的线代整理与det(C-D_b)恒等式、McCormick二值分情况、虚拟流连通性证明为教材推导。
- P1从增广目标推出Sylvester、特征值正根、行投影及PGD梯度；停止残差、回溯、符号明确的树后处理及重建对角为实现解释，按正文声明使用。
- P2的B方向固定及理想可达关系的传递约简、P3后代集合组织、P4纵向固定点/等长交叠反例为教学形式化。
- P5三序列相关、秩一ERA和RLC系数匹配均为独立教材例子；不是原实验数据。
- 复杂度为文中直接实现的上界/常规密集运算成本，未归于原作者已证明的最优复杂度。

本次没有把全文未公开的检测阈值、设备同步细节、普适反射消歧、缺失记录定向或连续模型转换补成虚构的“原文步骤”。

