# EV 局部拓扑辨识：文献与创新性核查（v7）

核查批次：2026-10-11 开始的 v7 研究。范围限于既有主线：历史相容参数域、共同源空间对比、T 型区分设计、可执行 EV 约束、观测后完整联合重拟合。本文仅记录已核实来源及方法边界，不新增实验，不声称穷尽检索。

## 1. 核心判断

最有希望形成连贯成果的是：**利用源响应的空间结构，保留历史相容的线路参数集合，针对当前尚不能排除的局部拓扑设计可执行 EV 实验，再用原始数据进行完整联合重拟合，检验设计收益是否保留。**

以下组成部分本身均不能作为新颖性主张：主动电网 probing、EV/wallbox 电流阶跃辅助拓扑识别、竞争模型参数重拟合、T-optimal 设计、sequential 设计、集合相容性、消去已知干扰子空间，以及固定均值下端点输入的二阶矩最优性。将这些术语拼接并不足以构成研究贡献。

当前可检验的贡献候选是：**在明确的局部拓扑类和源空间结构下，指出逐输出误差盒或历史参数重置如何消除本来存在的结构信息，给出保留信息的条件；证明并验证有设备约束的动作选择在完整竞争重拟合后仍改善局部区分。** 这仍是本项目的研究定位，尚不能由有限文献检索确认其首创性。三末端机制证明有价值，但远不足以支撑一般配网或未知源位置下的算法新颖性。

## 2. 八项最相关原始研究与阅读边界

下表的“正文”只表示实际阅读列出的部分；不表示整篇逐页读完。除已存在本地的 Bhela/Scott 等文件外，本轮采用在线开放正文，不新增永久 PDF 归档。未使用 HKU 订阅：核心论证所需开放来源已足够；Hotz 浏览器预览可读，而下载工具挂起后被中断，不能宣称新增 PDF 已下载或归档。

| 编号 | 确切书目与来源 | 本轮实际阅读范围 | 已知结果与本项目边界 |
|---|---|---|---|
| R1 | Holger Dette, Stefanie Titoff. **Optimal discrimination designs**. *Annals of Statistics* 37(4):2056–2082, 2009. DOI [10.1214/08-AOS635](https://doi.org/10.1214/08-AOS635). [arXiv:0908.1912v1](https://arxiv.org/pdf/0908.1912). | 正文 §2，式(2.4)–(2.7)、Theorem2.1、Remark2.4；PDF pp.4–5、9。 | 固定参考响应，竞争模型参数随设计重新拟合；不要求模型嵌套。统计功效解释另外依赖局部备择、噪声等条件。§2已明确提到 sequential/Bayesian/maximin 处理参考参数不确定性，因此不能把使用历史参考称为新理论。 |
| R2 | David Mogalle, Philipp Seufert, Jan Schwientek, Michael Bortz, Karl-Heinz Küfer. **Computing T-optimal designs via nested semi-infinite programming and twofold adaptive discretization**. *Computational Statistics* 39:2451–2478, 2024；online 20 June2023. DOI [10.1007/s00180-023-01370-4](https://doi.org/10.1007/s00180-023-01370-4). | 出版社开放 HTML 正文 §2.2 Definition1，§5.2.2，多响应例子，§6及 AppendixA。 | 已有双层重拟合、参数域约束、多响应设计与离散设计空间算法。§2.2还指出已有数据可以估计参考模型。故“多输出+受限参数+T-opt”不是创新；当前小问题没有必要照搬双重自适应离散化。注意本文收敛/等价定理的条件不可直接用于所有重拟合不唯一的本项目情形。 |
| R3 | Guido Cavraro, Vassilis Kekatos. **Graph Algorithms for Topology Identification using Power Grid Probing**. *IEEE Control Systems Letters*, 2018. [arXiv:1803.04506v2](https://arxiv.org/pdf/1803.04506). 本轮未独立核对正式版 DOI。 | 作者稿正文 §II–III式(2)–(5)、§IV–V Assumptions1–2及reduced grid、§VI式(11)–(18)；PDF pp.2–5。 | noiseless 模型利用秒级扰动期间非探测负荷近似不变；全部叶节点激励结合全节点电压足以恢复，局部电压一般仅恢复 reduced topology。噪声部分不是“所有负荷永远不变”，而另设独立零均值误差与协方差模型。因此我们应比较确定性空间结构假设与这些统计假设，而非把既有工作描述成完全不能处理负荷变化。 |
| R4 | Sina Taheri, Vassilis Kekatos, Guido Cavraro. **An MILP Approach for Distribution Grid Topology Identification using Inverter Probing**. [arXiv:2004.02370](https://arxiv.org/pdf/2004.02370), 2020作者稿。正式发表元数据本轮未独立核对。 | 正文 §II-B式(3)–(5)，§III开头与线路参数最小二乘拟合；PDF pp.2–3。 | 给定候选拓扑先拟合线路电阻，再选择最佳拟合拓扑，随后用MILP规避逐拓扑枚举。秒级负荷变化最初忽略，随后并入误差E。因此“错误拓扑必须重新拟合参数”是已有基本做法；本项目应突出历史与未来共享参数及结构化源约束。 |
| R5 | Siddharth Bhela, Vassilis Kekatos, Sriharsha Veeramachaneni. **Smart Inverter Grid Probing for Learning Loads: Part II—Probing Injection Design**. *IEEE Transactions on Power Systems*, 2019. DOI [10.1109/TPWRS.2019.2906306](https://doi.org/10.1109/TPWRS.2019.2906306). [arXiv:1806.08836v2](https://arxiv.org/pdf/1806.08836). | 正文 §I–II-A/B，式(1)–(8)；PDF pp.1–3；已有本地清单另记录§II-C/III历史阅读，本轮不据此扩大已读范围。 | 已将设备可执行注入库、未知负荷范围、网络安全筛选、增加状态多样性结合。其目标主要是已知图下的未知负荷/状态估计，不能直接等同于拓扑竞争区分。电池SoC因秒级实验被忽略；6小时EV实验必须补上电量、SoC、接入及切换约束。 |
| R6 | Christian Hotz, Sergej Baum, Eberhard Waffenschmidt, Ingo Stadler. **Topology Estimation in Low Voltage Grids Using Wallbox Charging Data Recordings**. CIRED Porto, 2–3 June2022. [Zenodo公开作者资料](https://zenodo.org/records/10046340), DOI [10.5281/zenodo.10046340](https://doi.org/10.5281/zenodo.10046340). | 浏览器公开PDF预览实际读到pp.1–2：Introduction、Simulation Environment、Identification of switch events、式(1)–(2)。共9页；其余页面本轮未读。仓储题名末词为“Records”，PDF首页为“Recordings”。 | 明确利用EV开始/停止充电导致的电流阶跃与各测量节点电压变化进行拓扑估计，研究基于含波动家庭负荷的仿真环境。故“EV可用于拓扑辨识”已有直接先例；本轮已读部分不足以断言其是否覆盖移动选点、完整历史联合重拟合或所有扰动条件。此前主报告列出的10.1049/icp.2022.0823本轮未由正式版复核，建议以确认的仓储DOI作为当前证据入口。 |
| R7 | Nicholas Piaquadio, N. Eva Wu, Morteza Sarailoo, Jianzhuang Huang. **Pulse Compression Probing for Tracking Distribution Feeder Models**. [arXiv:2305.19465v2](https://arxiv.org/pdf/2305.19465), 2 June2023作者稿。作者由本轮 root 核对 arXiv 摘要页；正式发表DOI本轮未独立核对。 | 正文 §I–II，式(1)–(4)、§IV末尾与TableI讨论；PDF pp.1–2、5。 | 二值伪随机脉冲与相关处理用于提取馈线等效动态响应，时间尺度是电气周波/动态系统记忆长度。它为“有目的激励”提供邻近研究，但不构成15分钟EV槽的ABBA抗漂移保证，也不能由二值波形相似推导本项目最优性。 |
| R8 | Vito Cerone, Jean-Bernard Lasserre, Dario Piga, Diego Regruto. **A Unified Framework for Solving a General Class of Conditional and Robust Set-Membership Estimation Problems**. *IEEE Transactions on Automatic Control* 59(11):2897–2909, 2014. DOI [10.1109/TAC.2014.2351695](https://doi.org/10.1109/TAC.2014.2351695). [arXiv:1408.0532v1开放正文](https://arxiv.org/html/1408.0532v1). | 在线正文§II与§III框架；聚焦可行参数集、条件估计/鲁棒估计问题。 | 这是集合估计统一框架，不是我们当前主动局部辨识的直接算法。相容性使用“存在合法解释”；鲁棒点估计使用最坏误差评价，量词不同。不能把它改名为guaranteed discrimination，也无须为了引用它额外引入对抗性源假设。 |

补充基础与近年核查（不作为新主线）：

- **端点设计基础**：Steftcho P. Dokov, David P. Morton, *Higher-Order Upper Bounds on the Expectation of a Convex Function*, January2002作者研究稿；[机构开放全文](https://edoc.hu-berlin.de/server/api/core/bitstreams/7814d9ad-657d-41fd-8b38-49ac2744b298/content)。本轮读首页和§2前三步推导（PDFpp.1–3）。其回顾的一维Edmundson–Madansky弦上界正是固定均值端点分布最大化凸损失的基础。此处不是采用鲁棒分布优化；同一个初等凸性不等式也适用于确定性的经验设计。未核对原始1956/1957 Edmundson报告或Madansky1959正式全文，不把它们标为已读。
- **sequential近年研究**：Mingyao Ai, Holger Dette, Zhengfu Liu, Jun Yu, *A New Approach to Optimal Design under Model Uncertainty Motivated by Multi-Armed Bandits*, *JASA*120(552):2709–2722, 2025，DOI[10.1080/01621459.2025.2487223](https://doi.org/10.1080/01621459.2025.2487223)。本轮得到出版社搜索返回的摘要及引言片段；直接全文打开失败，**不算阅读全文或已核实定理**。只能据摘要记录其研究sequential模型区分与参数估计平衡；不迁移它的统计保证。
- **集合分离邻近研究**：Joseph K. Scott, Rolf Findeisen, Richard D. Braatz, Davide M. Raimondo, *Input design for guaranteed fault diagnosis using zonotopes*, *Automatica*50:1580–1589,2014，DOI[10.1016/j.automatica.2014.03.016](https://doi.org/10.1016/j.automatica.2014.03.016)，[作者机构PDF](https://web.mit.edu/braatzgroup/input_design_for_guaranteed_fault_diagnosis_using_zonotopes.pdf)。现有本地清单记录§2、§3.2、§4，Definition1与Theorem3；本子任务未重新阅读PDF正文，root已独立复核，故最终报告阅读声明应依据root本轮证据。该工作表明有界扰动下的可达输出集分离早已有系统研究，不能把“保证区分”作为泛化新意。

## 3. 端点命题的准确定位

令单节点EV输入u属于[0,U]，均值固定为μ；参考与竞争模型之差为d0(θ)+u d1(θ)，各时点使用相同度量，背景恒定，参数可行域不随u变化。对每个θ，平方损失fθ(u)=||d0(θ)+u d1(θ)||²是凸函数，因此

fθ(u) ≤ (1-u/U)fθ(0)+(u/U)fθ(U).

对任意均值μ的设计ξ积分，再对θ取最小值，即得端点设计ξend=(1-μ/U)δ0+(μ/U)δU的T评分不低于ξ。关键是**先得到对每个竞争参数都成立的同一端点上界，再取最小值**，不是声称参数重拟合后的函数仍然凸。有限N槽且端点比例可实现时得到精确端点方案；否则对平方损失用二阶矩论证得到至多一个中间功率槽。

这是经典凸性结论在本问题的直接推论，应称简化命题/实施原则。它不能保证严格改进、不能处理任意时变背景、不能消除最小启停时间或设备爬坡限制，也不为ABBA特定顺序赋予独特优势。在恒定背景、静态响应、固定度量且无时序约束时，同样功率计数的排列完全等价。若算例宣称顺序改善，必须能指出究竟哪个模型项使其不再置换不变。

## 4. 推荐的论文论证顺序

1. **问题与反例**：被动历史下存在多个参数可重拟合的局部拓扑；若源在各输出独立变化，目标可能不辨识。这是核心障碍，先给最小反例。
2. **物理信息与假设**：源的空间方向为何共同、幅值如何允许随时间变化、哪些P/Q实际测量、哪些只是计划预测、参数跨历史与实验为何共享。已知共同方向比任意未知源位置更强，应明确代价。
3. **已有方法与迁移**：Cavraro/Taheri给主动观测与候选参数重拟合；Dette给面向模型差异而非全参数估计的设计；Bhela给设备与网络可行注入的组织方式；Cerone/Scott只为相容性和集合分离提供背景。
4. **最小新方法**：历史相容参数域→当前候选拓扑→共同源对比用于动作评分→有限可执行动作库→原始数据联合重拟合。T评分只用于挑实验，不是拓扑概率或排除证书。
5. **证明**：历史集合与联合拟合的等价性；对比消源成立范围及信息损失；端点简化命题；具体局部拓扑类中正分离及噪声扰动界。分别标出通用初等性质、既有理论迁移和本项目机制推导。
6. **验证**：控制电量、时长、真实模型、参数域与误差预算；比较常充、随机可行曲线、端点曲线、选点/选时；让所有竞争模型同样充分重拟合。用未参与动作评分的未来轨迹评估执行后结果；参考错设和源方向错设必须有负例。
7. **边界与可扩展性**：单一源方向、三候选线性模型是机制证据；未知源位置、多相AC与全WZZT候选生成属于后续，不在没有验证时并入贡献标题。

## 5. 最值得检验的创新问题与否证标准

| 研究问题 | 支撑它所需证据 | 会推翻/削弱它的结果 |
|---|---|---|
| 共同源结构保留被误差盒丢失的辨识信息 | 同一物理误差幅值、同一观测和竞争参数域下的集合包含与正分离 | 稍许允许源位置变化便全部重合；收益只来自人为过小源域 |
| 历史联合重拟合使节点选择更合理 | 真参数在域内；历史充分收缩却不泄漏未来；留出未来完整refit | 一旦允许合理历史模型误差即选点排序失效；选点只追踪参考偏差 |
| 移动EV的地点/时段选择有意义 | 与同电量固定点及随机可行点比较，并计入可接入性 | 全节点完全对称时仍报告显著地点收益；静态恒背景中报告独特时序收益 |
| T评分能预测执行后改善 | 多种参考与留出未来轨迹上的排序/排除余量关系 | 高T但完整源模型低余量，或选择不如简单端点/固定点策略 |
| 方法比复杂现有体系更适用 | 小问题凸内层、动作有限枚举、可复核上下界与计算成本 | 需要大量未经依据的分布/鲁棒假设才产生非平凡结果 |

**稿件建议的中心问题**：在明确的结构化源变化模型下，能否以有限、可执行、等电量的EV实验，缩小历史数据留下的局部拓扑相容集合？围绕这个问题组织论证，比同时声称新EV设备、新WZZT、新置信度、新鲁棒设计更可审查。

## 6. 检索覆盖与未完成项

本轮查询了T-optimal sequential/prior/constrained design、distribution topology probing with varying/unobserved loads、wallbox/EV topology、2024–2025相关后续、端点凸性背景。额外发现2024综述及2025被动拓扑/神经网络论文，仅用于判断检索邻域，未作为方法事实依据；没有把“更近但任务不同”论文塞入核心比较。未对专利、所有数据库、2026全部新稿进行穷尽审查；未知源位置下是否存在直接重合工作仍需后续定向核查。

本子任务对现有算法、实验结果和主报告没有写操作；只新增本文件。不执行git。核心参考来源优先本地清单与开放作者/出版社正文。参考阅读范围不能由下载成功、搜索摘要或论文题名自动推定。
