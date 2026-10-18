# 标准馈线与论文风格算例的适配审查

日期：2026-10-18。审查对象为 canonical 源码 C:/Users/23761/.codex/worktrees/36a1/Topo，原 full_pipeline_v1 冻结批次保持不动。本次只读代码与已有说明，未运行求解器、测试或新科学实验。

## 结论

现有完整 AC 预测、导数、R/X 拟合和 G-SQ 目标可以复用。真实线路表不能仅替换 cases.asset_parameters：量测位置、公共源接口、单位、边界、表计精度及 EV 接入映射必须一起适配。原 case33bw 有中间节点负荷，现有自动候选生成假设量测全部在叶端；二者有实质区别。

建议最快的第一层主实验为“严格终端化后的标准 case33bw + 现有 paper-style LV”。必须明示新增服务边和其物理假设；原 case33bw 的完整 AC 对照保留原中间负荷，不能把终端化称为原网络原样复现。第二层再扩到可保留中间量测的候选模型。若一个标准 LV 数据集本就满足末端量测条件，优先采用它，避免人为迁移负荷。

## 1. 查到的算例来源范围

- rnj_wzzt_core/rnj_wzzt/data/paper_style_terminal_lv.py：22 节点、15 量测终端；名称含 paper-style，说明为 Soumalas/Park/Flynn 风格，线路阻抗乘 5，且为增强 X 灵敏度特意将 R/X 调为相近量级。它是项目自建合成算例，不能称原论文的精确复现。
- paper_style_case_bank.py：soumalas11、flynn16、pengwah18 也是 inspired/style；阻抗倍数分别 5.2、4.8、4.5。保留原版本作历史连续性对照，可另设除去该倍数的 declared-unscaled 变体，但后者同样不是作者原始线路表。
- 请求中的 rnj_wzzt_core/rnj_wzzt/data/case33bw_raw.py 不存在。现有真实入口是 terminal_case33/data/case33bw_raw.py。它含 Baran-Wu/MATPOWER case33bw 的 33 母线、32 闭合支路、5 可选常开联络线、12.66 kV、10 MVA、kW/kvar/ohm。
- terminal_case33/scenario/terminalize.py 已提供 hybrid_leaf 与 all_loads_to_new_terminal_leaves。hybrid_leaf 保留原物理叶端负荷；将沿线负荷转移到新增服务叶节点，保留总 P/Q，但改变损耗、端电压及量测位置。新增服务线默认 scaled_original = 相邻原支路中位阻抗的 0.2 倍；这是工程化合成规则，不是实测服务线路数据。
- aggregate_to_original_leaves 把中间负荷重新分配到原叶端，改变空间注入；不建议用于“完整标准馈线”主结论。
- 新论文/标准数据的原文核对由文献与数据工作流另行完成；此审查只证明本地代码标注范围。

## 2. 当前物理表示的限制

likelihood.RadialCandidate 实际重用 mobile_ev_probing/ac_general_v9.py 的 RadialCandidate：

1. from_supports() 强制包含 full-support 边以及每个量测标签的 singleton 边。
2. 所有量测 P/Q 注入在 singleton 的叶端；其他隐藏节点零注入。
3. full-support 边 index 0 的下游节点固定是源接口 node 1。likelihood.ac_predict() 的 loads[:,0] 加 source_p + i source_q；源 P/Q 全时变化、位置不变。
4. 单元中相同 descendant support 的纯零注入单链被聚合；聚合阻抗为逐段 R/X 相加，纯零注入单链这样做对串联 AC 等效成立。
5. 论文风格根有多个出线，原网络没有公共上游边。强制增加正 full-support 边会无中生有添加阻抗。可选办法：
   - 引入有来源参数的真实公共变压器/上游线路，并明确原模型因此改造；源置其下游。
   - full-support 为严格零阻抗虚拟接口，源等效在已知电压根处。此时源对端电压无影响，仅影响总表；适合作为零公共路径的对照，不能声称该源激励了所有分支。
   - 不允许把“很小的正数边”默认为有物理依据的公共干线。
6. 原 case33bw 的共同首边 1→2 实际存在；source=node2 有清楚物理含义。沿线 bus2…32 的负荷仍是内部量测，现有 leaf-only proposal 不表达它们。

ac_predict() 与 _response_derivatives() 在算术上使用 candidate.B 和 candidate.terminal_edges。若单独构造 measured-node candidate，terminal_edges 可以索引内部测点，loads、V 输出和精确导数仍适用；但 from_supports() 自动构造及 WZZT/RNJ 的叶端候选不自动支持这种模型。直接让 simulator 用原内部负荷、selector 用叶端假设，会形成未声明的结构失配。

## 3. 单位与参数域：原值不能照搬

物理统一规则：

Z_base[ohm] = V_LL_base[kV]^2 / S_3phi_base[MVA]，
p_pu = P[kW]/(1000 S_base[MVA])，
q_pu = Q[kvar]/(1000 S_base[MVA])，
r_pu = R[ohm]/Z_base，x_pu = X[ohm]/Z_base。

这是平衡三相等值；V 的线/相制与三相功率必须配套。

|对象|旧全流程|paper15 原参数|case33bw 原参数|
|---|---:|---:|---:|
|额定电压 kV|0.4|0.4|12.66|
|功率基准 kW|20|200|10000|
|Z_base ohm|8|0.8|16.02756|

基准选择本身可改变，但所有功率、阻抗、sigma、偏差和源/EV 都要同步换算，物理预测应保持不变。

现有 cases.fit_options() 写死：
R∈[.206*.005/8,.641*.25/8]pu，X∈[.080*.005/8,.090*.25/8]pu，
X/R∈[.080/.641,.090/.206]≈[0.1248,0.4369]。
这个域排除 paper-style 约0.69–0.81 的真实 X/R，也排除 case33 多条高 X/R 线路。paper15 首段在其原基准下 R≈0.2415pu、X≈0.1955pu，而旧 R/X 上界约0.0200/0.00281pu，差异不是微调。

新的域必须来自公开资产/给定标准线路表所声明的**可用先验**，而非读取待辨识真实拓扑后逐边±若干百分比。建议区分：
- topology-only/known-asset benchmark：公开线路资产表已知，但支路连接未知；各候选按可用资产约束匹配，明确这是额外信息。
- unknown-line benchmark：由标准馈线事先公布的线路类型/全局工程范围给所有候选相同宽域，保留真实值越界审计。不能利用评测真边与候选边的一一对应。
- 零虚拟根接口需候选依赖的 array bounds，严格固定0；不能在所有边统一正下界下拟合。
- 最小线路阻抗是情景资产假设，不是全配电网普适正下界。

ev_rx_fit._RxMap 支持长度 m 的数组边界，但底层 lk._bounds 要求每个box坐标lower<upper；不可直接指定(0,0)固定边。它只能在box/cone交点自动去掉某些退化坐标。现有 refit.fit_history() 将一套 options 给不同 m 候选，需一个 candidate_options(candidate, public_model) 调度。全局统一0下界、正上界可直接复用调用，不利用评测真stub位置来固定候选边。

## 4. 源、Q 与表计

- 现有 AC 源同时估 P 和 Q：fit_rx_inputs() 的共享线路2m参数，源逐时2H参数，V + root master P/Q 残差；P/Q 表读数去已校准偏差后作为输入，未传播输入误差。这是条件输出似然。
- 默认初值 source≈master−sum terminal，包含暂未扣除线路损耗；拟合会修正。源不是单独由 WZZT 估计的。
- 现有源界±.5pu按旧20kW等于±10kW；若原样换10MVA基准即±5MW，不再表示同一设备能力。应由公开源容量和功率因数/逆变器能力换算，不按基准照抄。
- 现有 MeterModel 来自 simulation.model()，p_sigma=.0006、q_sigma=.0004、master sigma=.001pu、v_sigma=.00012pu。20kW下端P sigma=12W，总P=20W；10MVA下变成6kW/10kW，足以吞没单辆7kW EV。必须先在kW/kvar/额定精度中定义再换标幺，不能把标幺数值视为跨馈线同精度。
- 现有固定输入路径拒绝独立 source telemetry。如果新增标准资料恰有源表计，应单独扩展而非悄悄不计其似然。
- 现有 EV Q=0，含义是单位功率因数充放电。若 EV 给定非单位PF，需要 q_EV=k*u 或逆变器P/Q能力约束，且在 actual、forecast和导数中同时使用。普通负荷Q与新能源Q也必须在模拟和估计中配套。
- 多个分布式新能源若已在各终端总P/Q内，作为实测净注入输入即可；额外“未计量共同源”只在一个固定物理接口。把多个未测内部分布源压成一个公共源可能失配，宜先作单源主场景，再作明确失配对照。

## 5. EV位置映射与可执行性

现有 study.select() 令 n_sites=D.n，默认所有量测终端都可接入；initial_site=0，所有travel=1 slot，setup=1 slot；每轮可选最多1次移动。真实馈线需要 site→bus→terminal_column 映射和公开接入可用性，不能把每个中压聚合负荷视为可停靠插座。

建议最小接口：

- site_to_observation: one-hot矩阵 E，shape(n_observed,n_sites)；
- expand_ev(u_site) = u_site @ E.T，输出(time,n_observed) kW；
- g_site = g_all @ E，导数与实际注入使用同一映射；
- travel_slots[i,j]、travel_energy_kwh[i,j]、setup_slots[j]、access[t,j] 由公开场景规则给出；地点距离在拓扑真值之外给出，不作为推断信号；
- EV 模板以真实 kW/kWh 定义，目标同服务任务，移动不强制发生。

原 case33bw 单负荷母线通常几十至几百kW；5.7kW单车影响可能小。这是有效的失败场景，不能为了让G-SQ获胜按负荷规模任意放大EV。可单独注册“多个EV聚合”或更大移动储能对照并给出车辆数、容量、功率和移动成本。

## 6. 最简新 adapter 的明确接口

建议放新 reference_pipeline/ 下，不改旧 full_pipeline_v1：

\`\`\`python
@dataclass(frozen=True)
class ReferenceNetwork:
    case_id: str
    buses: DataFrame        # bus_id, nominal_P_kw, nominal_Q_kvar, observed
    branches: DataFrame     # from,to,R_ohm,X_ohm,closed
    root_bus: int
    base_kv: float
    base_kw: float
    source_bus: int
    observation_bus_ids: tuple[int, ...]
    site_bus_ids: tuple[int, ...]
    provenance: dict       # original vs adapted fields, source/hash/license

def load_reference(case_id, variant) -> ReferenceNetwork: ...
def terminalize_reference(network, service_rule) -> tuple[ReferenceNetwork, dict]: ...
def encode_candidate(network, observation_bus_ids) -> tuple[RadialCandidate, ndarray, ndarray, dict]: ...
def public_fit_options(candidate, public_assumptions) -> dict: ...
def measurement_model(network, physical_precision, H, T) -> tuple[MeasurementModel, MeasurementModel]: ...
def observed_planning_inputs(D, dm, fm, timestamps, site_map, public_assumptions) -> dict: ...
def expand_ev(site_power_kw, site_map) -> ndarray: ...
\`\`\`

load_reference 返回公开网络资产信息和适配元信息，**不能把真实candidate/RX/未来序列传入planner**。模拟器私有接口 make_case/actual_observation 持有truth；公共fit_options仅从公开声明构造。

若沿用 leaf-only 方法，encode_candidate 必须拒绝任何内部负荷/内部量测，先显式 terminalize。根据根定向得到每条边所有下游量测标签作为support，按support集合聚合相同support的串联纯零注入边，R/X逐段相加；保存原边→约化边、原bus→量测列的双向映射，孤立无量测支路不能无声删除。对独立输出矩阵B检查路径等价。避免用算法生成真candidate后直接加入历史bank。

保留内部量测的后续接口需要额外 encode_measured_node_candidate()，不能称当前 from_supports() 已实现该功能。

## 7. 单初值、WZZT预算的开发阶段修复

### AC拟合

旧 config 是 multistart=1, max_nfev=250。G-SQ 需要 all-candidate historical complete，任一失败会整族退回P-SQ，因此先改开发计算规则，比直接比较新目标更优先：

1. 保持数据模型、目标和域一致，用历史/公开先验构造多初值：投影后的线性R/X初值、域中点、固定种子2个扰动。线性电压平方drop的一阶系数是2r/2x，不能直接把被动系数当r/x。
2. 初值用同一历史，若阻抗偏大导致初始AC不收敛，可从较低合法阻抗启动/延续；不能从评测真R/X启动。
3. 开发固定候选预算梯级如2初值×500nfev；记录每次状态、最优可行SSE、站立性、活动边界、物理误差和时间。若再加应急预算，预先给所有方法相同规则。
4. 联合拟合保留 fixed-history-line 的可行源拟合作证，不得让更差收敛解覆盖更好可行解。保持 failed最优与converged证据分开。
5. 数据规模增加后，当前 trf/exact dense 求解和 _physical_information() 全Jacobian/SVD会成为瓶颈。可开发 sparse TRF/LSMR或块源消元，但需在小例对照目标/梯度/结果，属于另一个明确算法改动，先计时再决定。
6. 保存所有候选失败，不通过丢弃困难竞争者取得G目标；不可用与拓扑错误分开。

### WZZT

passive_bridge→fit_source_aware_wzzt 默认 HiGHS，旧0.5秒是每个solve时间而非总墙钟。该实现仅接受证明最优的 initial/refit/extension，超时extension会停在已接受star，所以“原生WZZT无非平凡树”可能是预算问题。

开发预注册5/15/30秒每solve ×全调用30/60秒上限，比较：
- 是否进入非平凡已接受path；
- solver status/incumbent/dual bound/gap；
- 强候选覆盖、历史线性残差、耗时；
- 超时未接受incumbent若提取为额外结构，只作为replaceable proposal，不能称原贪心gain已证明。

不要求随更大网络用无限预算求全局WZZT。可并行RNJ原始/中心化/多tol、少量块bootstrap，全部只读当前历史；候选上限用预注册分层保留，防止一种提案挤掉另一种。每次统计generated/pool/capped/post-union coverage，但true coverage仅用于开发诊断和最终评价，不能用于选bank。

source_inputs 目前固定None，且bridge显式拒绝 nominal source P<0。新增PV按负消耗不能直接喂给此接口；先保持空源并标注proposal失配，或开发符号正确的观测端名义源adapter。都不能向WZZT传模拟真源。WZZT的源输入和AC固定接口源模型是两套不同组件，不可混称已联合估计。

## 8. 开发验收与冻结前阻塞

|验收项|通过条件|
|---|---|
|线路表原样性|每个原始R/X/闭合状态逐项匹配；所有改造/倍率/新边单列|
|标幺不变性|两个合理Sbase产生相同物理V、总P/Q、损耗与每kW梯度；只读标准参数不能满足此项，需实际开发测试|
|树编码|原与约化AC对若干合法P/Q/源场景一致；纯零注入单链可聚合，内部负荷不得被删除|
|源位置|source_bus映射唯一；P/Q同时影响同一物理节点；root-star零接口效应正确|
|EV映射|每个可接点一小步kW扰动，导数/actual注入/观测列一致；不可接点禁止|
|计量预算|同一物理sigma/采样预算；不随任意Sbase变化|
|预测边界|开发/校准/评价按日期及家庭/馈线分组；所有归一化只读历史；新能源Q/PF规则有来源并标合成|
|候选覆盖|真值不插bank；初/后完整覆盖及裁剪损失分别保留|
|AC稳定性|多初值和应急规则固定；可行/收敛/失败分别记；联合SSE不高于合法固定线见证|
|WZZT预算|预算只在开发选择；明确per-solve与whole-call，两种超时均保留|
|目标对照|P-SQ/G-SQ/G-AQ同域、同任务、同计量、同求解预算；记录上下界/未闭gap|
|统计|校准样本不足输出未决；评价不调域/预测/阈值/故障回退|

冻结前的主要阻塞按先后为：
1. 选定标准LV/原论文真实参数来源；现有paper-style不能承担“真实论文复现”标签。
2. 决定主场景保留内部负荷还是显式终端化；不能混用。
3. 指定公共源接口，特别是多出线根没有天然共同正阻抗边。
4. 改单位/表计/源/候选参数域，验证真值位于**事先声明**域。
5. 开发确认历史候选覆盖与多初值计算可用性后才冻结正式比较。

本报告的代码事实已直接核验。建议接口和验收尚未实现或数值验证；没有把其表述为当前支持功能。

## 9. 随后采用的精确改造（更新）

已实现 reference_benchmark/representation.py。它优先保留原内部负荷，不使用正阻抗服务线迁移：每个内部测点添加精确0阻抗的虚拟量测叶。设物理节点电压为 V，虚拟叶电压为 V'，支路阻抗 z=0；由 V'=V−zI=V，其恒功率电流仍为 conj(S/V)。虚拟支路无损耗，母节点KCL与原节点注入完全相同，故根复功率和所有量测电压不变。

对无任何中间注入的串联链，KCL给出相同电流 I，每段电压差 z_e I 相加为 (Σz_e)I，损耗为 |I|²Σz_e。因此同support边聚合 R/X 相加是完整AC等价；不能跨过未计量源接口聚合。实现将固定源接口限定为全部量测节点的最近共同祖先（LCA），即full-support物理链末端。case33的bus2、CIGRE的R1符合；paper多根出线的LCA=root，full边精确0，源在已知电压根处对电压无影响。

独立物理树oracle按原边逐次前推回代，不调用support/B/zbus；12项测试通过，包括内部负荷、零边、根多出线、串联聚合、源P/Q、不同Sbase及内部测点导数。首次测试的精确浮点相等断言已改用数值容差后全部通过，科学逻辑未更改。

候选拟合域因此必须允许0端点。不同候选中0边可能表达同一电气网络，结构评分应区分候选层级标签与折叠0边的电气等价；虚拟边不算真实导线，不能用0域宣称存在统一可辨识间隔。

## 10. 新稀疏拟合器（开发更新）

reference_benchmark/fitting.py 复用原 _RxInputProblem 完整AC残差与精确Jacobian；把Jacobian转换CSR并调用TRF/LSMR，省略原全历史信息谱SVD。最小化目标、共享R/X、逐时同接口源P/Q、box与X/R-cone均未改变。信息谱明确标注 unavailable，结果保留optimality/status/各初值/边界/计数/物理误差，仍是局部拟合。

默认初值只读当前历史：由V²一阶式拟合非负R/X（线性系数2r、2x），master减ΣPQ给源初值，再投影原box-cone；初始AC非法时仅缩小初始R并记录，不收缩最终域。fixed_r/fixed_x同样复用旧残差/Jacobian但改为CSR/TRF-LSMR，省略dense信息SVD；固定线初始AC非法时只调整源初值，不改固定R/X。本轮所有候选使用统一0下界<正上界，不按真值固定虚拟边位置。

已在小例验证稀疏/原dense目标和预测一致、有限差分Jacobian、0虚拟叶真实响应、预算失败完整保留、非法初始AC的缩小记录，以及等box界明确拒绝。有噪声固定线源拟合与旧dense的SSE差<1e-6、源P/Q差<1e-7；源Jacobian和有限差分一致。原始case33、CIGRE居民、paper11、paper15的保留内部负荷编码均与独立原物理树oracle一致。当前两个文件27测试通过（2026-10-18，1.79秒）。大网络耗时与正式候选覆盖由root后续pilot独立检验。
