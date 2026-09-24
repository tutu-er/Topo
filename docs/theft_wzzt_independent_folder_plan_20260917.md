# 偷电识别独立文件夹实施方案（人工审阅稿）

日期：2026-09-17。状态：**待审阅，尚未动手实现**。本文给出新独立文件夹的结构、拷贝清单、新增模块接口、实验矩阵与里程碑；第 8 节列出需要审阅者拍板的决策点。方案依据：当天三份设计文档（[两模块设计](wzzt_credibility_joint_theft_model_20260917.md)、[隐藏节点可辨识性](wzzt_hidden_node_theft_identifiability_20260917.md)）与[代码核对结论](wzzt_theft_feasibility_code_review_20260917.md)。

## 1. 目标与范围

新建与 `rnj_wzzt_core` 平级、可独立安装运行的包，目标是在最小代码量下打通：

**真实算例 → 偷电注入仿真（含首端总表）→ RX75-RNJ + wzzᵀ-MILP 主线 → 模块 2 偷电联合优化（版本 B）→ 模块 1 可信度评价与校准 → 六组最小实验报告。**

明确不做：版本 C 自由幅值非凸模型（仅留接口注释）、EM 概率化、多候选树联合搜索、现场数据、cli/legacy 兼容层。

## 2. 已存在的可复用资产（影响方案）

`rnj_wzzt_core/tests/meter_theft_pilot/`（今天 17:18–18:39 创建）已实现版本 B 的核心数学：

- `model.py`（227 行）：固定幅值 + 逐时整数位置 + R/X 联合重估的精确 MILP；含 v=r·b 四约束精确线性化、零幅值禁止误报、原始乘积/可行性/目标值三重直接校验、H0 嵌套保证。目前只在玩具树上运行。
- `credibility.py`：统一损失（电压 + 总表平衡双通道）、H0/H1 公平比较、完整流程零分布 rank 校准、位置剖面重求解。
- `fixtures.py`：玩具树生成器，已覆盖 hidden/terminal/switching/spur/normal 五种场景。

方案将这些代码**提升并推广**为新包的 theft 模块，不重复造轮子。

## 3. 新文件夹结构

~~~text
theft_wzzt/                          # 新顶级目录，独立 pyproject，包名 theft_wzzt
├─ pyproject.toml                    # 依赖同 rnj_wzzt_core（numpy/scipy/pandas/networkx）
├─ README.md                         # 一条命令复现 + 边界声明
├─ theft_wzzt/
│  ├─ models/                        # 【拷贝】network.py, lin_distflow.py, ac_powerflow.py
│  ├─ estimation/                    # 【拷贝】preprocessing.py, multiscenario.py,
│  │                                 #   constrained_least_squares.py, matrix_constraints.py,
│  │                                 #   laminar_l1_milp.py
│  ├─ graph/                         # 【拷贝】rooted_neighbor_joining.py, rooted_hierarchy.py,
│  │                                 #   sensitivity_geometry.py, bootstrap.py
│  ├─ scenario/                      # 【拷贝】settings.py, profiles.py, simulation.py,
│  │                                 #   validate_scenario.py
│  ├─ data/                          # 【拷贝】paper_style_terminal_lv.py, paper_style_case_bank.py
│  ├─ pipeline.py                    # 【拷贝，原样冻结】主线编排，不改逻辑
│  └─ theft/                         # 【新建】本项目的全部新增
│     ├─ theft_simulation.py         # 偷电注入 + 首端总表（§4.1）
│     ├─ theft_model.py              # 版本 B 联合 MILP，由 pilot/model.py 推广（§4.2）
│     ├─ credibility.py              # 模块 1，由 pilot/credibility.py 推广（§4.3）
│     ├─ scan.py                     # 基线 A：冻结 R/X 闭式非负扫描（§4.4）
│     ├─ baselines.py                # 基线 B：纯总表能量平衡；基线 C：H0 主线（§4.4）
│     └─ report.py                   # 六组实验的结果表与相容集合报告
├─ experiments/
│  └─ run_theft.py                   # 唯一实验入口（§6）
└─ tests/                            # 单元测试（§7 含等价类反例测试）
~~~

拷贝集约 4,100 行（不含 pipeline 766 行）。不拷：`cli.py`、`reporting.py`、`recipes.py`、`scenario/sc.py`（冗余转发）、`experiments/`（消融与文献基线）、`scripts/`、原 tests。import 前缀批量改写 `rnj_wzzt` → `theft_wzzt`，逐文件 diff 复核。

## 4. 新增模块接口

### 4.1 theft/theft_simulation.py（偷电生成 + 总表）

~~~python
@dataclass(frozen=True)
class TheftSpec:
    bus_id: int                 # 内部 hidden 或末端节点；逐时切换用列表
    amplitude_kw: np.ndarray    # 逐时非负有功，0 表示该时刻无偷电
    kappa: float                # 偷电无功 = kappa * 有功（第一版固定功率因数）

def simulate_with_theft(case_key, theft: TheftSpec | None, ...) -> tuple[TerminalizedNetwork, list[dict]]
~~~

- AC 潮流按**实际负荷**（报告 + 偷电）求解；识别器只见报告 P/Q 与电压。
- scenario dict 新增三个键：`P0_measured`/`Q0_measured`（首端总表，独立带噪）、`theft_truth`（真值，只许评估用，不进任何拟合/选阈）。
- 独立的线损通道：返回无偷电时的潮流线损 `loss_true`，供"扣除估算线损"路线构造 â_t；同时保留有符号残差序列（设计文档 §0.1 要求）。
- 新独立生成器函数，**不改** `validate_scenario` 的 hidden-load 禁令（原验证继续守护无偷电主线）。

### 4.2 theft/theft_model.py（模块 2，版本 B）

由 pilot/model.py 推广，接口不变式保留：

~~~python
def fit_theft_milp(tree, data, *, allow_theft=True, fixed_locations=None,
                   fixed_rx=None, weight_bounds, time_limit) -> Fit
~~~

- 候选位置集 H = 约化树上允许的插入点（内部节点祖先链 + 末端），按设计文档 §3.1 的 c_eh 构造；逐时 Σ_h s_ht ≤ 1。
- 幅值 â_t 固定输入（来自总表路线）；保留有符号 ẫ_t 供模块 1 评分。
- 沿用 pilot 的三重直接校验（可行性、v=r·b 原始乘积、目标一致性）与"零幅值不报检出"纪律。
- 工程差异点：pilot 在 ≤5 节点玩具树上秒解；真实算例 T×|H| 达 96×十数，需实测 HiGHS 耗时，超阈值则按时刻分块 + 合并候选（在 README 记录实际规模上限）。

### 4.3 theft/credibility.py（模块 1）

~~~python
def compare_h0_h1(tree, data) -> dict        # 公平重估比较，H1 嵌套检查
def calibrate_null(run_full_procedure, null_scenarios, n_replicates) -> NullDist
def compatibility_report(tree, data, null, ...) -> Report
~~~

- 双通道损失（电压 + 总表平衡），尺度固定、预先声明。
- 零分布校准：无偷电场景上**重跑完整流程**（R/X 估计 → 树 → theft MILP → Δ），输出目标误报率阈值；复用 graph/bootstrap.py 的 moving-block 重采样做时间相关保持。
- 输出为诊断报告（异常显著性、扩展后残差相容性、位置相容集合宽度、拓扑集合变化），**不输出单一 0–100 分数**。

### 4.4 对照算法（baselines）

| 基线 | 内容 | 作用 |
|---|---|---|
| A | 冻结主线 R/X，逐位置闭式非负扫描（[fᵀWr]₊²/fᵀWf） | 证明联合重估的必要性 |
| B | 纯总表能量平衡阈值（â_t 超阈即告警，不用电压） | 证明电压通道的定位增量 |
| C | H0：无偷电 wzzᵀ 主线原样 | 公平 H0/H1 比较的天然基线 |

## 5. 算例与接口

- 保留四个 paper 算例 builder 不变；实验默认 **paper15 + soumalas11**（快），**pengwah18** 必跑（与 Pengwah 2023 NTL 文献直接同源对照）。
- 偷电注入位置：每算例选 1 个内部 hidden 节点 + 1 个末端节点；另设逐时切换位置场景。
- 接口约定：拓扑主线维持 daily_demean（辨识树用变化量）；theft 模块用 raw 绝对量 + 受约束截距（恒定偷电不被吸收——这是可辨识性的硬条件，README 中显著声明）。

## 6. 实验矩阵（experiments/run_theft.py）

| # | 场景 | 期望行为 |
|---|---|---|
| 1 | 无偷电正常 | 基线 A/B 与版本 B 均不误报（校准后 FPR 达标） |
| 2 | 单个内部节点偷电 | 检出 + 定位到该节点或其相容集合 |
| 3 | 单个末端节点偷电 | 检出 + 逐户定位 |
| 4 | 位置逐时切换 | 逐时定位正确率 |
| 5 | 比例歧义（无观测侧支路） | 输出区域集合，**不**强行单点 |
| 6 | 无偷电但 R/X/根参考有偏差 | 不误报（模型遗漏的判退） |

每组报告：检测 FPR/检出率、位置集合覆盖/宽度、幅值误差、拓扑 F1 变化、MILP 耗时与最优性证书。

## 7. 单元测试要点

- 等价类反例必须**输出不可辨**：成比例响应列、尺度混淆（无总表锚点）、恒定偷电被自由截距吸收——三个测试断言"相容集合 >1 或拒绝判断"，而非断言检出。
- MILP 校验：v=r·b 直接乘积误差 < 2e-6；H1 目标 ≤ H0 目标。
- 拷贝保真：paper15 上 RX75-RNJ F1 = 1.0 与主线一致（复现性锚点）。

## 8. 待审阅决策点

| # | 决策 | 选项 | 建议 |
|---|---|---|---|
| D1 | 文件夹名/位置 | `theft_wzzt/`（顶级） | 采纳 |
| D2 | pipeline.py 处理 | (a) 原样拷贝冻结 (b) 精简重写 | (a)：保真优先，简洁靠不拷 experiments/cli 体现 |
| D3 | meter_theft_pilot 处置 | (a) 迁移进新包、原处删除 (b) 迁移、原处留只读副本 | (a)，但删除前请确认（不可逆） |
| D4 | 算例全集 | 4 个全留 / 只留 3 个 | 全留，实验默认跑 2+1 |
| D5 | 截距处理 | theft 通道 raw+受约束截距 | 采纳，拓扑通道不动 |
| D6 | 里程碑规模 | §9 五步 | 采纳 |

## 9. 里程碑（每步可独立验证）

1. **M1 骨架**：拷贝 + import 改写 + 安装，`python -m experiments.run_mainline --cases paper15` 复现 F1=1.0。
2. **M2 仿真**：theft_simulation 生成六组场景数据，真值/报告/总表三通道隔离，单元测试通过。
3. **M3 模块 2**：theft MILP 在真实算例上求解，三重直接校验通过，记录耗时。
4. **M4 模块 1**：零分布校准编排 + H0/H1 比较，场景 1/5/6 的判退行为正确。
5. **M5 实验报告**：六组实验全表 + 相容集合报告，写入 outputs/theft/。
