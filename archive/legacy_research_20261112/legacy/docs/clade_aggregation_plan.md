# 聚合精确性定理 + 相图实验 实施计划

## 目标

1. **理论**：新写一份独立 LaTeX 理论文档，证明"一层 clade 聚合"的精确性引理与两阶段样本复杂度定理。所有模型假设严格对齐项目根目录的**实际代码**（不是 .md 文档）。
2. **实验**：新增"聚合相图"实验——受控 clade 尺寸的案例族 × 数据/噪声 regime × 聚合开/关配对，验证定理的三条预测。

## 已核实的代码事实（理论与实验的地基）

- 仿真：`models/ac_powerflow.py` AC 前推回代；`experiments/common.py:98-100` 相对高斯噪声 `P_meas = P + N(0, σ_PQ·|P|)`；drop 目标 `Y = v_root² − V_i²`（`common.py:106`，`estimation/preprocessing.py:38`）。
- 估计器：`estimation/multiscenario.py:29 fit_projected_sensitivity`——`drop = [P|Q|场景示性]·Θ` 的 `lstsq`（`:47-68`），对称化+截负+ordered 投影（`matrix_constraints.py:70`）。**不是** `sensitivity.py` 里的旧估计器。
- 距离：`models/lin_distflow.py:63-67` `d_ij = R_ii+R_jj−2R_ij`；RX75 各自归一后 0.75/0.25 混合、根深 = 对角线同权混合（`estimation/baseline.py:61-94`）。
- RNJ：`graph/rooted_neighbor_joining.py:88 rooted_neighbor_joining(shared_path_matrix, root_depths, ...)`，容差 `0.16·median(depths)`。
- 聚合去嵌入（`graph/rooted_hierarchy.py`）：
  - boundary 行 `_boundary_sensitivity_row:182-202`：簇内取 `clip(quantile_0.2(off-diag R[C,C]), 0, min diag)`，簇外取 `median_{i∈C} R_ij`；
  - `aggregate_rooted_scenarios:271-287`：`ṽ²_{g,i} = v_i² + Σ_{j∈C}[R_ij−R_gj]₊P_j + Σ_{j∈C}[X_ij−X_gj]₊Q_j`，对子节点取 median，再 `mean + w·(median−mean)`，**w 默认 0.50**（`pipeline/unified_topology_pipeline.py:71`；`docs/nj_edge_aggregation_final.md` 写的 0.15 与代码不符）。
- 门控：`pipeline/peripheral_edge_proposals.py:335`（support≥0.875、margin≥0.10、长度比≥0.02、簇≤5）；外部确认 `_confirmed_peripheral_clusters`（`unified_topology_pipeline.py:302-334`，4 源：Ordered base / GTLS≥0.5 / AC / quartet）。
- 排序：`pipeline/ac_likelihood.py:522` 留出场景 AC 预测 NLL；oracle 指标只在实验脚本里算（`run_unified_topology_pipeline_sweep.py:75-79`）。
- 理论模板：`docs/latent_tree_neighbor_joining_proof.tex`（ctexart 中文定理环境、证明思路风格）；其 NJ 精确恢复定理与 `eq:nj-update`（cherry 收缩恒等式）是引理的天然锚点；Atteson 安全半径只被引用未证明。
- 符号冲突预警：ch04 用 `T` 表树、ch09 用 `T` 表样本数——新文档树用 `T`、每场景样本数用 `m`、场景数用 `S`。
- 案例基建：`data/paper_style_case_bank.py` 的 `_make_case(hidden, terminals, edges)` 声明式模式 + `CASE_BUILDERS` 注册表（`:214-219`）；`experiments/common.py simulate_case()` 按 key 取 builder，注册即接入。隐藏节点必须 degree≥3（度 2 不可辨识）。
- Sweep 基建：`run_unified_topology_pipeline_sweep.py` 的 `_evaluate/_persist/_write_report` 结构；消融先例 `run_two_level_aggregation_ablation.py --paired-nj-ablation`（`dataclasses.replace(config, ...)` 配对重跑）；sparse 单条件约 2 分钟、串行、支持 `--resume`。
- 无现成相图绘图器；`run_ac_quartet_large_sweep.py:336-362 _write_plot` 是模板。

## Part A：理论文档

**新文件**：`docs/clade_aggregation_theory.tex`（独立 ctexart 中文文档，环境/风格对齐 `latent_tree_neighbor_joining_proof.tex`，用 xelatex 编译）。

章节与内容：

1. **引言与代码对应**：一段说明本文假设逐一对应代码位置；末尾附录 A 放"符号 ↔ 代码 file:line"对照表（上面"已核实"清单）。
2. **模型与符号**（§2）：
   - 树 `T=(V,E)`、根 0、终端集 `L`（|L|=n）、隐藏节点 `H`（degree≥3，最小隐藏树假设，对齐 proof 文档 § assumption）；
   - `R_ij = Σ_{e∈path_0i∩path_0j} r_e`（ch08 式）；`d^R_ij = R_ii+R_jj−2R_ij`；
   - **估计模型严格按代码写**：场景 `s=1..S`、样本 `t=1..m`；`Y_i(t) = v_0²(t)−V_i²(t)`；`Y = P Rᵀ + Q Xᵀ + γ_s 1ᵀ + E`，`E` 为逐终端独立、相对量测噪声经 `2V` 缩放后的次高斯误差；注明线性模型与 AC 仿真的差距作为有界模型失配项并入 margin（AC 只用于最终排序，不进定理）。
3. **引理 1（外部等价）**：悬挂 clade `C`（附着点 `g`）⟹ 对任意 `i∉C`、`j,k∈C`：`R_ij = R_ik =: R_ig`。由公共路径形式一行证得。推论：簇外注入对簇内各终端的电压降贡献相同——这是代码去嵌入"只用簇内 P/Q 列"（`rooted_hierarchy.py:216-218` 注释）的精确依据。
4. **引理 2（聚合精确性）**：
   - (a) KCL：`P_g = Σ_{j∈C}P_j` 恰为流过附着边的总注入；
   - (b) 去嵌入恒等式：理想量 `Y_g := Y_i − Σ_{j∈C}[R_ij−R_gj]₊P_j − Σ_{j∈C}[X_ij−X_gj]₊Q_j` 与 `i∈C` 无关，且恰等于收缩树 `T/C` 上 pseudo 节点的 drop 模型 `Y_g = Σ_j R_gj P_j + Σ_j X_gj Q_j + γ_s + ε_g`。即**在真实 clade + 线性模型下聚合无近似误差**；代码的 `median`/`λ=0.5` 混合是含噪鲁棒版；
   - (c) 附着深度恒等式：若 `g` 在 `C` 内分出 ≥2 支，则 `R_gg = min_{j≠k∈C} R_jk`；代码用 `quantile_0.2` 是其鲁棒替代（`rooted_hierarchy.py:191-194`）。
5. **引理 3（降噪与降维）**：对 `i∈C` 的 `Y_{g,i}` 取均值（噪声跨终端独立）⟹ pseudo 观测方差 `σ²/|C|`；主干回归维数 `2n+S → 2(n−|C|+1)+S`，设计阵条件数不降（代码跟踪为 `base_condition_number`）。
6. **定理 1（两阶段样本复杂度）**：
   - 步骤 1（集中不等式，新机器）：lstsq 行误差 `=(ZᵀZ)⁻¹Zᵀε`，次高斯 + 对 `n²` 个元素 union bound ⟹ 以概率 `≥1−δ`，`‖R̂−R‖_max ≤ η = O(σ_eff·κ(Z)·√(log(n/δ)/(S·m)))`（证明思路级，对齐模板文档风格）；
   - 步骤 2：`|d̂−d| ≤ 4η`（ch09 链），NJ/RNJ 安全半径 `4η < ½ℓ_min`（Atteson，对齐 proof 文档 `eq:atteson` 与 ch04 `thm:noisy_margin_mst` 的 `ε<γ_min/2`）；
   - 步骤 3（两阶段）：**条件于确认 clade 为真**，主干阶段有效噪声 `σ²→σ²/|C|`、未知树尺寸 `n→n−|C|+1`，故主干精确恢复所需样本量 `S·m ≳ σ²κ²log(n/δ)/γ̃²` 约按 `|C|` 倍下降，且收缩树 margin `γ̃_min` 不再受簇内短边限制；局部阶段以 `v_g` 为局部根、问题规模仅 `|C|`。clade 检测正确性由 quartet/four-point 检验 + `O(n⁴)` union bound 单独给条件（对应 `quartet_significance.py`）。
7. **推论**：(i) 真实 clade 下聚合单调不伤（方差、维数、margin 三个通道分别论述——对应代码"软收缩、未聚合候选始终保留"的设计）；(ii) 主干恢复对簇内负荷相关免疫（把 ch05 反例转化为动机）；(iii) 门限解释：support≥0.875 等是 P(clade 为真) 的经验估计，定理给出"何时该聚合"的定量判据方向。
8. **假设与局限**：线性模型 vs AC、跨终端误差相关时 `σ²/|C|` 退化为 `σ²`（指出 simulator 噪声逐终端独立，假设成立）、度 2 链、λ 混合暂无理论。

**编译验证**：xelatex 两遍，无错误、无未解引用。

## Part B：相图实验

**新文件 1**：`terminal_case33/data/clade_grid_case_bank.py`
- 工厂 `make_clade_grid_case(n_terminals, clade_size, ...)`：骨干隐藏节点链 + 每个附着点挂 `clade_size` 个 service terminal（保证每个隐藏节点 degree≥3：附着点 = clade(≥2) + 骨干后继；链尾挂 ≥2 个 terminal）；阻抗/长度沿用 `_make_case` 的两档线路库模式；
- 预注册网格点进 `CASE_BUILDERS`（如 `grid16_k2/k4/k8`，n=16 固定）——`experiments/common.py simulate_case()` 零改动接入；对照组直接用已有的平衡二叉树 `flynn16`（无大 clade）。
- `aggregation_max_cluster_size` 在实验 config 里按 k 设置（`max(5,k)`），避免 5 的上限截断 k=8。

**新文件 2**：`experiments/run_aggregation_phase_diagram.py`
- 复用 `run_unified_topology_pipeline_sweep.py` 的 `_build_scenarios/_metrics/_persist/report` 结构与输出行 schema；
- 新增 `--aggregation {on,off}` 轴：`dataclasses.replace(config, enable_ordered_aggregation=flag, enable_nj_edge_aggregation=flag, aggregation_max_cluster_size=...)`，折叠进 `condition_id`；**同 seed 配对**（对齐 ablation 先例）；保留 `--resume`；
- 额外记录 k、clade 数、配对增益（on−off 的 F1/exact），输出 `paired_aggregation_gain.csv`；按 `_filtered_f1` 先例拆分 backbone/peripheral F1（验证"增益集中在主干"）。

**新文件 3**：`experiments/plot_aggregation_phase_diagram.py`
- 读 metrics.csv → 热图（k × noise，分 regime）+ 配对增益曲线；在图上标注定理预测的三条单调性（增益随 k 增、随噪声增、随样本减）。

**新文件 4**：`tests/test_clade_grid_case_bank.py`
- 生成案例连通、隐藏节点 degree≥3、terminal 数正确、`CASE_BUILDERS` 键可调用、`rooted_clades` 真值可计算。

**实验矩阵**（控制总时长）：
- 案例 {`flynn16` 对照, `grid16_k2`, `grid16_k4`, `grid16_k8`} × regime {sparse, medium} × noise {low, nominal, high} × 聚合 {on, off} × trials 3 = **144 条件**；
- 串行约 5h → 按案例拆 4 个进程并行（不同 `--cases`/`--output` 子目录）≈ 1.5h；rich regime 留作可选 flag（定理预测那里增益饱和）；
- smoke 模式：1 案例 × sparse × nominal × 1 trial × on/off，quartet/bootstrap replicates 降到 2，确认端到端跑通再全开。

**定理↔实验的三条待验预测**：
1. 配对增益随 k、随噪声单调增，随样本量减；
2. `ac_candidate_oracle_f1`/`contains_exact` 在 sparse×high 改善最大（候选覆盖机制）；
3. 增益集中在 backbone F1，peripheral 不变。

## Part C：文档-代码不一致的小修正（顺手，保持最小）

- `docs/nj_edge_aggregation_final.md`：λ 0.15 → 代码实际默认 0.50（示例 config 同步）；
- `docs/current_technical_mainline.md`：第 2 步"remove the observed root squared-voltage mode"改为准确描述（drop 目标 `v_root²−V_i²` + daily demean）；外部确认源 3 → 4（含 AC-validated）。

## 评审补强（证明中必须处理的技术点）

1. **ordered 投影纳入集中不等式**：定理 1 步骤 1 的 η 界针对 `lstsq` 输出，但代码随后做 `project_ordered_sensitivity_matrix` 循环半空间投影（`matrix_constraints.py:70`）。需论证半空间投影链在 entrywise max-norm 下非扩张（每步是向闭半空间的正交投影，max-norm 不增），使 η 界覆盖真实估计器输出。
2. **RX75 归一化的随机分母**：代码先按正项均值归一化 `d̂^R/d̂^X` 再 0.75/0.25 混合（`baseline.py:83-89`），归一化因子是随机量。处理方案：主定理对未归一化距离陈述并注明差异，另加一个"归一化因子集中于常数"的小引理（正项均值的集中由 entrywise 集中直接推出）。
3. **RNJ 安全半径 ≠ Atteson 原版**：RNJ 输入是 shared-path 矩阵 + 估计根深 `diag(R̂)`（`rooted_neighbor_joining.py:88`、`baseline.py:90-93`）。根深误差 ≤η 会经 `ρ=(h_i+h_j−d_ij)/2` 进入 shared-path，需显式重算误差预算（预计 ≤3η，写证明时核实），给出 RNJ 版安全半径，而不是直接引用 `4η < ½ℓ_min`。

另两处小修正：

4. **对照组表述**：flynn16 是平衡二叉树（全是 size-2 cherry），聚合仍会触发——对照预期改为"增益随 k 单调递增"的相对比较，不设"对照组零增益"预期。
5. **运行环境**：实验命令统一为在项目根目录下 `conda activate Topo` 后以 `python -m experiments.run_aggregation_phase_diagram ...` 运行（包已 `pip install -e .`）。

## 执行顺序

1. 写 `clade_aggregation_theory.tex` 并 xelatex 编译通过；
2. 写案例工厂 + 测试，`pytest tests/test_clade_grid_case_bank.py` 通过；顺带跑 `tests/` 中 data/estimation 相关子集确认无回归；
3. 写相图入口 + 绘图脚本，smoke 跑通并检查 metrics.csv 字段；
4. 后台并行启动完整 144 条件；完成后绘图、写 `report.md` 式小结，对照三条预测下结论（若预测失败，在理论文档"局限"节补说明）；
5. Part C 文档修正。

## 验证清单

- [ ] tex 编译无错误；定理假设逐条有代码 file:line 对应（附录 A 对照表）；
- [ ] 新测试通过，相关旧测试无回归；
- [ ] smoke 实验端到端成功，on/off 配对同 seed；
- [ ] 完整相图产出热图 + `paired_aggregation_gain.csv`；
- [ ] 三条预测逐条给出"支持/不支持"结论。
