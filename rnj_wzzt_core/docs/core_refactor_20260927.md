# 核心代码整理记录（2026-09-27）

> 本文保留当日早期职责拆分的历史记录及验证结果。后续按减少文件数量的要求，
> 数据和模型辅助模块已并回 `laminar_l1_milp.py`，当前 MILP 实现仅保留该文件和
> `gurobi_milp.py`。最新结构见 [代码审核指南](code_structure.md)。

本轮整理保留当前数学模型、默认配置和已知实验适配接口，重点让求解流程与模型组装可以分别阅读。基线包含整理前工作区的未提交修改，不以 Git HEAD 代替当前算法。

## 代码职责与阅读顺序

1. `rnj_wzzt/pipeline.py`：数据准备 → RNJ 选择 → 可选收缩 → 允许支撑映射 → MILP 完成。
2. `rnj_wzzt/estimation/laminar_l1_milp.py`：先读 `fit_laminar_l1_sensitivity`，再读 `_solve_extension_prepared`、`_solve_fixed_prepared` 和 `_run_milp`。
3. `rnj_wzzt/estimation/_laminar_l1_model.py`：从 `build_extension_model` 查看残差、非空、允许集合、乘积线性化、层状与不重复约束。
4. `rnj_wzzt/estimation/_laminar_l1_common.py`：按需查阅数据与结果类型、输入校验、支撑规则、矩阵重建、验证误差和选模规则。
5. `rnj_wzzt/reporting.py`：结构评价与求解证据的输出。

| 模块 | 整理前行数 | 整理后行数 | 职责 |
|---|---:|---:|---|
| `laminar_l1_milp.py` | 1497 | 613 | 求解与前向路径 |
| `_laminar_l1_common.py` | — | 554 | 数据、结果与纯辅助函数 |
| `_laminar_l1_model.py` | — | 378 | 变量、稀疏约束与模型构造 |
| `pipeline.py` | 788 | 786 | 主流程与支撑映射 |
| `reporting.py` | 235 | 223 | 评价和序列化 |

三个 MILP 文件合计 1545 行。职责拆分增加了显式导入和模型记录；同时合并了基础变量组、R/X 乘积约束、重复重拟合调用和路径记录。主要收益是阅读时需要同时跟踪的逻辑减少。

## 具体简化

- `L1Model` 集中保存约束与变量切片；解的读取使用 `model.r`、`model.x` 和 `model.intercepts`。
- 固定族 LP 与扩展 MILP 共用基础变量创建，扩展模型再追加新原子和组合约束。
- R/X 有界乘积使用同一段约束循环；仍在每个终端对内先加入 R、后加入 X，保留原行序。
- 残差 helper 直接接收模型，不再逐个传递所有变量切片。
- 前向路径通过局部 `refit`、`path_point` 复用重复参数，扩界后读取当前边界。
- 主流程内部统一使用 `support_search_mode`、`allowed_supports`；支撑展开和新旧报告字段共用计算结果。

## 行为与兼容边界

- 默认仍是 RNJ 支撑白名单；`unrestricted` 仍表示不施加白名单限制。
- 已接受支撑不得重复；新支撑与已有族层状相容；已有权重联合重新拟合。
- `initial_supports` 的冻结、收缩、边界扩大和从初始族重新开始的语义保持一致。
- 空允许集合与 `None` 不混同。前者不允许扩展，后者搜索全部合法支撑。
- 超时 incumbent 不作为已认证新支撑接受；目标值和对偶界仍供外层判断无增益。
- 验证复用训练截距，按原一标准误差规则选择路径点。
- 公共函数、结果类型、旧参数别名和父项目转发入口保留。
- `_run_milp`、`_solve_fixed_prepared`、`_solve_extension_prepared`、`_make_path_point` 仍在主模块定义和调用；实验适配器的 monkeypatch 继续生效。
- 模型内部残差 helper 的私有签名调整为接收 `L1Model`；仓库未发现其外部调用者。

## 验证结果

- 独立核心完整测试：**719 passed**。
- 父项目完整兼容测试：**152 passed**。
- 行为审计：**19 组结果，128 个实际 LP/MILP 模型，0 项差异**。
- 隔离模式导入全部 **30 个核心模块**，没有父项目模块导入；离线 wheel 包含全部 **30 个 Python 源文件**，与当前源码逐字节一致。
- 稀疏模型比较目标、整数标记、变量边界、约束矩阵及行边界，要求数值完全一致。
- 路径、支撑和状态精确比较；解的浮点数使用 `rtol=1e-8, atol=1e-9`。运行时间不参与比较。
- 真实小规模流程覆盖默认 RNJ、旧参数别名、收缩 RNJ、单终端增删扩展；使用固定训练和验证复制编号。
- 单终端增删的第十次扩展在原基线中未通过严格最优性检查，整理后保留相同停止结果。该审计不把它计作完整认证的路径，也不构成全局拓扑恢复保证。

## 审计文件与复现

全部证据保存在仓库根目录下 `outputs/core_refactor_20260927_095726/`：

- `before/`、`source_before.json`：当前工作区的整理前源码及 SHA-256 清单。
- `behavior_audit.py`、`baseline.json`、`after.json`：可复现模型与行为比较。
- `after.summary.json`：`comparison.passed=true`、`difference_count=0`。
- `core_tests.txt`、`parent_tests.txt`：完整测试输出。
- `source_after.json`、`source_changes.json`、`source_diff.patch`：整理后的源码记录与差异。
- `boundary_audit.json`：独立导入和 wheel 模块覆盖检查。

在仓库根目录使用已有 Topo 环境：

```powershell
python -m pytest -q rnj_wzzt_core/tests
python -m pytest -q tests
python outputs/core_refactor_20260927_095726/behavior_audit.py `
  --source-root rnj_wzzt_core `
  --output outputs/core_refactor_20260927_095726/recheck.json `
  --compare outputs/core_refactor_20260927_095726/baseline.json
```

需要查看整理前实现时，使用 `before/rnj_wzzt_core/rnj_wzzt/`；恢复时先检查后续修改，按具体文件合并，避免整目录覆盖新的工作。
