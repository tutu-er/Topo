"""额外负荷源模型：固定支撑路径选择，以及源路径与单 atom 的联合扩展。

固定外部源 P/Q 幅值，使用 h[k]、w_r[k]=r[k]*h[k]、w_x[k]=x[k]*h[k]。
h 表示当前原子树中整条边是否位于根到源的路径，跨时间和场景共用。
这里只覆盖原子树节点接入，不表示压缩边内部任意物理母线的接入。

通用记录和普通拓扑约束直接引用核心，不复制实现；所有预测复用核心
同一组绝对残差行。固定模型选择一个已有 clade 及其全部祖先，排除根。
每次建模只创建一个完整 h：固定模型为 K 维，扩展模型为 K+1 维，末项
对应本轮待优化 z 的 atom。不接收上一轮 h，也不固定任何历史路径结果。
扩展时复用核心的支撑关系变量，联合约束完整 h 与待优化的 z，
再建立源乘积并将源响应加入同一组残差。
None 或 P/Q 全零按普通模型构建，不增加源变量。高层搜索尚未接入。

同一已建路径且不增加阻抗时，额外 clade 标记不改变源响应；新增非零
线路或压缩边内部接入则未必等价，不能普遍合并为上级节点。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from rnj_wzzt.estimation.laminar_l1_milp import (
    ExtensionSolution,
    FixedSupportSolution,
    IndexSupport,
    L1Model,
    LaminarL1PathPoint,
    LaminarL1Result,
    SolverDiagnostics,
    SourceTermProvider,
    _ConstraintBuilder,
    _PreparedScenarios,
    _VariableBuilder,
    _add_absolute_residual_constraints,
    _add_extension_constraints,
    _add_atom_product_constraints,
    _add_laminar_extension_constraints,
    _base_model,
    _base_extension_model,
    _prepare_scenarios,
    _require_positive_finite,
    build_extension_model,
    build_fixed_model,
    normalize_supports,
)
from rnj_wzzt.estimation.unmetered_inputs import PreparedSourceInputs, _positions


@dataclass(frozen=True)
class SourceBlocks:
    """当前模型完整的源变量切片，长度均为当前 atom 总数。

    h: 二进制源路径；w_r/w_x: 有界连续乘积变量。
    固定模型长度 K；扩展模型长度 K+1，末项对应待优化的 atom。
    切片表示本次 MILP 的列，不包含上一轮解，也不是物理母线 ID。
    求解后用 result.x[blocks.h] 等读取，build 阶段尚无变量数值。
    """

    h: slice
    w_r: slice
    w_x: slice


SourceSelectionBuilder = Callable[
    [L1Model, tuple[IndexSupport, ...], SourceBlocks], None
]


@dataclass(frozen=True)
class SourceExtensionBlocks(SourceBlocks):
    """完整 h/w_r/w_x 加上待优化支撑的终端响应乘积。

    继承的 h/w_r/w_x 均为 K+1 维、一次分配的连续变量块。
    v_r/v_x 为 n 维，分别是 w_r[K]*z[i]、w_x[K]*z[i]。
    h[K] 与其他 h 分量共同优化；这里不保存任何先前路径结果。
    """

    v_r: slice
    v_x: slice


SourceExtensionSelectionBuilder = Callable[
    [L1Model, tuple[IndexSupport, ...], SourceExtensionBlocks, slice], None
]


def _copy_source_amplitudes(
    source_inputs: PreparedSourceInputs | None,
    *,
    expected_lengths: tuple[int, ...] | None = None,
    time_indices: tuple[object, ...] | None = None,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]] | None:
    """复制固定幅值；若提供原观测索引，则再次按标签对齐。"""
    if source_inputs is None:
        return None
    if not isinstance(source_inputs, PreparedSourceInputs):
        raise ValueError("source_inputs must be PreparedSourceInputs or None")
    count = len(source_inputs.p)
    if count == 0 or any(len(values) != count for values in (
        source_inputs.q, source_inputs.time_indices, source_inputs.provenance,
    )):
        raise ValueError("source inputs must have matching nonempty scenario blocks")
    if expected_lengths is not None and count != len(expected_lengths):
        raise ValueError("source inputs must have one block per scenario")

    p_blocks, q_blocks = [], []
    for scenario_index, (p, q, index, provenance) in enumerate(zip(
        source_inputs.p, source_inputs.q,
        source_inputs.time_indices, source_inputs.provenance, strict=True,
    )):
        try:
            p = np.asarray(p, dtype=float).copy()
            q = np.asarray(q, dtype=float).copy()
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("source P/Q blocks must contain numeric values") from exc
        if p.ndim != 1 or p.size == 0 or q.shape != p.shape:
            raise ValueError("source P/Q blocks must have matching shape (T,)")
        if not np.isfinite(p).all() or not np.isfinite(q).all() or (p < 0.0).any():
            raise ValueError("source P must be finite and nonnegative; Q must be finite")
        if len(index) != p.size or not getattr(index, "is_unique", False):
            raise ValueError("source time indices must have matching unique labels")
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError("source provenance must be a nonempty string")
        if expected_lengths is not None and p.size != expected_lengths[scenario_index]:
            raise ValueError("source length must match each scenario time block")
        if time_indices is not None and time_indices[scenario_index] is not None:
            positions = _positions(index, time_indices[scenario_index], "source time")
            p, q = p[positions], q[positions]
        p_blocks.append(p)
        q_blocks.append(q)
    if all(not np.any(p != 0.0) and not np.any(q != 0.0)
           for p, q in zip(p_blocks, q_blocks, strict=True)):
        return None
    return tuple(p_blocks), tuple(q_blocks)


def _make_response_provider(
    amplitudes: tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]],
    response_terms: Callable[
        [L1Model, int], tuple[Mapping[int, float], Mapping[int, float]]
    ],
) -> SourceTermProvider:
    """将已经复制、对齐的幅值乘到线性响应上，不分配变量或添加约束。"""
    if not callable(response_terms):
        raise ValueError("response_terms must be callable")
    p_blocks, q_blocks = amplitudes

    def source_terms(model, scenario_index, time_index, output_index):
        if scenario_index < 0 or time_index < 0:
            raise ValueError("source observation indices must be nonnegative")
        try:
            p_value = float(p_blocks[scenario_index][time_index])
            q_value = float(q_blocks[scenario_index][time_index])
        except IndexError as exc:
            raise ValueError("source amplitudes do not cover the requested observation") from exc
        r_terms, x_terms = response_terms(model, output_index)
        if not isinstance(r_terms, Mapping) or not isinstance(x_terms, Mapping):
            raise ValueError("response_terms must return two coefficient mappings")
        coefficients: dict[int, float] = {}
        for amplitude, terms in ((p_value, r_terms), (q_value, x_terms)):
            for variable, coefficient in terms.items():
                coefficients[variable] = (
                    coefficients.get(variable, 0.0) + amplitude * float(coefficient)
                )
        return coefficients, 0.0

    return source_terms


def make_source_response_terms(
    source_inputs: PreparedSourceInputs | None,
    response_terms: Callable[
        [L1Model, int], tuple[Mapping[int, float], Mapping[int, float]]
    ],
) -> SourceTermProvider | None:
    """返回 uP[s,t]*g_R[i]+uQ[s,t]*g_X[i] 的表达式回调，不返回模型。

    response_terms(model, i) 返回两个线性系数字典，key 是已经分配的
    求解器变量索引。幅值复制后固定；P 非负，Q 可带符号。
    None/全零返回 None。输入必须已按同一组场景的时间顺序准备。
    """
    amplitudes = _copy_source_amplitudes(source_inputs)
    if amplitudes is None:
        return None
    return _make_response_provider(amplitudes, response_terms)


def _source_weight_columns(model: L1Model) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """按当前 atom 顺序列出全部 R/X 列；核心的新权重列并不连续。

    固定模型顺序为 supports；扩展模型顺序为 (*supports, 当前 z)。
    只读取变量列号，不读取权重或源路径的已求解数值。
    """
    r_columns = tuple(range(model.r.start, model.r.stop))
    x_columns = tuple(range(model.x.start, model.x.stop))
    if len(r_columns) != len(x_columns):
        raise ValueError("source R/X blocks must have matching atom counts")
    if (model.new_r is None) != (model.new_x is None):
        raise ValueError("source extension requires both R/X weight blocks")
    if model.new_r is not None:
        if model.new_r.stop - model.new_r.start != 1 or model.new_x.stop - model.new_x.start != 1:
            raise ValueError("source extension requires one additional R/X weight")
        r_columns += (model.new_r.start,)
        x_columns += (model.new_x.start,)
    return r_columns, x_columns


def _add_source_variables(
    model: L1Model,
    atom_count: int,
    *,
    r_upper_bound: float,
    x_upper_bound: float,
) -> SourceBlocks:
    """一次分配当前全部 atom 的 h、w_r、w_x；不继承历史求解值。"""
    if (isinstance(atom_count, (bool, np.bool_))
            or not isinstance(atom_count, (int, np.integer)) or atom_count < 0):
        raise ValueError("source atom_count must be a nonnegative integer")
    r_columns, x_columns = _source_weight_columns(model)
    if len(r_columns) != atom_count or len(x_columns) != atom_count:
        raise ValueError("source atom_count must match all current R/X weights")
    r_upper_bound = _require_positive_finite("r_upper_bound", r_upper_bound)
    x_upper_bound = _require_positive_finite("x_upper_bound", x_upper_bound)
    return SourceBlocks(
        h=model.variables.add("source_h", atom_count, upper=1.0, integral=True),
        w_r=model.variables.add("source_w_r", atom_count, upper=r_upper_bound),
        w_x=model.variables.add("source_w_x", atom_count, upper=x_upper_bound),
    )


def _add_bounded_source_product(
    model: L1Model, *, weight: int, indicator: int, product: int, upper_bound: float,
) -> None:
    """对非负有界权重添加 product=weight*indicator 的三条线性化约束。

    indicator 由调用方保证二元。
    """
    if model.variables.lower[weight] < 0.0 or model.variables.upper[weight] > upper_bound:
        raise ValueError("source product bound must cover nonnegative R/X weights")
    model.rows.add({product: 1.0, indicator: -upper_bound}, upper=0.0)
    model.rows.add({product: 1.0, weight: -1.0}, upper=0.0)
    model.rows.add(
        {product: 1.0, weight: -1.0, indicator: -upper_bound}, lower=-upper_bound,
    )


def _add_source_product_constraints(
    model: L1Model,
    source_blocks: SourceBlocks,
    *,
    r_upper_bound: float,
    x_upper_bound: float,
) -> None:
    """对当前全部 atom 精确线性化 w_r[k]=r[k]*h[k]、w_x[k]=x[k]*h[k]。

    变量非负由 bounds 保证；每个乘积添加 w<=U*h、w<=weight、
    w>=weight-U*(1-h)。只有 h 为二进制且 U 覆盖 weight 上界时才精确。
    扩展模型的末项使用核心 new_r/new_x 的列，和其他分量同等处理。
    """
    bounds = (
        _require_positive_finite("r_upper_bound", r_upper_bound),
        _require_positive_finite("x_upper_bound", x_upper_bound),
    )
    count = source_blocks.h.stop - source_blocks.h.start
    for columns, products, upper_bound in zip(
        _source_weight_columns(model), (source_blocks.w_r, source_blocks.w_x), bounds, strict=True,
    ):
        if len(columns) != count or products.stop - products.start != count:
            raise ValueError("source product blocks must match R/X atom counts")
        for k, weight_column in enumerate(columns):
            _add_bounded_source_product(
                model, weight=weight_column,
                indicator=source_blocks.h.start + k, product=products.start + k,
                upper_bound=upper_bound,
            )


def _add_source_selection_constraints(
    model: L1Model,
    supports: tuple[IndexSupport, ...],
    source_blocks: SourceBlocks,
    *,
    require_nonempty: bool = True,
) -> None:
    """对同一个完整 h 添加已知支撑关系和全向量非空约束。

    固定模型中，最深选中 clade 表示源挂接点，并选中其全部已有祖先。
    不要求源 clade 与已有 clade 不同；h 本身不是 one-hot。
    扩展模型的 h 有 K+1 项：前 K 项间的关系已知，末项与它们的关系
    取决于 z，交给联合选择函数。非空约束始终作用于完整 h，因而允许
    仅末项为 1。require_nonempty=False 可显式允许根位置。
    """
    count = len(supports)
    atom_count = source_blocks.h.stop - source_blocks.h.start
    expected_count = count + int(model.z is not None)
    if atom_count != expected_count or atom_count != len(_source_weight_columns(model)[0]):
        raise ValueError("source path block must cover all current atom supports")
    if require_nonempty and atom_count == 0:
        raise ValueError("a non-root source requires at least one support")
    sets = [set(support) for support in supports]
    if any(not support for support in sets):
        raise ValueError("source supports must be nonempty")
    for a in range(count):
        h_a = source_blocks.h.start + a
        for b in range(a + 1, count):
            h_b = source_blocks.h.start + b
            if sets[a] < sets[b]:
                model.rows.add({h_a: 1.0, h_b: -1.0}, upper=0.0)
            elif sets[b] < sets[a]:
                model.rows.add({h_b: 1.0, h_a: -1.0}, upper=0.0)
            elif sets[a].isdisjoint(sets[b]):
                model.rows.add({h_a: 1.0, h_b: 1.0}, upper=1.0)
            else:
                raise ValueError("source supports must form a distinct laminar family")
    if require_nonempty:
        model.rows.add({source_blocks.h.start + k: 1.0 for k in range(atom_count)}, lower=1.0)


def _prepare_source_model_inputs(
    scenarios: Sequence[dict] | _PreparedScenarios,
    supports: Iterable[Iterable[int]],
    source_inputs: PreparedSourceInputs | None,
    *,
    r_upper_bound: float,
    x_upper_bound: float,
) -> tuple[
    _PreparedScenarios, tuple[IndexSupport, ...],
    tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]] | None, float, float,
]:
    """固定模型与扩展模型共用场景、支撑、幅值对齐及有限上界检查。"""
    prepared = _prepare_scenarios(scenarios)
    supports = normalize_supports(supports, prepared.n)
    r_upper_bound = _require_positive_finite("r_upper_bound", r_upper_bound)
    x_upper_bound = _require_positive_finite("x_upper_bound", x_upper_bound)
    time_indices = None if isinstance(scenarios, _PreparedScenarios) else tuple(
        scenario["P_terminal"].index
        if hasattr(scenario["P_terminal"], "columns") else None
        for scenario in scenarios
    )
    amplitudes = _copy_source_amplitudes(
        source_inputs,
        expected_lengths=tuple(block.shape[0] for block in prepared.p),
        time_indices=time_indices,
    )
    return prepared, supports, amplitudes, r_upper_bound, x_upper_bound


def build_fixed_source_model(
    scenarios: Sequence[dict] | _PreparedScenarios,
    supports: Iterable[Iterable[int]],
    source_inputs: PreparedSourceInputs | None,
    *,
    r_upper_bound: float,
    x_upper_bound: float,
    selection_constraints: SourceSelectionBuilder | None = None,
) -> tuple[L1Model, SourceBlocks | None]:
    """构建固定支撑的源模型，不执行求解；返回模型和源变量切片。

    默认联合拟合 R/X 和一条非根源路径。可用同签名 selection_constraints
    替代默认选择域。source_inputs 来自同组场景的 prepare_source_inputs；
    原始 DataFrame 会再次按时间标签对齐，_PreparedScenarios 无标签，
    其时间顺序由调用方保证。支撑使用终端位置下标。

    None/全零返回普通核心模型及 None，不调用选择函数、不增加源变量。
    高层求解器、搜索、refit、pruning 和验证尚未接入此模型。
    """
    prepared, supports, amplitudes, r_upper_bound, x_upper_bound = _prepare_source_model_inputs(
        scenarios, supports, source_inputs,
        r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound,
    )
    if amplitudes is None:
        return build_fixed_model(
            prepared, supports,
            r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound,
        ), None
    if not supports:
        raise ValueError("nonzero source requires at least one fixed support")
    selection = (_add_source_selection_constraints if selection_constraints is None
                 else selection_constraints)
    if not callable(selection):
        raise ValueError("selection_constraints must be callable or None")

    model = _base_model(
        prepared, len(supports),
        r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound,
    )
    source_blocks = _add_source_variables(
        model, len(supports),
        r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound,
    )
    _add_source_product_constraints(
        model, source_blocks,
        r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound,
    )
    if selection(model, supports, source_blocks) is not None:
        raise ValueError("selection_constraints must add rows and return None")

    def response_terms(current_model, output_index):
        atoms = [k for k, support in enumerate(supports) if output_index in support]
        return (
            {source_blocks.w_r.start + k: 1.0 for k in atoms},
            {source_blocks.w_x.start + k: 1.0 for k in atoms},
        )

    source_terms = _make_response_provider(amplitudes, response_terms)
    _add_absolute_residual_constraints(
        prepared, supports, model, source_terms=source_terms,
    )
    return model, source_blocks


def _add_source_extension_variables(
    model: L1Model,
    atom_count: int,
    *,
    r_upper_bound: float,
    x_upper_bound: float,
) -> SourceExtensionBlocks:
    """分配 atom_count=K+1 的完整源向量和 n 维支撑乘积。"""
    if model.z is None or model.new_r is None or model.new_x is None:
        raise ValueError("source extension requires a core atom-extension model")
    blocks = _add_source_variables(
        model, atom_count, r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound,
    )
    n = model.z.stop - model.z.start
    return SourceExtensionBlocks(
        h=blocks.h,
        w_r=blocks.w_r,
        w_x=blocks.w_x,
        v_r=model.variables.add("source_new_v_r", n, upper=r_upper_bound),
        v_x=model.variables.add("source_new_v_x", n, upper=x_upper_bound),
    )


def _add_source_extension_product_constraints(
    model: L1Model,
    source_blocks: SourceExtensionBlocks,
    *,
    r_upper_bound: float,
    x_upper_bound: float,
) -> None:
    """先建立完整 w=weight*h，再精确线性化 v[i]=w[K]*z[i]。

    h[K] 是同一个源向量的末项，无需另外创建路径位或 h[K]*z 的中间量。
    全部 h 与 z 仍是二进制变量；w/v 为连续辅助变量。
    普通负荷的 new_r*z[i]*z[j] 继续由核心负责。
    """
    if model.z is None or model.new_r is None or model.new_x is None:
        raise ValueError("source extension requires a core atom-extension model")
    r_upper_bound = _require_positive_finite("r_upper_bound", r_upper_bound)
    x_upper_bound = _require_positive_finite("x_upper_bound", x_upper_bound)
    _add_source_product_constraints(
        model, source_blocks,
        r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound,
    )
    n = model.z.stop - model.z.start
    if any(
        block.stop - block.start != n
        for block in (source_blocks.v_r, source_blocks.v_x)
    ):
        raise ValueError("source extension product shapes must match z")
    for i in range(n):
        z_i = model.z.start + i
        for weight, products, bound in (
            (source_blocks.w_r.stop - 1, source_blocks.v_r, r_upper_bound),
            (source_blocks.w_x.stop - 1, source_blocks.v_x, x_upper_bound),
        ):
            _add_bounded_source_product(
                model, weight=weight, indicator=z_i,
                product=products.start + i, upper_bound=bound,
            )


def _add_source_extension_selection_constraints(
    model: L1Model,
    supports: tuple[IndexSupport, ...],
    source_blocks: SourceExtensionBlocks,
    relations: slice,
) -> None:
    """利用核心三选一关系，约束完整 h 与本轮待优化支撑 z。

    K=len(supports)。h 的全部 K+1 项都是本轮变量，其中 h[K] 对应 z。
    变量列为 source_blocks.h.start+k，不存在先前路径的输入或固定值。
    builder 已建立已知 supports 间的祖先/不交约束、sum(h)>=1，以及
    核心的 z 非空/层状/禁止重复约束。本函数只补 z 所决定的路径关系，
    不分配变量或重复添加三选一约束，不求解 h，也不建立乘积与残差。

    relations.start+3*k 的三个位依次表示新支撑是 S_k 的子集、超集、
    不交；核心已排除重复支撑，因此前两者在可行解中是严格包含。
    对应条件分别为 h[K]<=h[k]、h[k]<=h[K]、h[K]+h[k]<=1。
    利用二进制变量的 0/1 界，将每个条件写成一条线性蕴含约束。
    K=0 时无需关系行，完整 h 的非空约束已强制唯一的 h[0]=1。
    """
    if model.z is None:
        raise ValueError("source extension selection requires a core z block")
    count = len(supports)
    if source_blocks.h.stop - source_blocks.h.start != count + 1:
        raise ValueError("source extension path must cover all K+1 atoms")
    if relations.stop - relations.start != 3 * count:
        raise ValueError("source extension requires three relation variables per support")

    rows = model.rows
    h_K = source_blocks.h.start + count
    for k in range(count):
        h_k = source_blocks.h.start + k
        subset = relations.start + 3 * k
        superset = subset + 1
        disjoint = subset + 2
        # 新 atom 在 S_k 下游：源经过新 atom 时，也必须经过 S_k。
        rows.add({h_K: 1.0, h_k: -1.0, subset: 1.0}, upper=1.0)
        # 新 atom 在 S_k 上游：源经过 S_k 时，也必须经过新 atom。
        rows.add({h_k: 1.0, h_K: -1.0, superset: 1.0}, upper=1.0)
        # 两个 atom 不交：单源路径不能同时经过两者。
        rows.add({h_K: 1.0, h_k: 1.0, disjoint: 1.0}, upper=2.0)


def build_extension_source_model(
    scenarios: Sequence[dict] | _PreparedScenarios,
    supports: Iterable[Iterable[int]],
    source_inputs: PreparedSourceInputs | None,
    *,
    r_upper_bound: float,
    x_upper_bound: float,
    selection_constraints: SourceExtensionSelectionBuilder | None = None,
) -> tuple[L1Model, SourceExtensionBlocks | None]:
    """构建同时优化完整 h、全部权重和一个新 atom 的 MILP。

    默认复用核心三选一关系建立 h/z 联动，也可用四参数回调替代。
    顺序为核心结构约束、源路径约束、源乘积、统一残差；所有变量在
    同一次 MILP 中联合求解。source_inputs 的对齐约定与固定模型相同。
    每次创建完整 h/w_r/w_x，长度 len(supports)+1，末项对应当前 z。
    不接收或固定上一轮的 h。None/全零返回核心模型；supports 可为空。
    返回模型和变量切片，不执行求解，也尚未接入高层贪心搜索流程。
    """
    prepared, supports, amplitudes, r_upper_bound, x_upper_bound = _prepare_source_model_inputs(
        scenarios, supports, source_inputs,
        r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound,
    )
    if amplitudes is None:
        return build_extension_model(
            prepared, supports,
            r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound,
        ), None
    selection = (_add_source_extension_selection_constraints if selection_constraints is None
                 else selection_constraints)
    if not callable(selection):
        raise ValueError("selection_constraints must be callable or None")

    model, atom_blocks = _base_extension_model(
        prepared, supports,
        r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound,
    )
    source_blocks = _add_source_extension_variables(
        model, len(supports) + 1, r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound,
    )
    _add_extension_constraints(
        prepared, supports, model, atom_blocks,
        r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound,
    )
    _add_source_selection_constraints(
        model, supports, source_blocks,
    )
    if selection(model, supports, source_blocks, atom_blocks.relations) is not None:
        raise ValueError("selection_constraints must add rows and return None")
    _add_source_extension_product_constraints(
        model, source_blocks,
        r_upper_bound=r_upper_bound, x_upper_bound=x_upper_bound,
    )

    def response_terms(current_model, output_index):
        atoms = [k for k, support in enumerate(supports) if output_index in support]
        r_terms = {source_blocks.w_r.start + k: 1.0 for k in atoms}
        x_terms = {source_blocks.w_x.start + k: 1.0 for k in atoms}
        r_terms[source_blocks.v_r.start + output_index] = 1.0
        x_terms[source_blocks.v_x.start + output_index] = 1.0
        return r_terms, x_terms

    _add_absolute_residual_constraints(
        prepared, supports, model,
        new_u_r_block=atom_blocks.u_r, new_u_x_block=atom_blocks.u_x,
        pair_lookup=atom_blocks.pair_lookup,
        source_terms=_make_response_provider(amplitudes, response_terms),
    )
    return model, source_blocks


__all__ = [
    "ExtensionSolution",
    "FixedSupportSolution",
    "IndexSupport",
    "L1Model",
    "LaminarL1PathPoint",
    "LaminarL1Result",
    "SolverDiagnostics",
    "SourceTermProvider",
    "SourceBlocks",
    "SourceSelectionBuilder",
    "SourceExtensionBlocks",
    "SourceExtensionSelectionBuilder",
    "build_extension_model",
    "build_fixed_model",
    "build_fixed_source_model",
    "build_extension_source_model",
    "make_source_response_terms",
]
