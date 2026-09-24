"""Render audited laminar L1-MILP CSV/JSON results as LaTeX fragments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_SMALL6 = Path("outputs/laminar_l1_milp_small6_final/summary.csv")
DEFAULT_CASE33 = Path("outputs/laminar_l1_milp_case33_bound2_final/metrics.json")
DEFAULT_SMALL6_TEX = Path("docs/laminar_l1_milp_results_table.tex")
DEFAULT_CASE33_TEX = Path("docs/laminar_l1_case33_results_table.tex")


def _escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _sci(value: object) -> str:
    return f"{float(value):.3e}"


def _fixed(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _yes_no(value: object) -> str:
    if isinstance(value, str):
        checked = value.strip().lower() in {"1", "true", "yes"}
    else:
        checked = bool(value)
    return "是" if checked else "否"


def _stop_reason(value: object) -> str:
    names = {
        "no_significant_one_atom_gain": "无显著单原子增益",
        "laminar_family_limit": "达到 laminar 上限",
        "extension_not_proven_optimal": "扩展未获最优证书",
        "max_atoms": "达到原子数上限",
    }
    text = str(value)
    return names.get(text, _escape(text))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def render_small6(summary_path: Path, output_path: Path) -> None:
    rows = _read_csv(summary_path)
    if not rows:
        raise ValueError(f"small6 summary is empty: {summary_path}")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{small6 最终拟合、固定训练截距验证与测试结果}",
        r"\label{tab:small6-fit}",
        r"\scriptsize",
        r"\begin{tabularx}{\textwidth}{lcc>{\raggedright\arraybackslash}Xrrr r}",
        r"\toprule",
        r"制度 & 路径点 & 选中原子 & 停止原因 & 训练 MAE & 验证 MAE & 测试 MAE & 总时间/s \\",
        r"\midrule",
    ]
    display_names = {
        "noiseless": "无噪声",
        "laplace_outliers": "Laplace+异常值",
    }
    for row in rows:
        lines.append(
            (r"{} & {} & {} & {} & {} & {} & {} & {} \\").format(
                display_names.get(row["regime"], _escape(row["regime"])),
                int(row["selected_path_index"]),
                int(row["selected_atom_count"]),
                _stop_reason(row["stop_reason"]),
                _sci(row["train_mae"]),
                _sci(row["validation_mae_fixed_training_intercept"]),
                _sci(row["test_mae_fixed_training_intercept"]),
                _fixed(row["fit_wall_seconds"], 2),
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{small6 矩阵、clade 与求解证书审计}",
            r"\label{tab:small6-structure}",
            r"\scriptsize",
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r"制度 & $R$ 相对 Fro. & $X$ 相对 Fro. & 匹配/真值 & Precision & Recall & F1 & 全部扩展有证书 \\",
            r"\midrule",
        ]
    )
    for row in rows:
        lines.append(
            (r"{} & {} & {} & {}/{} & {} & {} & {} & {} \\").format(
                display_names.get(row["regime"], _escape(row["regime"])),
                _sci(row["R_matrix_relative_frobenius_error"]),
                _sci(row["X_matrix_relative_frobenius_error"]),
                int(row["matched_nontrivial_clade_count"]),
                int(row["true_nontrivial_clade_count"]),
                _fixed(row["nontrivial_clade_precision"]),
                _fixed(row["nontrivial_clade_recall"]),
                _fixed(row["nontrivial_clade_f1"]),
                _yes_no(row["all_extension_attempts_certified_optimal"]),
            )
        )
    maximum_absolute_gap = max(
        float(row["maximum_extension_absolute_gap"]) for row in rows
    )
    maximum_relative_gap = max(
        float(row["maximum_extension_mip_gap"]) for row in rows
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\par\smallskip 最大报告绝对 gap 为 "
                + _sci(maximum_absolute_gap)
                + r"，最大相对 MIP gap 为 "
                + _sci(maximum_relative_gap)
                + r"；两个制度均未触及系数上界。"
            ),
            r"\end{table}",
            "",
            r"\FloatBarrier",
            (
                r"\noindent\textit{数据来源：}\path{"
                + summary_path.as_posix()
                + r"}。验证和首要测试指标均固定使用训练 LP 返回的截距。"
            ),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def render_case33(metrics_path: Path, output_path: Path) -> None:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = payload["cases"]
    config = payload["config"]
    if not rows:
        raise ValueError(f"case33 metrics contain no cases: {metrics_path}")
    limits = {
        "aggregate_leaves": config["aggregate_time_limit_seconds_per_solve"],
        "hybrid_full": config["hybrid_time_limit_seconds_per_solve"],
    }
    display_names = {
        "aggregate_leaves": "四物理叶聚合",
        "hybrid_full": "32 终端 hybrid",
    }
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{IEEE case33 派生算例的求解状态与路径结果}",
        r"\label{tab:case33-status}",
        r"\scriptsize",
        r"\begin{tabularx}{\textwidth}{lrrrr>{\raggedright\arraybackslash}Xccr}",
        r"\toprule",
        r"算例 & $n$ & 时限/s & 接受原子 & 选中原子 & 停止原因 & 状态全 0 & 全部有证书 & 总时间/s \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            (r"{} & {} & {} & {} & {} & {} & {} & {} & {} \\").format(
                display_names.get(row["case"], _escape(row["case"])),
                int(row["terminal_count"]),
                _fixed(limits[row["case"]], 0),
                int(row["accepted_path_length_excluding_zero"]),
                int(row["selected_atom_count"]),
                _stop_reason(row["stop_reason"]),
                _yes_no(row["all_attempts_status_zero"]),
                _yes_no(row["all_attempts_certified_optimal"]),
                _fixed(row["fit_wall_seconds"], 2),
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{IEEE case33 派生算例的外推、矩阵与 clade 指标}",
            r"\label{tab:case33-quality}",
            r"\scriptsize",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"算例 & 训练 MAE & 验证 MAE & 测试 MAE & $R$ 相对 Fro. & $X$ 相对 Fro. & F1 \\",
            r"\midrule",
        ]
    )
    for row in rows:
        lines.append(
            (r"{} & {} & {} & {} & {} & {} & {} \\").format(
                display_names.get(row["case"], _escape(row["case"])),
                _sci(row["train_mae"]),
                _sci(row["validation_mae_fixed_training_intercept"]),
                _sci(row["test_mae_fixed_training_intercept"]),
                _sci(row["R_matrix_relative_frobenius_error"]),
                _sci(row["X_matrix_relative_frobenius_error"]),
                _fixed(row["nontrivial_support_f1"]),
            )
        )
    hybrid = next((row for row in rows if row["case"] == "hybrid_full"), None)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\FloatBarrier",
            (
                r"\noindent 四叶聚合算例的 7 次扩展均取得严格证书，矩阵达到机器精度且两个"
                r"非平凡 clade 全部匹配。"
            ),
        ]
    )
    if hybrid is not None:
        lines.append(
            (
                r"32 终端算例在上界 "
                + _fixed(hybrid["coefficient_bound"], 0)
                + r"、单轮 "
                + _fixed(limits["hybrid_full"], 0)
                + r" 秒内状态为 "
                + _escape(",".join(map(str, hybrid["attempt_statuses"])))
                + r"，最大报告 gap 为 "
                + _fixed(hybrid["maximum_reported_mip_gap"])
                + r"。实现按设计拒绝未获证书的 incumbent，故空模型指标只表示规模压力测试，"
                r"不能解释为已完成的 32 终端回归结果。"
            )
        )
    lines.extend(
        [
            (
                r"\par\smallskip\textit{数据来源：}\path{"
                + metrics_path.as_posix()
                + r"}；真值仅用于生成响应和事后评分，未用于候选支撑搜索。"
            ),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small6-summary", type=Path, default=DEFAULT_SMALL6)
    parser.add_argument("--case33-metrics", type=Path, default=DEFAULT_CASE33)
    parser.add_argument("--small6-tex", type=Path, default=DEFAULT_SMALL6_TEX)
    parser.add_argument("--case33-tex", type=Path, default=DEFAULT_CASE33_TEX)
    args = parser.parse_args()
    render_small6(args.small6_summary, args.small6_tex)
    render_case33(args.case33_metrics, args.case33_tex)


if __name__ == "__main__":
    main()
