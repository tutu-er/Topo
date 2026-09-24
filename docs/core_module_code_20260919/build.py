"""Freeze actual source and produce checked LaTeX code references (stdlib only)."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORE = ROOT / 'rnj_wzzt_core'
DESCRIPTIONS = {
    'run.py': '普通命令行入口',
    'rnj_wzzt/cli.py': '普通入口配置与高级参数解析',
    'rnj_wzzt/pipeline.py': '数据准备、RNJ、收缩、MILP、输出的流程编排',
    'rnj_wzzt/reporting.py': '真值评价、候选审计与求解器记录',
    'rnj_wzzt/data/paper_style_case_bank.py': '四算例注册表、三种网络及资源配置',
    'rnj_wzzt/data/paper_style_terminal_lv.py': 'paper15 网络及异质终端资源配置',
    'rnj_wzzt/models/network.py': '网络对象、图转换与校验入口',
    'rnj_wzzt/models/ac_powerflow.py': '径向 AC 后向前向扫掠潮流',
    'rnj_wzzt/models/lin_distflow.py': '标幺换算、理想共享路径 R/X 与距离',
    'rnj_wzzt/scenario/settings.py': '物理工况、根观测与量测噪声独立配置',
    'rnj_wzzt/scenario/profiles.py': '负荷、光伏、风电、小型机组及无功曲线',
    'rnj_wzzt/scenario/simulation.py': 'AC 仿真、噪声、训练验证复制与子采样',
    'rnj_wzzt/scenario/validate_scenario.py': '径向、末端负荷、可观测性与守恒检查',
    'rnj_wzzt/estimation/preprocessing.py': '平方电压降、去均值及其他时序变换',
    'rnj_wzzt/estimation/multiscenario.py': '标签对齐、消截距、回归接口与诊断',
    'rnj_wzzt/estimation/constrained_least_squares.py': '对称非负有序 R/X 凸 QP 数值求解',
    'rnj_wzzt/estimation/matrix_constraints.py': '矩阵可行性修正和树度量诊断',
    'rnj_wzzt/estimation/recipes.py': '历史预处理配方导入路径的兼容层',
    'rnj_wzzt/graph/sensitivity_geometry.py': 'R/X 距离归一化及 RNJ 共享路径输入',
    'rnj_wzzt/graph/rooted_neighbor_joining.py': '按共享路径递归合并的一般树 RNJ',
    'rnj_wzzt/graph/bootstrap.py': '循环分块重采样、边界块及不相交筛选',
    'rnj_wzzt/graph/rooted_hierarchy.py': 'clade 转换、伪终端反嵌入、局部重辨识工具',
    'rnj_wzzt/estimation/laminar_l1_milp.py': '固定支持 LP、一步 MILP、路径与验证选择',
}

def tex(value: str) -> str:
    mapping = {'\\': r'\textbackslash{}', '_': r'\_', '%': r'\%', '&': r'\&', '#': r'\#', '{': r'\{', '}': r'\}'}
    return ''.join(mapping.get(c, c) for c in value)

def file_inventory(path: Path) -> dict:
    raw = path.read_bytes()
    text = raw.decode('utf-8-sig')
    symbols = []
    if path.suffix == '.py':
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.append({'name': node.name, 'line': node.lineno, 'end': node.end_lineno, 'kind': type(node).__name__})
                if isinstance(node, ast.ClassDef):
                    for method in node.body:
                        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            symbols.append({'name': node.name + '.' + method.name, 'line': method.lineno, 'end': method.end_lineno, 'kind': 'method'})
    return {'path': path.relative_to(CORE).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest(), 'lines': len(text.splitlines()), 'symbols': symbols}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    files = [CORE / 'run.py', *sorted((CORE / 'rnj_wzzt').rglob('*.py')), CORE / 'pyproject.toml']
    inventory = [file_inventory(p) for p in files]
    manifest = HERE / 'source_manifest.json'
    if args.check:
        original = json.loads(manifest.read_text(encoding='utf-8'))['files']
        if original != inventory:
            raise SystemExit('Current core differs from documented snapshot; review required.')
        for item in original:
            raw = (HERE / 'source_snapshot' / item['path']).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == item['sha256'], item['path']
        print('Source and snapshot hashes match: %d files.' % len(files))
        return
    for p in files:
        dest = HERE / 'source_snapshot' / p.relative_to(CORE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    payload = {'review_date': '2026-09-19', 'scope': 'rnj_wzzt_core/run.py + rnj_wzzt/*.py + pyproject.toml', 'files': inventory}
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    generated = HERE / 'generated'
    generated.mkdir(exist_ok=True)
    refs, index, modules = [], [], []
    known_ids = set()
    module_count = 0
    for item in inventory:
        p = item['path']
        if not p.endswith('.py'):
            continue
        module_id = p.removesuffix('.py').replace('/', '.').removeprefix('rnj_wzzt.')
        module_count += 1
        desc = DESCRIPTIONS.get(p, '包初始化（无独立算法定义）')
        modules.append(r'\nolinkurl{%s} & %d & %s \\' % (p, item['lines'], desc))
        if not item['symbols']:
            continue
        index.append(r'\Needspace{8\baselineskip}\subsection{\texorpdfstring{\nolinkurl{%s}}{%s}}\label{module:%s}' % (p, tex(p), module_id))
        index.append(desc + '。所有行号对应随附源码快照。')
        index.append(r'\begin{longtable}{P{0.73\textwidth}P{0.18\textwidth}}\toprule 符号（类或函数） & 原始行号\\\midrule\endhead')
        for sym in item['symbols']:
            key = module_id + '.' + sym['name']
            known_ids.add(key)
            refs.append(r'\expandafter\def\csname coderef@%s\endcsname{\hyperref[module:%s]{\nolinkurl{%s}}，第 %d--%d 行（\nolinkurl{%s}）}' % (key, module_id, sym['name'], sym['line'], sym['end'], p))
            index.append(r'\nolinkurl{%s} & %d--%d \\' % (sym['name'], sym['line'], sym['end']))
        index.append(r'\bottomrule\end{longtable}')
    (generated / 'references.tex').write_text('\n'.join(refs), encoding='utf-8')
    (generated / 'index.tex').write_text('\n'.join(index), encoding='utf-8')
    module_table = [r'\begin{longtable}{P{8.0cm}P{0.8cm}P{6.0cm}}', r'\toprule 核心相对路径 & 行数 & 功能\\\midrule\endhead', *modules, r'\bottomrule\end{longtable}']
    (generated / 'modules.tex').write_text('\n'.join(module_table) + '\n', encoding='utf-8')
    main_text = (HERE / 'main.tex').read_text(encoding='utf-8')
    requested = set(re.findall(r'\\CodeRef\{([^}]+)\}', main_text))
    missing = requested - known_ids
    if missing:
        raise SystemExit('Missing symbols: ' + str(sorted(missing)))
    lines_by_file = {x['path']: x['lines'] for x in inventory}
    excerpts = re.findall(r'\\SourceCode\{([^}]+)\}\{(\d+)\}\{(\d+)\}', main_text)
    for file, start, end in excerpts:
        assert 1 <= int(start) <= int(end) <= lines_by_file[file], (file, start, end)
    summary = {'python_files': module_count, 'implementation_files_excluding_init': sum(1 for x in inventory if x['path'].endswith('.py') and not x['path'].endswith('__init__.py')), 'python_lines': sum(x['lines'] for x in inventory if x['path'].endswith('.py')), 'indexed_symbols': len(known_ids), 'checked_references': len(requested), 'checked_excerpts': len(excerpts)}
    (generated / 'reference_check.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (generated / 'counts.tex').write_text('\n'.join(r'\newcommand{\%s}{%d}' % (key.replace('_', '').title(), val) for key, val in summary.items()), encoding='utf-8')
    print(json.dumps(summary))

if __name__ == '__main__':
    main()
