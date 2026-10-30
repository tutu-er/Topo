"""Publish metadata/fits only; private observed packets are never copied."""
import hashlib,json,shutil,zipfile,os
from pathlib import Path

BASE=Path(os.environ.get('WZZT_RANDOM_GATE_SCRATCH',r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a'))
REPO=Path(r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo')
STUDY=REPO/'research_experiments/wzzt_dr_selection'
PACKAGE=STUDY/'dr_random_observation_gate'
GEN=BASE/'wzzt_dr_random_observation_generation_v2'
RUN=BASE/'wzzt_dr_random_observation_execution_v1'
EVIDENCE=STUDY/'evidence/random_observation_gate_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):Path(p).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    if EVIDENCE.exists():raise FileExistsError('public evidence already retained')
    EVIDENCE.mkdir(parents=True)
    for origin,name in [('EXPERIMENT.md','EXPERIMENT.md'),('REPORT.md','REPORT.md'),('README.md','README.md'),
        ('independent_run_audit.py','independent_run_audit.py'),('independent_structure_summary.py','independent_structure_summary.py'),
        ('inspect_mechanisms.py','inspect_mechanisms.py'),('publish_evidence.py','publish_evidence.py')]:shutil.copy2(Path(__file__).parent/origin,PACKAGE/name)
    for name in ('evaluate_gate.py','test_evaluate_gate_pure.py','simple_root_audit.py'):
        shutil.copy2(BASE/'dr_random_eval_stage'/name,PACKAGE/name)
    shutil.copy2(BASE/'dr_random_protocol_audit_stage/audit_selected.py',PACKAGE/'audit_selected.py')
    contents=[]
    def add(path,arc):
        path=Path(path)
        if path.suffix.lower() in ('.pkl','.pickle','.npz','.csv','.rdata'):raise ValueError('private trajectory prohibited')
        contents.append((path,arc))
    for name in ('generation_protocol.json','generation_protocol.sha256.json','generation_summary.json','config.json','evaluator_truth.json'):
        add(GEN/name,'generation/'+name)
    for folder in ('discovery','control_records'):
        for p in (GEN/folder).glob('*.json'):add(p,'generation/'+folder+'/'+p.name)
    for name in ('protocol.json','protocol.sha256.json','summary.json','environment.json','records.json',
                 'independent_run_audit.json','independent_structure_summary.json','mechanism_diagnostics.json'):
        add(RUN/name,'execution/'+name)
    for folder in ('history','records'):
        for p in (RUN/folder).glob('*.json'):add(p,'execution/'+folder+'/'+p.name)
    for p in (BASE/'evaluation_v1').glob('*.json'):add(p,'evaluation/'+p.name)
    for p in (BASE/'dr_random_protocol_audit_stage').glob('*'):
        if p.is_file() and p.suffix in ('.py','.json'):add(p,'independent_protocol_audit/'+p.name)
    for name in ('generation_protocol.json','generation_protocol.sha256.json','failure_receipt.json'):
        p=BASE/'wzzt_dr_random_observation_generation_v1'/name
        add(p,'failed_generation_v1/'+name)
    for p in (BASE/'wzzt_dr_random_observation_generation_v1/source_failed').glob('*.py'):
        add(p,'failed_generation_v1/source/'+p.name)
    frozen={}
    for cfg in (read(GEN/'generation_protocol.json'),read(RUN/'protocol.json')):frozen.update(cfg['source_sha256'])
    for i,(path,h) in enumerate(sorted(frozen.items())):
        p=Path(path)
        if sha(p)!=h:raise ValueError('frozen source drift '+path)
        rel=str(p.relative_to(REPO)).replace('\\','/')
        add(p,'frozen_source/'+rel)
    for p in PACKAGE.iterdir():
        if p.is_file() and p.suffix in ('.py','.md'):add(p,'delivery_source/'+p.name)
    for name in ('tests_receipt.json',):
        p=BASE/'dr_random_experiment_stage'/name
        if p.exists():add(p,'tests/'+name)
    add(BASE/'dr_random_eval_stage/pure_tests.json','tests/evaluator_pure_tests.json')
    packed=EVIDENCE/'metadata_fits_and_audits.zip'
    with zipfile.ZipFile(packed,'w',zipfile.ZIP_DEFLATED) as z:
        for p,arc in contents:z.write(p,arc)
    expected={arc:sha(p) for p,arc in contents}
    with zipfile.ZipFile(packed) as z:
        if len(z.namelist())!=len(expected):raise AssertionError('duplicate archive entry')
        for arc,h in expected.items():
            if hashlib.sha256(z.read(arc)).hexdigest()!=h:raise AssertionError('archive readback mismatch '+arc)
    manifest=dict(status='verified_metadata_and_fits_only',archive=packed.name,archive_sha256=sha(packed),
        bytes=packed.stat().st_size,files=len(expected),entry_sha256=expected,
        private_observed_packets_included=False,household_or_aggregate_trajectories_included=False,
        independent_structure_evaluation_mappings=408,main_DR_innovation_achieved=False,
        candidate_extension_incremental_gain_verified=False,solver_run='one bounded run;759 LP;28/28 snapshots',
        failed_generation_v1_retained=True,production_defaults_modified=False,
        private_recovery_roots=[str(GEN),str(RUN),str(BASE/'evaluation_v1')])
    write(EVIDENCE/'manifest.json',manifest)
    for origin,name in ((GEN/'generation_summary.json','generation_summary.json'),(RUN/'summary.json','execution_summary.json'),
        (RUN/'independent_run_audit.json','independent_run_audit.json')):shutil.copy2(origin,EVIDENCE/name)
    print(json.dumps(dict(status=manifest['status'],files=manifest['files'],bytes=manifest['bytes'],sha256=manifest['archive_sha256'])))

if __name__=='__main__':main()
