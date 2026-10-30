"""Export frozen metadata/fits and audits; observed/derived trajectory arrays stay private."""
import hashlib,json,os,shutil,zipfile
from pathlib import Path
BASE=Path(os.environ.get('WZZT_REFERENCE_NOISE_SCRATCH',r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a'))
REPO=Path(os.environ.get('WZZT_REPO_ROOT',r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo'))
STUDY=REPO/'research_experiments/wzzt_dr_selection';PACKAGE=STUDY/'dr_reference_noise'
GEN=BASE/'wzzt_dr_reference_noise_generation_v1';RUN=BASE/'wzzt_dr_reference_noise_execution_v1'
PUB=STUDY/'evidence/reference_noise_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    if PUB.exists():raise FileExistsError('public evidence already retained')
    PUB.mkdir(parents=True)
    sources=[(Path(__file__).parent/name,name) for name in ('REPORT.md','README.md','independent_run_audit.py','independent_structure_summary.py','publish_evidence.py')]
    sources += [(BASE/'dr_reference_noise_audit_stage/audit.py','audit_reference_noise.py'),
                (BASE/'dr_reference_noise_audit_stage/interpretation.py','interpretation.py'),
                (BASE/'dr_reference_noise_decomposition_audit/audit.py','audit_decomposition.py')]
    for src,name in sources:shutil.copy2(src,PACKAGE/name)
    contents=[]
    def add(p,arc):
        p=Path(p)
        if p.suffix.lower() in ('.pkl','.pickle','.npz','.csv','.rdata'):raise ValueError('trajectory export prohibited')
        contents.append((p,arc))
    for name in ('generation_protocol.json','generation_protocol.sha256.json','generation_summary.json','config.json','evaluator_truth.json'):
        add(GEN/name,'generation/'+name)
    for folder in ('discovery','control_records'):
        for p in (GEN/folder).glob('*.json'):add(p,'generation/'+folder+'/'+p.name)
    for name in ('protocol.json','protocol.sha256.json','summary.json','environment.json','records.json','independent_run_audit.json','independent_structure_summary.json'):
        add(RUN/name,'execution/'+name)
    for folder in ('history','records'):
        for p in (RUN/folder).glob('*.json'):add(p,'execution/'+folder+'/'+p.name)
    for folder,arc in [('dr_reference_noise_audit_stage','independent_audit'),('dr_reference_noise_decomposition_audit','decomposition_audit')]:
        for p in (BASE/folder).iterdir():
            if p.is_file() and p.suffix.lower() in ('.json','.md','.py'):add(p,arc+'/'+p.name)
    add(BASE/'dr_reference_noise_driver_stage/pure_tests.json','tests/driver_attempts.json')
    add(Path(__file__).parent/'test_receipt.json','tests/final_eight_tests.json')
    frozen={}
    for cfg in (read(GEN/'generation_protocol.json'),read(RUN/'protocol.json')):frozen.update(cfg['source_sha256'])
    for p,h in sorted(frozen.items()):
        p=Path(p)
        if sha(p)!=h:raise ValueError('frozen source drift '+str(p))
        add(p,'frozen_source/'+str(p.relative_to(REPO)).replace('\\','/'))
    for p in PACKAGE.iterdir():
        if p.is_file() and p.suffix in ('.py','.md'):add(p,'delivery_source/'+p.name)
    archive=PUB/'metadata_fits_and_audits.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p,arc in contents:z.write(p,arc)
    expected={arc:sha(p) for p,arc in contents}
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist())!=len(expected):raise AssertionError('archive duplicate entries')
        for arc,h in expected.items():
            if hashlib.sha256(z.read(arc)).hexdigest()!=h:raise AssertionError('archive readback mismatch '+arc)
    manifest=dict(status='verified',archive=archive.name,archive_sha256=sha(archive),bytes=archive.stat().st_size,
        files=len(expected),entry_sha256=expected,private_PQV_or_derived_trajectory_arrays_exported=False,
        main_DR_price_innovation_proven=False,formal_RNJ_optional_root_fitted=True,
        case_units=36,unique_generating_model_noise_combinations=30,case_policy_mappings=72,unique_snapshots=33,
        LP_calls=1222,all_paths_scanned=True,impedance_scale=1.,new_solver_calls_in_audits=0,
        same_prediction_different_contraction_witness='decomposition_audit/receipt_v2.json',
        original_witness_roundoff_version_retained=True,old_high_or_exact_batches_rerun=False)
    (PUB/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    for src,name in [(GEN/'generation_summary.json','generation_summary.json'),(RUN/'summary.json','execution_summary.json'),
        (RUN/'independent_run_audit.json','independent_run_audit.json'),
        (BASE/'dr_reference_noise_audit_stage/mechanism_summary_v1.json','mechanism_summary.json'),
        (BASE/'dr_reference_noise_decomposition_audit/receipt_v2.json','decomposition_witness.json')]:shutil.copy2(src,PUB/name)
    print(json.dumps({k:manifest[k] for k in ('status','archive_sha256','bytes','files')}))
if __name__=='__main__':main()
