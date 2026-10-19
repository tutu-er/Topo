"""Pure evidence verification; two prespecified saved-state AC replays only."""
from __future__ import annotations
import base64
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from research_experiments.mobile_ev_mle import ev_mobile_v3_evidence as evidence
from research_experiments.mobile_ev_mle import ev_mobile_v3_summary as summary
from research_experiments.mobile_ev_mle import ev_mobile_v3_physics_audit as physics


def main():
    root=Path('outputs/mobile_v3_pilot2_delivery_v1')
    output=root/'delivery_validation_v1.json'
    if output.exists():raise FileExistsError(output)
    manifest=evidence._read(root/'manifest.json');frozen=evidence._read(root/'freeze.json')
    if manifest['status']!='complete':raise ValueError('pack incomplete')
    if manifest['policy_id']!=frozen['policy_id']:raise ValueError('manifest/freeze policy differs')
    evidence._check_snapshot(root,frozen)
    if len(frozen['source_sha256'])!=122:raise ValueError('expected 122 frozen sources')
    current=[]
    module=Path(evidence.__file__).parent
    for name,expected in frozen['source_sha256'].items():
        file=module.parents[1]/name if name.startswith('rnj_wzzt_core/') else module/name
        actual=sha256(file.read_bytes()).hexdigest()
        current.append(dict(name=name,expected=expected,actual=actual,equal=expected==actual))
    if not all(r['equal'] for r in current):raise ValueError('current scientific source differs from freeze')
    baseline=evidence._read(root/'physics_eval_v1.json')
    baseline_rows={r['spec']['index']:r for r in baseline['case_results']}
    development=evidence._read(root/'attachments'/'mobile_v3_evidence_dev2_roundtrip_v1.json')
    development_rows={r['spec']['index']:r for r in development['cases']}
    rows=[];method_count=0;selected_audits=[]
    seen=set()
    audit_keys=('success','complete_physical_coverage','expected','metrics','errors','missing','states','methods','recorded_failures')
    for saved in manifest['records']:
        identity=(saved['phase'],saved['spec']['index'])
        if identity in seen:raise ValueError('duplicate case')
        seen.add(identity)
        file=root/saved['file'];payload=file.read_bytes()
        if sha256(payload).hexdigest()!=saved['compact_sha256']:raise ValueError('compact file bytes differ')
        compact=evidence._read(file);restored=evidence.restore(compact)
        original_file=Path(compact['source_record']['raw_path']);original_payload=original_file.read_bytes()
        raw_sha=sha256(original_payload).hexdigest()
        if raw_sha!=saved['raw_sha256'] or raw_sha!=compact['source_record']['raw_sha256']:
            raise ValueError('original raw hash differs')
        original=evidence._read(original_file)
        if evidence._hash(original)!=compact['source_record']['raw_object_sha256']:
            raise ValueError('original object hash differs')
        if restored['spec']!=saved['spec'] or restored['policy_id']!=frozen['policy_id']:
            raise ValueError('restored identity differs')
        methods={}
        for name in summary.METHODS:
            a=summary.project_method(original,name,candidate_count=4)
            b=summary.project_method(restored,name,candidate_count=4)
            equal=a==b
            methods[name]=dict(equal=equal,raw_sha256=evidence._hash(a),restored_sha256=evidence._hash(b))
            method_count+=1
            if not equal:raise ValueError('restored method summary differs: '+str(identity)+'/'+name)
        row=dict(spec=saved['spec'],raw_sha256=raw_sha,compact_sha256=saved['compact_sha256'],
                 summary_methods=methods,all_eight_equal=True,exclusion_count=len(compact['exclusions']),
                 retained_failure_nodes=len(compact['failure_evidence']))
        if saved['phase']=='development':
            old=development_rows[saved['spec']['index']]
            if old['raw_sha256']!=raw_sha or not old['success'] or not old['physical_replay_equal']:
                raise ValueError('development full physical roundtrip not applicable to these bytes')
            row['physical_check']='prior complete development roundtrip; raw bytes match'
        elif saved['spec']['index'] in (0,8):
            actual=physics.audit_record(restored)
            old=baseline_rows[saved['spec']['index']]
            if old['input_sha256']!=raw_sha:raise ValueError('raw evaluation physical-audit identity differs')
            equivalent=all(actual.get(k)==old.get(k) for k in audit_keys)
            if not actual['success'] or not equivalent:raise ValueError('selected restored physical audit differs')
            selected_audits.append(dict(index=saved['spec']['index'],raw_sha256=raw_sha,
                equal_to_full_raw_physics_audit=equivalent,audit=actual))
            row['physical_check']='new compact restore physical replay, predeclared evaluation index'
        else:row['physical_check']='full raw independently audited; no repeated compact AC requested'
        rows.append(row)
        print('VERIFIED',identity,'methods',len(methods),flush=True)
    counts=Counter(r['spec']['phase'] for r in rows)
    if dict(counts)!={'development':4,'evaluation':16} or method_count!=160:
        raise ValueError('20-case/160-method coverage differs')
    if sorted(r['index'] for r in selected_audits)!=[0,8]:raise ValueError('predeclared physical subset differs')
    files=[p for p in root.rglob('*') if p.is_file()]
    largest=max(files,key=lambda p:p.stat().st_size)
    result=dict(success=True,policy_id=frozen['policy_id'],case_count=20,phase_counts=dict(counts),
        method_summary_comparisons=method_count,all_method_summaries_equal=True,
        source_snapshot_original_bytes_verified=122,current_scientific_sources_verified=122,
        scientific_source_mismatches=[],source_sha256=frozen['source_sha256'],
        manifest_sha256=sha256((root/'manifest.json').read_bytes()).hexdigest(),records=rows,
        selected_evaluation_restore_physical_indices=[0,8],selected_physical_audits=selected_audits,
        development_full_roundtrip_raw_hash_match_count=4,
        raw_gzip_bytes=manifest['raw_gzip_bytes'],compact_gzip_bytes=manifest['compact_gzip_bytes'],
        compression_fraction=manifest['compact_gzip_bytes']/manifest['raw_gzip_bytes'],
        directory_bytes_before_this_report=sum(p.stat().st_size for p in files),file_count_before_this_report=len(files),
        largest_file=str(largest.relative_to(root)),largest_file_bytes=largest.stat().st_size,
        tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='compact evidence equivalence and saved-state replay; no new planning, fitting or policy evaluation')
    evidence._write(output,result)
    print(json.dumps({k:result[k] for k in ('success','case_count','method_summary_comparisons','compact_gzip_bytes',
                                          'directory_bytes_before_this_report','largest_file','largest_file_bytes')}),flush=True)


if __name__=='__main__':main()
