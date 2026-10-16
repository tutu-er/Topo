"""Read-only self-profile drift audit; no fit, case generator or future truth."""
from __future__ import annotations
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import math


def _read(path):
    path=Path(path);data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)


def _numbers(value):
    if isinstance(value,list):
        return [item for child in value for item in _numbers(child)]
    return [float(value)]


def audit_saved(path):
    path=Path(path);raw=_read(path)
    trained={row['name']:row for row in raw['training_result']['fits']}
    caches=[('reporting',raw.get('reporting_cache',{}))]+[(family,
        raw.get('families',{}).get(family,{}).get('cache',{})) for family in ('guarded','aligned')]
    profiles={};duplicates=0;conflicts=[]
    for origin,cache in caches:
        for key,profile in cache.items():
            identity=(profile.get('objective_id'),key)
            if identity in profiles:
                duplicates+=1
                old=profiles[identity][1]
                def self_values(p):
                    return {r['a']:{k:r.get('fits',{}).get(r['a'],{}).get(k)
                        for k in ('success','loss','gradient_per_kw','gradient_reliable')}
                        for r in p.get('references',[])}
                if self_values(old)!=self_values(profile):
                    conflicts.append(dict(action_key=key,first=profiles[identity][0],second=origin))
            else:
                profiles[identity]=(origin,profile)
    rows=[]
    for (objective_id,key),(origin,profile) in profiles.items():
        refs={r['a']:r for r in profile.get('references',[])}
        for a,training in trained.items():
            own=refs.get(a,{}).get('fits',{}).get(a,{})
            loss=own.get('loss');price=own.get('gradient_per_kw')
            values=None if price is None else _numbers(price)
            finite=values is not None and all(math.isfinite(v) for v in values)
            blocks=own.get('blocks',{})
            Dloss=blocks.get('training',blocks.get('observed',{})).get('loss')
            Eloss=blocks.get('future',{}).get('loss')
            training_half=.5*training['loss']
            own_half=None if loss is None else .5*loss
            rows.append(dict(a=a,action_key=key,objective_id=objective_id,first_cache=origin,
                profile_success=profile.get('success'),own_success=own.get('success'),
                own_gradient_reliable=own.get('gradient_reliable'),
                training_half_SSE=training_half,self_joint_half_SSE=own_half,
                self_D_half_SSE=None if Dloss is None else .5*Dloss,
                self_future_half_SSE=None if Eloss is None else .5*Eloss,
                drift_half_SSE=None if own_half is None else own_half-training_half,
                relative_drift=None if own_half is None else (own_half-training_half)/(1+abs(training_half)),
                own_gradient_L2=None if not finite else math.sqrt(sum(v*v for v in values)),
                own_gradient_Linf=None if not finite else max((abs(v) for v in values),default=0.),
                failure=own.get('failure',refs.get(a,{}).get('failure')),
                selected_origin=own.get('selected_origin'),optimizer_error=own.get('optimizer_error')))
    summary={}
    for a in trained:
        group=[row for row in rows if row['a']==a]
        drift=[row['drift_half_SSE'] for row in group if row['drift_half_SSE'] is not None]
        norm=[row['own_gradient_L2'] for row in group if row['own_gradient_L2'] is not None]
        summary[a]=dict(profile_slots=len(group),own_success=sum(row['own_success'] is True for row in group),
            missing_loss=len(group)-len(drift),training_half_SSE=.5*trained[a]['loss'],
            drift_min=None if not drift else min(drift),drift_max=None if not drift else max(drift),
            max_absolute_drift=None if not drift else max(abs(v) for v in drift),
            drift_range_across_actions=None if not drift else max(drift)-min(drift),
            own_gradient_L2_max=None if not norm else max(norm))
    return dict(input_path=str(path.resolve()),input_sha256=sha256(path.read_bytes()).hexdigest(),
        unique_action_objective_profiles=len(profiles),duplicate_cache_occurrences=duplicates,
        cold_cache_self_conflicts=conflicts,rows=rows,by_reference=summary,
        expected_reference_slots=len(profiles)*len(trained),uses_truth=False,
        uses_actual_future=False,optimizer_calls=0,AC_calls=0,
        scope='attained self joint half-SSE versus same-D training attainment; no global-minimum claim',
        helper_sha256=sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result=audit_saved(args.input);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result['by_reference'],ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
