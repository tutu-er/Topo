"""Replay one preserved stage-16 history with both default reviewers."""
from pathlib import Path
import hashlib,json,subprocess,types
from research_experiments.mobile_ev_mle.information_fit_recovery import read_json,write_json
from research_experiments.mobile_ev_mle import ev_original_location_review as current

stage=Path(__file__).resolve().parent
repo=Path(r'C:\Users\23761\.codex\worktrees\36a1\Topo')
git=r'C:\Users\23761\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe'
rel='research_experiments/mobile_ev_mle/ev_original_location_review.py'
before=subprocess.check_output([git,'-C',str(repo),'show','776280b:'+rel])
old=types.ModuleType('research_experiments.mobile_ev_mle._stage3_location_review')
old.__package__='research_experiments.mobile_ev_mle';old.__file__=str(stage/'baseline_location_review_from_git.py')
exec(compile(before,old.__file__,'exec'),old.__dict__)
saved=repo/'research_experiments/mobile_ev_mle/evidence/ev_original_domain/location_ablation_v1/history_988401.json.gz'
parent=repo/'research_experiments/mobile_ev_mle/evidence/ev_original_domain/evaluation_v2/trial_988401.json.gz'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected=sha(parent);saved_hash=sha(saved)
a=old.replay_history(read_json(saved),read_json(parent),expected)
b=current.replay_history(read_json(saved),read_json(parent),expected)
encode=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
assert encode(a)==encode(b),'default reviewer output changed'
assert sha(parent)==expected and sha(saved)==saved_hash
out=dict(success=True,seed=988401,scope='one preserved stage-16 history; default optional arguments',
    exact_output_equal=True,old_git_commit='776280b',old_code_sha256=hashlib.sha256(before).hexdigest(),
    new_code_sha256=sha(Path(current.__file__)),parent_raw_sha256=expected,saved_ablation_sha256=saved_hash,
    replayed_fit_states=b[0]['saved_fit_states_replayed'],physical_states=b[0]['physical_states_replayed'],
    no_fit_or_selector=True,output_sha256=hashlib.sha256(encode(b).encode()).hexdigest())
output=stage/'research_experiments/mobile_ev_mle/evidence/ev_compact/tooling_stage4_validation/legacy_default_replay.json'
write_json(output,out);print(encode(out))
