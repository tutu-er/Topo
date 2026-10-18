"""Calibration ordering and retained-denominator resume contracts."""
from copy import deepcopy
import pytest
from . import controller as c
from .io import read,write,serial


def spec(index=0,source='none'):
    return dict(phase='calibration',cluster_index=index,network='case33',source_mode=source,n=32,
                physical_seed=100+index,seed=100+index+(source!='none'))


def records(specs,identity='frozen'):
    return [dict(spec=s,freeze_identity=identity,status='failed',failures={'optimizer':'retained'}) for s in specs]


def setup_cal(monkeypatch,rows,expected):
    reads=[]
    def load(output,phase):
        reads.append(phase)
        assert phase=='calibration'
        return rows
    monkeypatch.setattr(c.analysis,'records',load,raising=False)
    monkeypatch.setattr(c.study,'specifications',lambda prepared,phase:expected)
    def seal(output,**kw):
        path=output/'thresholds.json';value=serial({'cut':float('inf'),'rows':len(rows)})
        if path.exists() and read(path)!=value:raise ValueError('existing thresholds differ')
        write(path,value)
        return value
    monkeypatch.setattr(c.analysis,'freeze_thresholds',seal)
    return reads


def test_seal_before_evaluation_does_not_read_outcomes(tmp_path,monkeypatch):
    specs=[spec(),spec(1,'PV')];reads=setup_cal(monkeypatch,records(specs),specs)
    receipt=c.seal_calibration(tmp_path,{},'frozen')
    assert reads==['calibration'] and receipt['evaluation_outcomes_read']==0
    assert read(tmp_path/'thresholds.json')=={'cut':None,'rows':2}
    # Nonfinite conservative thresholds are round-trippable and never replaced on resume.
    assert c.seal_calibration(tmp_path,{},'frozen')['threshold_sha256']==receipt['threshold_sha256']


@pytest.mark.parametrize('change',['missing','duplicate','identity'])
def test_bad_calibration_cannot_release_evaluation(tmp_path,monkeypatch,change):
    specs=[spec(),spec(1)];rows=records(specs)
    if change=='missing':rows.pop()
    if change=='duplicate':rows[1]=deepcopy(rows[0])
    if change=='identity':rows[1]['freeze_identity']='another'
    setup_cal(monkeypatch,rows,specs)
    with pytest.raises(ValueError):c.seal_calibration(tmp_path,{},'frozen')
    assert not (tmp_path/'thresholds.json').exists()


def test_existing_threshold_never_overwritten(tmp_path,monkeypatch):
    setup_cal(monkeypatch,records([spec()]),[spec()])
    write(tmp_path/'thresholds.json',{'cut':123})
    with pytest.raises(ValueError,match='existing thresholds'):c.seal_calibration(tmp_path,{},'frozen')
    assert read(tmp_path/'thresholds.json')=={'cut':123}


def test_failed_result_resumes_without_replacing_sample(tmp_path):
    specs=[spec(),spec(1)];path=c.study.path_for(tmp_path,specs[0])
    write(path,records([specs[0]])[0])
    saved,pending=c.saved_or_pending(tmp_path,specs,'frozen')
    assert len(saved)==1 and saved[0]['status']=='failed'
    assert pending==specs[1:] and read(path)['status']=='failed'


@pytest.mark.parametrize('field',['freeze_identity','spec'])
def test_resume_rejects_different_experiment(tmp_path,field):
    s=spec();row=records([s])[0];row[field]='wrong'
    write(c.study.path_for(tmp_path,s),row)
    with pytest.raises(ValueError):c.saved_or_pending(tmp_path,[s],'frozen')


def test_freeze_locks_calibration_rule_before_evaluation():
    from . import io,analysis
    path=io.Path(analysis.__file__).resolve()
    hashes=io.source_hashes()
    assert hashes[path.relative_to(io.ROOT).as_posix()]==io.digest(path)
