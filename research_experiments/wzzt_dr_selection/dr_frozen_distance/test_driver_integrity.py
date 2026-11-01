"""Receipt binding tests prevent post-decision plan or parameter drift."""
import importlib.util
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
# Under the installed package, relative imports are intentional.
from research_experiments.wzzt_dr_selection.dr_frozen_distance import driver


def fixture(tmp):
    for name in ('pool','history','problem','plan'):
        driver.write(tmp/(name+'.json'),dict(original=name))
    c=dict(status='frozen',source_sha256={},frozen_pool=driver.ref(tmp/'pool.json'),
        history_packet=driver.ref(tmp/'history.json'),problem=driver.ref(tmp/'problem.json'))
    driver.write(tmp/'protocol.json',c)
    decision=dict(protocol=driver.ref(tmp/'protocol.json'),frozen_pool=c['frozen_pool'],
        problem=c['problem'],plans=dict(optimized=driver.ref(tmp/'plan.json')))
    driver.write(tmp/'price_freeze.json',decision)
    driver.write(tmp/'preparation_ended.json',dict(price_freeze=driver.ref(tmp/'price_freeze.json')))


def test_frozen_receipt_chain_accepts_original(tmp_path):
    fixture(tmp_path)
    _,c=driver.config(tmp_path)
    assert c['status']=='frozen'


def test_mutating_price_and_new_decision_ref_is_rejected(tmp_path):
    fixture(tmp_path)
    driver.write(tmp_path/'plan.json',dict(changed_price=2))
    decision=driver.read(tmp_path/'price_freeze.json')
    decision['plans']['optimized']=driver.ref(tmp_path/'plan.json')
    driver.write(tmp_path/'price_freeze.json',decision)
    with pytest.raises(RuntimeError,match='source drift'):driver.config(tmp_path)


def test_mutating_frozen_physical_problem_is_rejected(tmp_path):
    fixture(tmp_path)
    driver.write(tmp_path/'problem.json',dict(changed_budget=9))
    with pytest.raises(RuntimeError,match='source drift'):driver.config(tmp_path)
