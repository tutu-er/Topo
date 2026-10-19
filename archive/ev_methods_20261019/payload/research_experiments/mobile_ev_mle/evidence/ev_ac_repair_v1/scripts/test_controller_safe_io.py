from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path

import pytest

spec=spec_from_file_location('safe_io',Path(__file__).with_name('controller_safe_io.py'))
module=module_from_spec(spec);spec.loader.exec_module(module)


def test_retry_preserves_payload_and_uses_original_writer():
    payload={'frozen':[1,2,3]}; calls=[]; notifications=[]
    def writer(path,value):
        calls.append((path,value))
        if len(calls)<3:raise PermissionError('temporary reader lock')
        return 'saved'
    assert module.retry_write(writer,'same.json',payload,pause=lambda _:None,
        clock=lambda:0.,notify=notifications.append)=='saved'
    assert len(calls)==3 and all(value is payload for _,value in calls)
    assert [r['attempt'] for r in notifications]==[1,2]


def test_permanent_permission_error_is_not_hidden():
    values=iter([0.,1.,2.,11.])
    def writer(path,value):raise PermissionError('permanent denial')
    with pytest.raises(PermissionError,match='permanent'):
        module.retry_write(writer,'same.json',{},pause=lambda _:None,
            clock=lambda:next(values),seconds=10.)


def test_other_errors_immediately_propagate():
    def writer(path,value):raise ValueError('scientific invariant')
    with pytest.raises(ValueError,match='scientific invariant'):
        module.retry_write(writer,'same.json',{},pause=lambda _:None)
