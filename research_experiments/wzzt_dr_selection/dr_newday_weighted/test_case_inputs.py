"""Input binding and complete-history tests; no case fit or future generated."""
from pathlib import Path
import pickle
import sys
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import case_inputs as m


class GuardedPacket(dict):
    def __getitem__(self,key):
        if key=='forecast':raise AssertionError('forecast was accessed')
        return super().__getitem__(key)


def fixture(tmp,*,rows=96,labels=(101,102)):
    day=dict(P_terminal=pd.DataFrame(1.,index=range(rows),columns=labels),
        Q_terminal=pd.DataFrame(.3,index=range(rows),columns=labels))
    history=dict(training=[day]*3,validation=[day]*3,labels=[101,102],root_label=1,
        base_kw=200.,base_kv=.4,base_mva=.2,source_case_name='unit_case')
    file=tmp/'history.pkl'
    file.write_bytes(pickle.dumps(GuardedPacket(history=history,forecast='never use'),protocol=4))
    return m.reference(file)


def test_old_hash_binding_is_required_and_checked(tmp_path):
    file=tmp_path/'a';file.write_bytes(b'old')
    ref=m.reference(file);assert m.checked(ref)==file
    file.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash drift'):m.checked(ref)
    with pytest.raises(ValueError,match='explicit old'):m.checked(dict(path=str(file)))


def test_complete_history_headers_without_forecast_access(tmp_path):
    result=m.history_header(fixture(tmp_path),'unit',1,200.)
    assert result['rows']==[288,288] and result['labels']==[101,102]


def test_incomplete_day_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='96-row'):m.history_header(fixture(tmp_path,rows=95),'unit',1,200.)


def test_changed_label_order_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='P order'):m.history_header(fixture(tmp_path,labels=(102,101)),'unit',1,200.)


def test_changed_original_base_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='root/power base'):m.history_header(fixture(tmp_path),'unit',1,100.)
