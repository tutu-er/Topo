"""Atomic research records and exact execution-source fingerprints."""
from dataclasses import asdict,is_dataclass
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import gzip,json,math,os,sys,zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[3]

def serial(x):
    if is_dataclass(x):return serial(asdict(x))
    if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [serial(v) for v in x]
    if isinstance(x,np.ndarray):return serial(x.tolist())
    if isinstance(x,np.generic):return serial(x.item())
    if isinstance(x,(Path,datetime)):return str(x)
    if isinstance(x,float) and not math.isfinite(x):return None
    return x

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(serial(value),ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
    if path.suffix=='.gz':data=gzip.compress(data,compresslevel=3,mtime=0)
    tmp=path.with_name(path.name+'.tmp');tmp.write_bytes(data);tmp.replace(path)

def read(path):
    path=Path(path);b=path.read_bytes()
    if path.suffix=='.gz':b=gzip.decompress(b)
    return json.loads(b)

def digest(path):return sha256(Path(path).read_bytes()).hexdigest()

def source_hashes():
    paths=[ROOT/'research_experiments/__init__.py']
    for rel in ('research_experiments/mobile_ev_mle','research_experiments/mobile_ev_probing','rnj_wzzt_core','terminal_case33'):
        paths.extend(p for p in (ROOT/rel).rglob('*.py') if not any(v in p.parts for v in ('evidence','__pycache__','outputs'))
            and not any(v.startswith('results') for v in p.parts)
            and p!=Path(__file__).with_name('analysis.py'))
    paths.extend(Path(__file__).parent/n for n in ('protocol.md','math.md'))
    return {p.relative_to(ROOT).as_posix():digest(p) for p in sorted(paths)}

def freeze(output,config,prepared):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    imported={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.copy().items()
        if name.split('.')[0] in ('research_experiments','terminal_case33','rnj_wzzt_core')
        and getattr(module,'__file__',None)}
    outside={name:path for name,path in imported.items() if not Path(path).is_relative_to(ROOT)}
    if outside:raise RuntimeError('repository dependency escaped execution snapshot: '+str(outside))
    payload=dict(config=config,prepared_sha256=digest(prepared),source_sha256=source_hashes(),
        selector_reads_truth=False,statistical_unit='one nonoverlapping real calendar window; network/source conditions paired')
    identity=sha256(json.dumps(serial(payload),sort_keys=True).encode()).hexdigest()
    value=dict(payload,identity=identity)
    path=output/'freeze.json'
    if path.exists():
        if read(path)!=serial(value):raise RuntimeError('frozen policy differs; retain output and use a new version')
    else:
        write(path,value)
        with zipfile.ZipFile(output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in payload['source_sha256']:z.write(ROOT/name,name)
        import scipy,pandas,gurobipy
        write(output/'environment.json',dict(python=sys.version,executable=sys.executable,
            numpy=np.__version__,scipy=scipy.__version__,pandas=pandas.__version__,gurobi=gurobipy.gurobi.version(),
            cpu_count=os.cpu_count(),threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
            imported_repository_modules=imported,recorded_utc=datetime.now(timezone.utc).isoformat()))
    return value

def compact_fits(value):
    """Remove reconstructible residual copies, preserving states, failures and bounds."""
    if isinstance(value,dict):
        result={}
        for k,v in value.items():
            if k in ('residual','jacobian','whitened_jacobian','predicted') and isinstance(v,(list,np.ndarray,dict)):
                b=json.dumps(serial(v),sort_keys=True).encode()
                result[k+'_omitted']=dict(sha256=sha256(b).hexdigest(),bytes=len(b),scope='reconstructible duplicate; final state and observations retained')
            else:result[k]=compact_fits(v)
        return result
    if isinstance(value,(tuple,list)):return [compact_fits(v) for v in value]
    return value
