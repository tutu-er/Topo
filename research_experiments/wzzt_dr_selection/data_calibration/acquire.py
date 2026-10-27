"""Acquire fixed public iFlex files; raw household data stay outside Git."""
import argparse, datetime, hashlib, json, pathlib, sys, os
import requests

ROOT = pathlib.Path(__file__).resolve().parent
DATA = pathlib.Path(os.environ.get('WZZT_DR_DATA_DIR', str(ROOT)))
RAW = DATA / 'raw'
RAW.mkdir(parents=True,exist_ok=True)
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
keys = ['price_signals.csv', 'experiment_days.csv', 'participants.csv', 'iFlex dataset.RData']
response = requests.get('https://zenodo.org/api/records/8248802', timeout=40)
response.raise_for_status()
meta = response.json()
(OUT / 'zenodo_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
manifest = []
for item in meta['files']:
    if item['key'] not in keys:
        continue
    path = RAW / item['key']
    expected = item['checksum'].split(':', 1)[1]
    if not path.exists() or hashlib.md5(path.read_bytes()).hexdigest() != expected:
        temp = path.with_suffix(path.suffix + '.partial')
        with requests.get(item['links']['self'], stream=True, timeout=(20, 60)) as r:
            r.raise_for_status()
            with temp.open('wb') as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
        assert temp.stat().st_size == item['size'], item['key']
        assert hashlib.md5(temp.read_bytes()).hexdigest() == expected, item['key']
        temp.replace(path)
    result = {'file': item['key'], 'bytes': path.stat().st_size, 'md5': expected,
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'url': item['links']['self'], 'md5_verified': True}
    manifest.append(result)
    print(json.dumps(result), flush=True)
(OUT / 'raw_manifest.json').write_text(json.dumps({'record': 8248802, 'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'license': meta['metadata']['license'], 'files': manifest}, indent=2), encoding='utf-8')
sys.path.insert(0, str(DATA / 'deps'))
import pyreadr
objects = pyreadr.list_objects(str(RAW / 'iFlex dataset.RData'))
(OUT / 'rdata_objects.json').write_text(json.dumps(objects, indent=2), encoding='utf-8')
print(json.dumps(objects), flush=True)
