"""Archive completed matched-budget evidence and verify all frozen members."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib
import shutil
from .run import ROOT,METHODS,read,write,sha,completion


def deliver(output):
    output=Path(output);delivery=ROOT/'docs'/output.name;p=read(output/'protocol.json')
    assert completion(output)['returned']==240 and read(output/'analysis/verification.json')['passed']
    assert read(delivery/'tests.json')['passed']
    manifest=p['parent_manifest'];assert sha(manifest['path'])==manifest['sha256']
    for name,record in manifest['files'].items():assert sha(Path(manifest['path']).parent/name)==record['sha256']
    for method in METHODS:
        for name,digest in p['source_hashes'][method].items():assert sha(output/'sources'/method/name)==digest
    audit=read(output/'analysis/verification.json')
    assert audit['analyzer_sha256']==sha(Path(__file__).with_name('analyze.py'))
    result_hashes={}
    for job in p['jobs']:
        for method in METHODS:
            file=output/'jobs'/job/method/'result.json';r=read(file)
            assert r['protocol_sha256']==sha(output/'protocol.json') and not r['total_cap_exceeded']
            result_hashes[job+'/'+method]=sha(file)
    write(delivery/'result_hashes.json',result_hashes)
    recovery='''# 恢复与核验

入口协议protocol.json固定120个输入、两份源码、数值精度适配、4小时共同总预算及前端成本。PLAN.md说明控制变量和归因范围，RESULTS.md为总表，per_model/paired_inputs/mechanism_inputs为逐输入证据；verification.json是独立LP/哈希检查回执，solver.csv保留子问题状态与界。result_hashes.json对应全部240个最终训练记录。

records.zip包含sources/before、sources/after两份真正执行的源码，当前分析/交付源码current_source、共同前端与旧300秒记录、240个新结果、checkpoint、进度日志及逐输入独立LP核验。它不包含观测NPZ，原始数据路径及SHA256由catalog/protocol引用。

恢复时先核验manifest.json，再核验协议源码哈希及240个结果哈希。run命令会跳过已有result/error；恢复前须检查state.json中的worker PID，避免与尚在运行的进程重复计算。保留失败记录及原分母，处理错误时另存新协议或记录必要的驱动适配，不覆盖历史源码/结果。

运行命令：python -B -m research_experiments.iterative_benchmark.run run outputs/iterative_exit_matched120_20261101

分析命令：python -B -m research_experiments.iterative_benchmark.analyze outputs/iterative_exit_matched120_20261101

分析器按源码SHA256复用逐输入核验缓存；修改分析源码会使旧缓存失效。该入口要求240个新结果齐全。重新分析会更新报告，不必重做训练求解。
'''
    (delivery/'RECOVERY.md').write_text(recovery,encoding='utf-8')
    shutil.copy2(output/'completion.json',delivery/'completion.json')
    archive=delivery/'records.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for directory in ('sources','common','jobs','analysis'):
            for f in (output/directory).rglob('*'):
                if f.is_file() and f.suffix in ('.py','.json','.csv','.log'):
                    z.write(f,f.relative_to(output).as_posix())
        for name in ('protocol.json','catalog.json','completion.json'):z.write(output/name,name)
        for f in Path(__file__).parent.glob('*.py'):z.write(f,'current_source/'+f.relative_to(ROOT).as_posix())
        f=ROOT/'research_experiments/tests/test_iterative_benchmark.py';z.write(f,'current_source/'+f.relative_to(ROOT).as_posix())
        for f in delivery.iterdir():
            if f.is_file() and f.suffix in ('.md','.json','.csv') and f.name not in ('manifest.json','delivery_audit.json'):
                z.write(f,'reports/'+f.name)
    with ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for method in METHODS:
            for name,digest in p['source_hashes'][method].items():assert hashlib.sha256(z.read('sources/'+method+'/'+name)).hexdigest()==digest
        for key,digest in result_hashes.items():assert hashlib.sha256(z.read('jobs/'+key+'/result.json')).hexdigest()==digest
    write(delivery/'delivery_audit.json',dict(passed=True,returned=240,archive_sha256=sha(archive),
        independent_LPs=audit['independent_LPs'],historical_files_preserved=len(manifest['files'])))
    write(delivery/'manifest.json',dict(files={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(delivery.iterdir()) if f.is_file() and f.name!='manifest.json'}))
    print('Archived 240 matched fresh results, executed sources and independent verification',flush=True)


if __name__=='__main__':
    import sys
    deliver(Path(sys.argv[1]).resolve())
