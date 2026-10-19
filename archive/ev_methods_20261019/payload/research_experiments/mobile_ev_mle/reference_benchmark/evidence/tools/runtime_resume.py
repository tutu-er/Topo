"""Resume the unchanged frozen batch after all old processes and its lock exit.

Only reference_benchmark.io.write is wrapped, before controller/study imports in
both the parent and Windows-spawn children. Scientific functions, configuration,
record bytes, seeds and saved failed cases are unchanged.
"""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / 'rnj_wzzt_core'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True
LOG_ROOT = Path(os.environ.get('TOPO_RUNTIME_IO_LOG', str(ROOT / 'outputs/reference_v1/runtime_io')))
MAX_RETRIES = 8
RETRY_SECONDS = (.05, .1, .2, .4, .5, .5, .5, .5)


def now():
    return datetime.now(timezone.utc).isoformat()


def append_event(event, **details):
    """One append per event, and one log per PID; logging errors remain visible."""
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(dict(event=event, pid=os.getpid(), utc=now(), **details),
                       sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8')
    descriptor = os.open(str(LOG_ROOT / ('pid_%d.jsonl' % os.getpid())),
                         os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o666)
    try:
        written = os.write(descriptor, line)
        if written != len(line):
            raise OSError('incomplete runtime audit append')
    finally:
        os.close(descriptor)


def retry_writer(original, *, sleep=time.sleep, retry_seconds=RETRY_SECONDS):
    def wrapped(path, value):
        for attempt in range(len(retry_seconds) + 1):
            try:
                result = original(path, value)
                if attempt:
                    append_event('write_recovered', path=str(path), failed_attempts=attempt)
                return result
            except PermissionError as error:
                append_event('write_permission_error', path=str(path), attempt=attempt + 1,
                             winerror=getattr(error, 'winerror', None), error=repr(error),
                             will_retry=attempt < len(retry_seconds))
                if attempt == len(retry_seconds):
                    raise
                sleep(retry_seconds[attempt])
    wrapped.__name__ = 'runtime_permission_retry_write'
    wrapped._topo_runtime_permission_retry = True
    wrapped._original = original
    return wrapped


# Installation must precede controller/study import. Their `from .io import write`
# aliases therefore receive this same wrapper in every spawned Python process.
from research_experiments.mobile_ev_mle.reference_benchmark import io
ORIGINAL_WRITE = io.write
if not getattr(io.write, '_topo_runtime_permission_retry', False):
    io.write = retry_writer(io.write)
PATCH_METADATA = dict(
    runtime_file=str(Path(__file__).resolve()),
    runtime_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    frozen_io_sha256=sha256(Path(io.__file__).read_bytes()).hexdigest(),
    scope='runtime wrapper of reference_benchmark.io.write only',
    retries=MAX_RETRIES, retry_seconds=list(RETRY_SECONDS),
    record_bytes='original serializer/gzip/temporary-write/replace are called unchanged',
    scientific_source_files_modified=0, scientific_fit_or_plan_changes=False)
append_event('patch_installed', **PATCH_METADATA)


def read_shared_json(path):
    """Do not deny deletion/atomic replacement of a live controller record."""
    path = Path(path)
    if os.name != 'nt':
        return json.loads(path.read_bytes())
    from ctypes import wintypes
    import msvcrt
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    # GENERIC_READ; FILE_SHARE_READ | WRITE | DELETE; OPEN_EXISTING.
    handle = create(str(path), 0x80000000, 0x7, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    with os.fdopen(descriptor, 'rb') as stream:
        return json.loads(stream.read())


def process_is_alive(pid):
    """Conservative: an inaccessible or recycled recorded PID blocks a resume."""
    pid = int(pid)
    if pid <= 0:
        raise ValueError('invalid recorded process ID')
    if os.name != 'nt':
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: PID no longer exists
            return False
        if error == 5:   # ERROR_ACCESS_DENIED: refuse, rather than guess dead
            return True
        raise ctypes.WinError(error)
    try:
        status = kernel.WaitForSingleObject(handle, 0)
        if status == 0:  # process handle signalled
            return False
        if status == 258:
            return True
        raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)


def runtime_preflight(output, prepared, workers):
    """Read only; never remove a lock or terminate/attach to an existing process."""
    output, prepared = Path(output).resolve(), Path(prepared).resolve()
    if not output.is_relative_to(ROOT / 'outputs') or not prepared.is_relative_to(ROOT / 'outputs'):
        raise ValueError('resume paths must remain inside this execution stage outputs')
    state_files = [output / 'controller.json', output / 'controller.json.tmp']
    states = [read_shared_json(path) for path in state_files if path.exists()]
    recorded = set()
    for state in states:
        if state.get('pid'):
            recorded.add(int(state['pid']))
        recorded.update(map(int, state.get('live_worker_pids', [])))
    alive = sorted(pid for pid in recorded if process_is_alive(pid))
    locked = (output / 'controller.lock').exists()
    if alive or locked:
        append_event('resume_refused', active_recorded_pids=alive, controller_lock_exists=locked)
        raise RuntimeError('resume refused: recorded processes still alive %s; controller lock exists=%s; '
                           'no process started and no lock removed' % (alive, locked))
    # Import only after patch installation and after the old run is inactive.
    from research_experiments.mobile_ev_mle.reference_benchmark import controller, study
    from research_experiments.mobile_ev_mle.reference_benchmark.config import CONFIG
    if workers < 1 or workers > CONFIG['max_workers']:
        raise ValueError('worker count outside frozen budget')
    if not (output / 'freeze.json').exists():
        raise ValueError('existing freeze.json required; this tool does not create a new scientific batch')
    frozen = io.read(output / 'freeze.json')
    if frozen.get('source_sha256') != io.source_hashes():
        raise ValueError('scientific source hash differs from existing freeze')
    if frozen.get('config') != io.serial(CONFIG) or frozen.get('prepared_sha256') != io.digest(prepared):
        raise ValueError('frozen configuration or prepared inputs differ')
    inputs = io.read(prepared)
    inventory = {}
    for phase in ('development', 'calibration', 'evaluation'):
        if len(inputs['windows'][phase]) != CONFIG['phase_counts'][phase]:
            raise ValueError('prepared count differs for ' + phase)
        specs = study.specifications(inputs, phase)
        saved, pending = controller.saved_or_pending(output, specs, frozen['identity'])
        expected_paths = {study.path_for(output, spec).resolve() for spec in specs}
        unexpected = [str(path) for path in (output / phase).glob('*.json.gz')
                      if path.resolve() not in expected_paths]
        if unexpected:
            raise ValueError('unexpected saved records: ' + str(unexpected))
        inventory[phase] = dict(saved=len(saved), pending=len(pending), total=len(specs),
                                saved_failed=sum(row['status'] == 'failed' for row in saved))
    # Also verifies repository imports all resolve inside this stage; existing
    # freeze is compared, not rewritten or assigned a different policy identity.
    checked = io.freeze(output, CONFIG, prepared)
    append_event('resume_preflight_passed', freeze_identity=checked['identity'],
                 inventory=inventory, workers=workers, scientific_resampling=False)
    return controller, inventory, checked['identity']


def child_probe(queue):
    queue.put(dict(pid=os.getpid(), installed=getattr(io.write, '_topo_runtime_permission_retry', False),
                   study_imported='research_experiments.mobile_ev_mle.reference_benchmark.study' in sys.modules,
                   runtime_sha256=PATCH_METADATA['runtime_sha256']))


def self_test():
    target = ROOT / 'outputs/reference_tools/runtime_resume_selftest'
    target.mkdir(parents=True, exist_ok=True)
    checks = {}
    attempts = []
    def transient(path, value):
        attempts.append(1)
        if len(attempts) <= 2:
            raise PermissionError('injected transient sharing violation')
        return ORIGINAL_WRITE(path, value)
    retry_writer(transient, sleep=lambda _: None)(target / 'injected.json', {'ok': True})
    checks['injected_first_two_permission_errors'] = len(attempts) == 3
    for error_type in (ValueError, OSError):
        called = []
        def fail_other(path, value):
            called.append(1)
            raise error_type('injected non-permission error')
        try:
            retry_writer(fail_other, sleep=lambda _: None)(target / 'nonpermission.json', {})
            checks[error_type.__name__ + '_propagated'] = False
        except error_type:
            checks[error_type.__name__ + '_propagated'] = len(called) == 1
    fixture = {'unicode': '原记录', 'array': [1, None, .25], 'nested': {'z': False}}
    for suffix in ('json', 'json.gz'):
        a, b = target / ('original.' + suffix), target / ('wrapped.' + suffix)
        ORIGINAL_WRITE(a, fixture)
        io.write(b, fixture)
        checks[suffix + '_identical_bytes'] = a.read_bytes() == b.read_bytes()
    if os.name == 'nt':
        locked_path = target / 'windows_lock.json'
        ORIGINAL_WRITE(locked_path, {'version': 1})
        handle = locked_path.open('rb')
        try:
            ORIGINAL_WRITE(locked_path, {'version': 2})
            checks['real_windows_read_handle_blocks_original_replace'] = False
        except PermissionError as error:
            checks['real_windows_read_handle_blocks_original_replace'] = True
            checks['real_windows_lock_winerror'] = getattr(error, 'winerror', None)
        finally:
            handle.close()
        handle = locked_path.open('rb')
        releaser = threading.Thread(target=lambda: (time.sleep(.12), handle.close()))
        releaser.start()
        io.write(locked_path, {'version': 3})
        releaser.join()
        checks['real_windows_lock_recovers_after_handle_close'] = io.read(locked_path) == {'version': 3}
        # This reader permits atomic replacement even with its handle alive.
        checks['share_delete_reader_round_trip'] = read_shared_json(locked_path) == {'version': 3}
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    children = [context.Process(target=child_probe, args=(queue,)) for _ in range(2)]
    for child in children:
        child.start()
    probes = [queue.get(timeout=60) for _ in children]
    for child in children:
        child.join(timeout=60)
        if child.is_alive() or child.exitcode != 0:
            raise RuntimeError('self-test spawn child did not exit normally')
    queue.close()
    queue.join_thread()
    checks['both_spawn_children_patched_before_study_import'] = all(
        item['installed'] and not item['study_imported']
        and item['runtime_sha256'] == PATCH_METADATA['runtime_sha256'] for item in probes)
    checks['no_scientific_study_imported'] = 'research_experiments.mobile_ev_mle.reference_benchmark.study' not in sys.modules
    passed = all(value for key, value in checks.items() if not key.endswith('_winerror'))
    receipt = dict(passed=passed, checks=checks, children=probes, metadata=PATCH_METADATA,
                   scientific_fit_calls=0, scientific_plan_calls=0, completed_utc=now())
    io.write(target / 'receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
    if not passed:
        raise RuntimeError('runtime recovery self-test failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/reference_v1')
    parser.add_argument('--prepared', type=Path, default=ROOT / 'outputs/inputs_v2/prepared.json.gz')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    controller, inventory, identity = runtime_preflight(args.output, args.prepared, args.workers)
    receipt = dict(recorded_utc=now(), pid=os.getpid(), metadata=PATCH_METADATA,
                   inventory=inventory, freeze_identity=identity, old_run_all_recorded_pids_exited=True,
                   old_lock_absent=True, failed_results_retained=True, source_code_unchanged=True)
    io.write(args.output / ('runtime_resume_%d.json' % os.getpid()), receipt)
    sys.argv = [str(Path(controller.__file__)), '--output', str(args.output),
                '--prepared', str(args.prepared), '--workers', str(args.workers)]
    controller.main()


if __name__ == '__main__':
    main()
