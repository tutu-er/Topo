"""Private physical DR worker; invoke only after the price lock is frozen.

No physical work runs on import. Output packet.pkl contains observed fields
only. Evaluator truth and physical records are separate private files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix+'.partial')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def write_pickle(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix+'.partial')
    temporary.write_bytes(pickle.dumps(value,protocol=4))
    temporary.replace(path)


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan-lock',type=Path,required=True)
    parser.add_argument('--expected-lock-sha',required=True)
    parser.add_argument('--policy',required=True)
    parser.add_argument('--path',type=int,required=True)
    parser.add_argument('--dependence',choices=('local','common'),required=True)
    parser.add_argument('--sigma',type=float,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repo-root',type=Path,default=Path.cwd())
    parser.add_argument('--source-root',type=Path,
        default=Path(r'D:\0-github_workspace\Topo\outputs\wzzt_paper_20261029_full\source'))
    parser.add_argument('--replicate-base',type=int,default=10000)
    parser.add_argument('--participation-seed',type=int,default=90310000)
    parser.add_argument('--event-profiles',type=int,nargs=4,default=[0,2,0,2])
    args=parser.parse_args(argv)
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        print(json.dumps(dict(status='existing_output_retained',output=str(out))),flush=True)
        return 3
    out.mkdir(parents=True,exist_ok=True)
    # A fresh worker ensures previously imported current core packages cannot
    # override the exact archived physical producer modules.
    sys.path.insert(0,str(args.repo_root.resolve()))
    sys.path.insert(0,str(args.source_root.resolve()))
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    receipt=dict(status='started',started_utc=datetime.now(timezone.utc).isoformat(),
        policy=args.policy,path=args.path,dependence=args.dependence,voltage_sigma=args.sigma,
        plan_lock_sha256_expected=args.expected_lock_sha,
        source_sha256={str(Path(__file__).resolve()):sha(__file__)},
        physical_events_expected=4,physical_event_rows_expected=96,
        zero_clip=True,automatic_retry=False,private_files={},
        public_receipt_contains_trajectories=False,old_test_or_truth_files_read=False)
    write_json(out/'events_receipt.json',receipt)
    try:
        from physical_bridge import generate_actual_events,evaluate_supports
        import physical_bridge
        receipt['source_sha256'][str(Path(physical_bridge.__file__).resolve())]=sha(physical_bridge.__file__)
        actual=generate_actual_events(args.plan_lock,args.expected_lock_sha,args.policy,
            path=args.path,dependence=args.dependence,voltage_sigma=args.sigma,
            source_root=args.source_root,replicate_base=args.replicate_base,
            event_profiles=tuple(args.event_profiles),participation_seed=args.participation_seed,
            repo_root=args.repo_root)
        # Identifier packet is strictly observed. It intentionally omits truth,
        # actual participants, the network object, and source profile parameters.
        packet=dict(training=actual['training'],validation=actual['validation'],metadata=dict(
            terminal_labels=actual['labels'],base_kw=200.,base_kv=.4,base_mva=.2,
            interval_hours=.25,synthetic_dates_are_order_labels=True,
            observed_only=True,plan_lock_sha256=args.expected_lock_sha,
            policy=args.policy,path=args.path,dependence=args.dependence,
            status=actual['status']))
        write_pickle(out/'packet.pkl',packet)
        empty_evaluation=evaluate_supports([],actual['physical_network'])
        truth_evaluation=dict(true_supports=empty_evaluation['missed_blocks'],
            terminal_labels=actual['labels'],full_effective_truth_blocks=empty_evaluation['full_effective_truth_blocks'],
            zero_impedance_merged=True,arbitrarily_weak_positive_lines_retained=True,
            use='Evaluator only after identifier output is fixed; never planner/fit input',
            plan_lock_sha256=args.expected_lock_sha)
        write_json(out/'truth_evaluation.json',truth_evaluation)
        # Preserve exact physical P/Q/V/root/AC results separately for later
        # independent physical replay. This file remains private and is never a
        # planner or identifier input. The network class is not pickled.
        write_pickle(out/'physical_truth.pkl',dict(events=actual['truth']))
        receipt.update(status=actual['status'],completed_utc=datetime.now(timezone.utc).isoformat(),
            actual_payment=actual['actual_payment'],worst_payment=actual['worst_payment'],
            budget_pass=actual['budget_pass'],all_operation_checks_pass=actual['all_operation_checks_pass'],
            observed_training_days=len(actual['training']),observed_validation_days=len(actual['validation']),
            physical_events_complete=sum(r['status']=='complete' for r in actual['event_records']),
            event_records=actual['event_records'],runtime_sha256=actual['runtime_sha256'],
            seeds=dict(replicate_base=args.replicate_base,participation_seed=args.participation_seed,
                       event_profiles=args.event_profiles),
            old_test_or_truth_files_read=actual['old_test_or_truth_files_read'])
        for name in ('packet.pkl','truth_evaluation.json','physical_truth.pkl'):
            path=out/name
            receipt['private_files'][name]=dict(sha256=sha(path),bytes=path.stat().st_size)
        write_json(out/'events_receipt.json',receipt)
        print(json.dumps({k:receipt[k] for k in ('status','physical_events_complete','actual_payment',
            'worst_payment','budget_pass','all_operation_checks_pass')}),flush=True)
        return 0 if actual['status']=='complete' else 2
    except Exception as error:
        receipt.update(status='worker_failed',completed_utc=datetime.now(timezone.utc).isoformat(),
            error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            failure_retained=True,automatic_retry=False)
        write_json(out/'events_receipt.json',receipt)
        print(json.dumps(dict(status=receipt['status'],error_type=receipt['error_type'],message=receipt['message'])),flush=True)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
