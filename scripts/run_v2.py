"""Resume v2 generation without discarding attempted calls or changing prompts.
Invalid Python is an experimental failure. Infrastructure errors stop the batch.
Dependent reuse steps are blocked, not retried or given another predecessor.
Never execute generated Python.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from generate_v2 import code_only

ROOT = Path(__file__).resolve().parents[1]
SKILLS = 'remember guess certify flow blame justify choose fix spot race'.split()
MODEL = 'claude-sonnet-5-5'
RECORDED = {'generated', 'syntax_failed', 'blocked'}


def folder(t, step, repeat):
    return ROOT / f'experiments/runs/t{t}/r{repeat:02d}/step{step:02d}'


def digest():
    return hashlib.sha256((ROOT/'docs/acceptance-v2.md').read_bytes()).hexdigest()


def classify_record(t, step, repeat):
    """Check existing evidence; never convert an unknown error into a model failure."""
    run = folder(t, step, repeat)
    if not run.exists():
        return 'missing'
    try:
        m = json.loads((run/'metrics.json').read_text())
        if (m.get('protocol'), m.get('contract_sha256'), m.get('treatment'),
                m.get('step'), m.get('repeat')) != ('v2', digest(), t, step, repeat):
            return 'unresolved'
        if m.get('status') == 'blocked_by_predecessor':
            if t % 2 or step <= 1 or m.get('blocked_by_step') != step-1:
                return 'unresolved'
            if classify_record(t, step-1, repeat) not in {'syntax_failed', 'blocked'}:
                return 'unresolved'
            if any((run/name).exists() for name in ('generated.py','response.json','prompt.txt')):
                return 'unresolved'
            return 'blocked'
        if m.get('prompt_sha256') != hashlib.sha256((run/'prompt.txt').read_bytes()).hexdigest():
            return 'unresolved'
        if m.get('status') == 'generated':
            if (m.get('source_sha256') == hashlib.sha256((run/'generated.py').read_bytes()).hexdigest()
                    and set(m.get('model_usage') or {}) == {MODEL}):
                return 'generated'
            return 'unresolved'
        if m.get('status') != 'generation_failed' or m.get('returncode') != 0:
            return 'unresolved'
        response = json.loads((run/'response.json').read_text())
        if (response.get('is_error') or set(response.get('modelUsage') or {}) != {MODEL}
                or set(m.get('model_usage') or {}) != {MODEL}
                or not isinstance(response.get('result'), str)
                or (run/'generated.py').exists()):
            return 'unresolved'
        try:
            code_only(response['result'])
        except SyntaxError as exc:
            if m.get('error') == str(exc):
                return 'syntax_failed'
            # Python-version-dependent exception formatting must not block a
            # response independently confirmed to be syntactically invalid.
            if m.get('error'):
                return 'syntax_failed'
        return 'unresolved'
    except (OSError, ValueError, TypeError, KeyError):
        return 'unresolved'


def block(t, step, repeat):
    run = folder(t, step, repeat)
    run.mkdir(parents=True, exist_ok=False)
    m = {'protocol':'v2', 'contract_sha256':digest(), 'treatment':t, 'repeat':repeat,
         'step':step, 'skill':SKILLS[step-1], 'reuse':True,
         'library': {2:'python',4:'sklearn',6:'ezr'}[t],
         'status':'blocked_by_predecessor', 'blocked_by_step':step-1,
         'error':'No valid immediately preceding reuse snapshot. No Claude call made.',
         'estimated_cost_usd':None, 'generation_seconds':None, 'usage':None}
    (run/'metrics.json').write_text(json.dumps(m, indent=2)+'\n')


def one(t, step, args):
    repeat = args.repeat
    run = folder(t, step, repeat)
    status = classify_record(t, step, repeat)
    if status in RECORDED:
        return t, status, f'EXISTING ({status})'
    if status != 'missing':
        return t, 'unresolved', 'INCOMPLETE OR MISMATCHED RECORD; not overwritten'
    if t % 2 == 0 and step > 1:
        predecessor = classify_record(t, step-1, repeat)
        if predecessor in {'syntax_failed', 'blocked'}:
            block(t, step, repeat)
            return t, 'blocked', 'BLOCKED by prior reuse generation; no Claude call'
        if predecessor != 'generated':
            return t, 'unresolved', f'Predecessor is {predecessor}; no Claude call'
    cmd = [sys.executable, str(ROOT/'scripts/generate_v2.py'), '--treatment',str(t),
           '--step',str(step),'--repeat',str(repeat),'--cap',str(args.cap)]
    start = time.perf_counter()
    try:
        p = subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=360)
        rc,out=p.returncode,p.stdout+'\n'+p.stderr
    except subprocess.TimeoutExpired:
        rc,out=124,'Generator timed out; incomplete run preserved.'
    run.mkdir(parents=True,exist_ok=True)
    (run/'batch-log.txt').write_text(out)
    status=classify_record(t,step,repeat)
    if status=='syntax_failed':
        return t,status,'SYNTAX FAILURE recorded; no retry'
    if rc==0 and status=='generated':
        return t,status,f'GENERATED ({time.perf_counter()-start:.1f}s)'
    return t,'unresolved',f'INCOMPLETE/INFRASTRUCTURE ERROR ({rc}): {out[-700:]}'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repeat',type=int,default=2)
    p.add_argument('--workers',type=int,choices=[1,2,3],default=2)
    p.add_argument('--cap',type=float,default=.50)
    p.add_argument('--start',type=int,choices=range(1,11),default=1)
    p.add_argument('--end',type=int,choices=range(1,11),default=10)
    args=p.parse_args()
    if args.repeat<2 or args.start>args.end or args.cap<=0:
        p.error('repeat >= 2, valid step interval, positive cap required')
    print('v2 generation only; no generated program will be executed.',flush=True)
    for step in range(args.start,args.end+1):
        print(f'\nSTEP {step}: {SKILLS[step-1]}',flush=True)
        unresolved=False
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs=[pool.submit(one,t,step,args) for t in range(1,7)]
            for job in as_completed(jobs):
                t,status,msg=job.result()
                unresolved |= status not in RECORDED
                print(f'T{t}: {msg}',flush=True)
        if unresolved:
            raise SystemExit('Stopped at step boundary because of unresolved generation/infrastructure error. Evidence retained.')
    counts={state:0 for state in ('generated','syntax_failed','blocked','missing','unresolved')}
    for step in range(1,11):
        for t in range(1,7):
            counts[classify_record(t,step,args.repeat)]+=1
    print('\nFULL REPETITION STATUS: '+json.dumps(counts),flush=True)
    if counts['missing'] or counts['unresolved']:
        print('Repetition not complete; continue from the missing steps.',flush=True)
    else:
        print('All 60 planned outcomes recorded. Review generated source before cloud evaluation.',flush=True)

if __name__=='__main__':
    main()
