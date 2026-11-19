"""Generate a new v2 repetition locally; never execute generated code.
Checkpoint data are immutable per successful generation; the original pilot stays intact.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SKILLS = 'remember guess certify flow blame justify choose fix spot race'.split()


def one(t, step, args):
    run = ROOT / f'experiments/runs/t{t}/r{args.repeat:02d}/step{step:02d}'
    if run.exists():
        try:
            metrics = json.loads((run/'metrics.json').read_text())
            snapshot = (run/'generated.py').read_bytes()
            contract = hashlib.sha256((ROOT/'docs/acceptance-v2.md').read_bytes()).hexdigest()
            if (metrics.get('status') == 'generated' and metrics.get('protocol') == 'v2'
                    and metrics.get('contract_sha256') == contract
                    and metrics.get('source_sha256') == hashlib.sha256(snapshot).hexdigest()):
                return t, True, 'existing validated snapshot'
        except (OSError, ValueError, KeyError):
            pass
        return t, False, 'existing incomplete or mismatched record; not overwritten'
    cmd = [sys.executable, str(ROOT/'scripts/generate_v2.py'), '--treatment', str(t),
           '--step', str(step), '--repeat', str(args.repeat), '--cap', str(args.cap)]
    start = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=360)
        rc, out = p.returncode, p.stdout + '\n' + p.stderr
    except subprocess.TimeoutExpired:
        rc, out = 124, 'Generator timed out; incomplete run preserved.'
    run.mkdir(parents=True, exist_ok=True)
    (run/'batch-log.txt').write_text(out)
    if rc:
        return t, False, f'generator exit {rc}: {out[-700:]}'
    return t, True, f'generated ({time.perf_counter()-start:.1f}s)'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repeat', type=int, default=2)
    p.add_argument('--workers', type=int, choices=[1,2,3], default=2)
    p.add_argument('--cap', type=float, default=.50)
    p.add_argument('--start', type=int, choices=range(1,11), default=1)
    p.add_argument('--end', type=int, choices=range(1,11), default=10)
    args=p.parse_args()
    if args.repeat < 2 or args.start > args.end or args.cap <= 0:
        p.error('repeat >= 2, valid step interval, positive cap required')
    print('v2 generation only; no generated program will be executed.', flush=True)
    for step in range(args.start,args.end+1):
        print(f'\nSTEP {step}: {SKILLS[step-1]}', flush=True)
        failed=False
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs=[pool.submit(one,t,step,args) for t in range(1,7)]
            for job in as_completed(jobs):
                t,ok,msg=job.result()
                failed |= not ok
                print(f'T{t}: {msg}',flush=True)
        if failed:
            raise SystemExit('Stopped at step boundary. Records retained; inspect before deciding on a new repetition.')
    print('\nGeneration complete. Review source and logs, then commit and dispatch cloud evaluation.',flush=True)

if __name__=='__main__':
    main()
