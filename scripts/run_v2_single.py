#!/usr/bin/env python3
"""Run just one frozen V2 repetition; never sleep or auto-start another."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import subprocess
import sys
from zoneinfo import ZoneInfo

from final_v2_pipeline import preflight, status, cost, stage_and_commit
from run_v2 import ROOT, RECORDED
from resume_v2_after_limits import exclusive, outstanding, archive_rejections, commit_archive


def show(rep):
    counts = Counter(status(rep).values())
    print(f'R{rep:02d}: ' + ', '.join(f'{k}={counts[k]}' for k in
          ('generated', 'syntax_failed', 'blocked', 'missing', 'unresolved')), flush=True)
    return counts


def inspect_limits(archive_expired=False):
    """Only archive the verified zero-token refusals, after their reset."""
    rejected = outstanding()
    if not rejected:
        return True
    until = max(x[1] for x in rejected)
    print(f'{len(rejected)} verified zero-token session-limit refusals.\n'
          f'Resume no earlier than {until.astimezone(ZoneInfo("America/New_York")):%Y-%m-%d %I:%M %p %Z}.', flush=True)
    if until > datetime.now(timezone.utc) or not archive_expired:
        print('No automatic wait or retry. Re-run the same repetition after the reset.', flush=True)
        return False
    if outstanding() != rejected:
        raise SystemExit('Rejected records changed; inspect manually.')
    archive_rejections(rejected)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repeat', type=int, choices=range(3,12), required=True)
    p.add_argument('--budget-usd', type=float, default=50, help='Cumulative soft estimated API-equivalent budget for r03-r11.')
    a = p.parse_args()
    if not 0 < a.budget_usd < 10000:
        p.error('Positive finite budget required.')
    rep = a.repeat
    with exclusive('csc491-limit-resume.lock'), exclusive('csc491-final.lock'):
        preflight()
        commit_archive()
        show(rep)
        if not inspect_limits(archive_expired=True):
            return 2
        for step in range(1,11):
            states = status(rep)
            if 'unresolved' in states.values():
                raise SystemExit('Unresolved record; stop without retrying.')
            missing = [t for t in range(1,7) if states[t,step] == 'missing']
            if not missing:
                continue
            calls = [t for t in missing if not (t%2==0 and step>1 and states[t,step-1] in ('syntax_failed','blocked'))]
            spent = cost()
            if spent + .50*len(calls) > a.budget_usd:
                raise SystemExit(f'Soft budget checkpoint: ${spent:.2f} recorded; {len(calls)} possible next calls.')
            print(f'R{rep:02d} step {step:02d}; recorded additional estimated cost ${spent:.3f}', flush=True)
            result = subprocess.run([sys.executable, str(ROOT/'scripts/run_v2.py'), '--repeat', str(rep), '--workers','3','--cap','.50','--start',str(step),'--end',str(step)], cwd=ROOT)
            if result.returncode or any(status(rep)[t,step] not in RECORDED for t in range(1,7)):
                show(rep)
                if outstanding():
                    inspect_limits()
                else:
                    print('Unexpected runner error. Records retained; inspect before restarting.', flush=True)
                return 2
        if set(status(rep).values())-RECORDED:
            raise SystemExit(f'R{rep:02d} is incomplete; no completion commit.')
        stage_and_commit(rep)
        show(rep)
        print(f'R{rep:02d} COMPLETE. Stopped intentionally. No other repetition started.', flush=True)
        return 0

if __name__ == '__main__':
    sys.exit(main())
