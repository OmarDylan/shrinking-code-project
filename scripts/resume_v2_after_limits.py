#!/usr/bin/env python3
"""Resume the existing V2 pipeline after confirmed zero-token session-limit refusals.

No changes to prompts, model, acceptance tests, syntax-failure policy or budget.
Run on the user's Mac. Never bypass a provider usage limit.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from run_v2 import classify_record, digest, folder, ROOT

ARCHIVE = ROOT / 'experiments/infra_rejections/session-limit'
RESET = re.compile(r"You've hit your session limit\s*·\s*resets\s+(\d{1,2})(?::(\d{2}))?\s*([ap]m)\s*\(([^()]+)\)\s*", re.I)
TOKEN_FIELDS = ('input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens')


def reset_time(message, observed):
    match = RESET.fullmatch(message.strip()) if isinstance(message, str) else None
    if not match:
        raise ValueError('Not a recognized session-limit refusal with a clock reset')
    hour, minute, ampm, zone = match.groups()
    h, m = int(hour), int(minute or 0)
    if not 1 <= h <= 12 or not 0 <= m <= 59:
        raise ValueError('Invalid reset clock time')
    tz = ZoneInfo(zone)
    local = observed.astimezone(tz)
    target = local.replace(hour=h % 12 + (12 if ampm.lower() == 'pm' else 0), minute=m, second=0, microsecond=0)
    if target < local - timedelta(minutes=2):
        target += timedelta(days=1)
    if target - local > timedelta(hours=12):
        raise ValueError('Ambiguous reset date or unusually long session limit; inspect manually')
    return target.astimezone(timezone.utc) + timedelta(minutes=5)


def inspect_rejection(t, s, r):
    p = folder(t, s, r)
    mp, rp = p/'metrics.json', p/'response.json'
    m, response = json.loads(mp.read_text()), json.loads(rp.read_text())
    if (m.get('protocol'), m.get('contract_sha256'), m.get('treatment'), m.get('step'), m.get('repeat')) != ('v2', digest(), t,s,r):
        raise ValueError('Record identity or frozen contract mismatch')
    if m.get('prompt_sha256') != hashlib.sha256((p/'prompt.txt').read_bytes()).hexdigest():
        raise ValueError('Prompt integrity mismatch')
    if m.get('status') != 'generation_failed' or m.get('returncode') != 1 or (p/'generated.py').exists():
        raise ValueError('Not an ungenerated rejected call')
    if m.get('estimated_cost_usd') != 0 or response.get('total_cost_usd') != 0:
        raise ValueError('Nonzero or unknown cost; do not retry automatically')
    for usage in (m.get('usage'), response.get('usage')):
        if not isinstance(usage,dict) or any(usage.get(k) != 0 for k in TOKEN_FIELDS):
            raise ValueError('Nonzero or unknown token usage; do not retry automatically')
    when = datetime.fromtimestamp(rp.stat().st_mtime, timezone.utc)
    return p, reset_time(response.get('result'), when)


def outstanding():
    rejected, errors = [], []
    for r in range(3,12):
        for t in range(1,7):
            for s in range(1,11):
                if classify_record(t,s,r) != 'unresolved':
                    continue
                try:
                    rejected.append(inspect_rejection(t,s,r))
                except Exception as exc:
                    errors.append(f'R{r:02d} T{t} Step {s:02d}: {exc}')
    if errors:
        raise SystemExit('Manual inspection required; no records moved:\n'+'\n'.join(errors))
    return rejected


def git(*args, capture=False):
    if capture:
        return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    subprocess.run(['git',*args],cwd=ROOT,check=True)


@contextmanager
def exclusive(name):
    with (ROOT/'.git'/name).open('a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another CSC491 pipeline is active. Stop the duplicate process.')
        yield


def commit_archive():
    if git('diff','--cached','--name-only',capture=True):
        raise SystemExit('Unrelated staged changes exist; commit or unstage them before resuming.')
    if ARCHIVE.is_dir() and any(ARCHIVE.rglob('metrics.json')):
        git('add','-f','--',str(ARCHIVE.relative_to(ROOT)))
        if git('diff','--cached','--name-only',capture=True):
            git('commit','-m','Preserve zero-token Claude session-limit refusals')


def archive_rejections(rejected):
    # Keep rejected attempts outside the active run folders. Never delete them.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if git('diff','--cached','--name-only',capture=True):
        raise SystemExit('Unrelated staged changes exist; no records moved.')
    for p, _ in rejected:
        rel = p.relative_to(ROOT/'experiments/runs')
        target = ARCHIVE/stamp/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            raise SystemExit(f'Archive path already exists: {target}')
        shutil.move(str(p),str(target))
        print(f'Preserved {rel} -> {target.relative_to(ROOT)}',flush=True)
    commit_archive()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--budget-usd',type=float,default=50)
    args=parser.parse_args()
    if not 0 < args.budget_usd < 10000:
        parser.error('positive finite budget required')
    if git('branch','--show-current',capture=True) != 'main':
        raise SystemExit('Run on the existing main branch.')
    with exclusive('csc491-limit-resume.lock'):
        waits=0
        while True:
            with exclusive('csc491-final.lock'):
                commit_archive()
                rejected=outstanding()
                if rejected:
                    waits += 1
                    if waits > 12:
                        raise SystemExit('Session-limit wait cap reached; all records retained.')
                    until=max(x[1] for x in rejected)
                    delay=max(0,(until-datetime.now(timezone.utc)).total_seconds())
                    print(f'{len(rejected)} verified zero-token refusals. Resume no earlier than {until.astimezone(ZoneInfo("America/New_York")):%Y-%m-%d %I:%M %p %Z}.',flush=True)
                    if delay:
                        print('Waiting locally for the provider reset plus a five-minute margin. No Claude calls during the wait.',flush=True)
                        time.sleep(delay)
                    # Revalidate before moving anything.
                    current = outstanding()
                    if current != rejected:
                        raise SystemExit('Rejected records changed during wait; inspect manually.')
                    archive_rejections(current)
            result=subprocess.run([sys.executable,str(ROOT/'scripts/final_v2_pipeline.py'),'--budget-usd',str(args.budget_usd)],cwd=ROOT)
            if result.returncode==0:
                return
            with exclusive('csc491-final.lock'):
                if not outstanding():
                    raise SystemExit(result.returncode)
            print('Pipeline paused at a verified session limit; will resume after the reported reset.',flush=True)


if __name__=='__main__':
    main()
