#!/usr/bin/env python3
"""One resumable local generation -> GitHub evaluation -> download pipeline."""
import argparse, fcntl, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
from run_v2 import classify_record, folder, RECORDED
ROOT=Path(__file__).resolve().parents[1]
REPO='OmarDylan/shrinking-code-project'
REPS=range(2,12)
PATCH=['scripts/final_v2_pipeline.py','scripts/analyze_v2_final.py','scripts/Dockerfile.v2-final','.github/workflows/csc491-v2-final.yml','docs/final-v2-pipeline.md']
WORKFLOW='csc491-v2-final.yml'

def run(*args, capture=False):
    cmd=[str(x) for x in args]
    if capture: return subprocess.check_output(cmd,cwd=ROOT,text=True).strip()
    subprocess.run(cmd,cwd=ROOT,check=True)

def git(*args): return run('git',*args,capture=True)

def status(rep): return {(t,s):classify_record(t,s,rep) for t in range(1,7) for s in range(1,11)}

def cost():
    total=0.
    for rep in range(3,12):
        for t in range(1,7):
            for s in range(1,11):
                p=folder(t,s,rep)/'metrics.json'
                if not p.exists(): continue
                m=json.loads(p.read_text())
                if m.get('status')=='blocked_by_predecessor': continue
                x=m.get('estimated_cost_usd')
                if x is None: raise SystemExit(f'Unknown attempt cost: {p}; inspect before resuming.')
                total+=float(x)
    return total

def stage_and_commit(rep):
    paths=[]
    for t in range(1,7):
        a=ROOT/f'experiments/runs/t{t}/r{rep:02d}'
        b=ROOT/f'src/t{t}/r{rep:02d}'
        if a.is_dir(): paths.append(str(a.relative_to(ROOT)))
        if b.is_dir(): paths.append(str(b.relative_to(ROOT)))
    run('git','add','-f','--',*paths)
    if git('diff','--cached','--name-only'):
        run('git','commit','-m',f'Record frozen V2 repetition r{rep:02d}')

def review_manifest():
    p=ROOT/'.git/csc491-final-review.txt'
    lines=['All generated snapshots (r02–r11). Review before cloud execution. Raw code is untrusted.','']
    for rep in REPS:
        for t in range(1,7):
            for s in range(1,11):
                d=folder(t,s,rep); f=d/'generated.py'; st=classify_record(t,s,rep)
                lines.append(f'{st:14} {str(f.relative_to(ROOT))}'+(f' {hashlib.sha256(f.read_bytes()).hexdigest()}' if f.exists() else ''))
    p.write_text('\n'.join(lines)+'\n'); return p

def lookup(sha):
    rows=json.loads(run('gh','run','list','--repo',REPO,'--workflow',WORKFLOW,'--limit','100','--json','databaseId,displayTitle,headSha,status,conclusion',capture=True))
    matches=[x for x in rows if x['headSha']==sha and x['displayTitle']==f'CSC491 V2 final {sha}']
    return max(matches,key=lambda x:x['databaseId']) if matches else None

def preflight():
    for bin in ('git','gh','claude'):
        if not shutil.which(bin): raise SystemExit(f'Missing executable: {bin}')
    if git('branch','--show-current')!='main': raise SystemExit('Run on existing main branch.')
    if git('remote','get-url','origin').rstrip('/').removesuffix('.git')!=f'https://github.com/{REPO}': raise SystemExit('Origin is not the expected private repository.')
    if run('gh','api','user','--jq','.login',capture=True)!='OmarDylan': raise SystemExit('Switch active gh account to OmarDylan first.')
    if run('gh','repo','view',REPO,'--json','visibility','--jq','.visibility',capture=True)!='PRIVATE': raise SystemExit('Refusing to push to a non-private repository.')
    if git('diff','--cached','--name-only'): raise SystemExit('There are staged changes; commit or unstage them before the pipeline.')
    for ref,paths in [('protocol-v2',['docs/acceptance-v2.md','scripts/generate_v2.py','vendor','tests','data']),('1de1c9e',['scripts/run_v2.py','scripts/evaluate_v2.py'])]:
        if git('diff','--name-only',ref,'--',*paths): raise SystemExit(f'Frozen protocol differs from {ref}: inspect before continuing.')
    # Check that the validation run was truly complete; do not regenerate it.
    if set(status(2).values())-RECORDED: raise SystemExit('r02 is not fully recorded; inspect.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--budget-usd',type=float,default=50,help='Conservative soft API-equivalent budget for r03–r11, not subscription charges');a=p.parse_args()
    if not 0<a.budget_usd<10000: p.error('positive finite budget required')
    os.chdir(ROOT)
    lock=(ROOT/'.git/csc491-final.lock').open('a+')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('Another final pipeline is running.')
    preflight()
    run('git','add','--',*PATCH)
    if git('diff','--cached','--name-only'): run('git','commit','-m','Add resumable V2 final evaluation and statistics pipeline')
    # New calls only; original .50 per-call setting is unchanged.
    for rep in range(3,12):
        for step in range(1,11):
            states=status(rep)
            if 'unresolved' in states.values(): raise SystemExit(f'r{rep:02d} has an unresolved generation record. Retained; inspect before any new call.')
            missing=[t for t in range(1,7) if states[t,step]=='missing']
            if not missing: continue
            calls=[t for t in missing if not (t%2==0 and step>1 and states[t,step-1] in ('syntax_failed','blocked'))]
            spent=cost()
            if spent+.50*len(calls)>a.budget_usd:
                raise SystemExit(f'Soft budget checkpoint: ${spent:.2f} recorded; next step up to {len(calls)} calls (nominal cap $.50 each). Re-run with an explicitly larger --budget-usd only if desired.')
            print(f'r{rep:02d} step {step:02d}; recorded additional estimated cost ${spent:.3f}',flush=True)
            run(sys.executable,'scripts/run_v2.py','--repeat',rep,'--workers',3,'--cap','.50','--start',step,'--end',step)
            if any(classify_record(t,step,rep) not in RECORDED for t in range(1,7)): raise SystemExit('Step not fully recorded; stopped.')
        if set(status(rep).values())-RECORDED: raise SystemExit(f'r{rep:02d} incomplete.')
        stage_and_commit(rep)
    sha=git('rev-parse','HEAD');manifest=review_manifest();reviewed=ROOT/'.git/csc491-final-reviewed-sha'
    if not reviewed.exists() or reviewed.read_text().strip()!=sha:
        print(f'\nAll 600 planned outcomes recorded. Additional estimated cost ${cost():.3f}.\nReview generated source listed in {manifest}.\nThe pipeline will not run generated code locally. Cloud execution is restricted but not a hardened oracle.\nAfter reviewing source, type REVIEWED to push and evaluate.',flush=True)
        if input('Approval: ').strip()!='REVIEWED': raise SystemExit('Not approved. Re-run the same command after review; generations are retained.')
        reviewed.write_text(sha+'\n')
    run('git','push','origin','main')
    remote=git('ls-remote','origin','refs/heads/main').split()[0]
    if remote!=sha: raise SystemExit('Remote main does not match reviewed batch.')
    found=lookup(sha)
    if found is None:
        run('gh','workflow','run',WORKFLOW,'--repo',REPO,'--ref','main','-f',f'batch_id={sha}')
        for _ in range(40):
            time.sleep(3);found=lookup(sha)
            if found: break
    if found is None: raise SystemExit('Workflow not visible yet. Re-run later; check Actions before dispatching manually.')
    rid=str(found['databaseId']); print(f'Cloud run: https://github.com/{REPO}/actions/runs/{rid}',flush=True)
    if found['status']=='completed' and found['conclusion']!='success':
        raise SystemExit(f'Cloud run {rid} failed. Inspect gh run view {rid} --log-failed; after fixing infrastructure use gh run rerun {rid} --failed. No new Claude generations required.')
    run('gh','run','watch',rid,'--repo',REPO,'--exit-status')
    dest=ROOT/'experiments/downloads'/f'final-v2-{rid}';dest.mkdir(parents=True,exist_ok=True)
    # Download final report separately in a fresh temporary directory; resume avoids an overwrite conflict.
    import tempfile
    with tempfile.TemporaryDirectory(dir=dest) as td:
        run('gh','run','download',rid,'--repo',REPO,'--dir',td)
        for item in Path(td).iterdir():
            target=dest/item.name
            if target.exists(): shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(item),target)
    report=dest/'csc491-v2-final-analysis'/'REPORT.md'
    if not report.is_file(): raise SystemExit('Analysis report missing; inspect artifacts.')
    archive=shutil.make_archive(str(Path.home()/'Desktop'/'CSC491_FINAL_V2_RESULTS'),'zip',dest)
    print(report.read_text());print(f'\nFinal archive: {archive}',flush=True)

if __name__=='__main__':main()
