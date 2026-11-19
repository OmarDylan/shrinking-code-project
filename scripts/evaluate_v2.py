"""Evaluate immutable snapshots, preserving current-skill and regression results separately."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SKILLS='remember guess certify flow blame justify choose fix spot race'.split()


def dumpcsv(path, rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def check(source, skill, folder, env):
    result=folder/f'{skill}.json'
    log=folder/f'{skill}.txt'
    result.unlink(missing_ok=True)
    cmd=[sys.executable,str(ROOT/'tests'/f'check_{skill}.py'),str(source)]
    if skill!='remember': cmd.append(str(result))
    start=time.perf_counter()
    try:
        p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,env=env,timeout=130)
        output=p.stdout+'\n'+p.stderr
        ok=p.returncode==0 and bool(re.search(r'(?m)^PASS(?:\s|$)',p.stdout))
    except subprocess.TimeoutExpired:
        output='Evaluator timed out'; ok=False
    wall=time.perf_counter()-start
    log.write_text(output)
    payload={}
    if result.exists():
        try: payload=json.loads(result.read_text())
        except (ValueError,OSError): pass
    summary=payload.get('summary',{})
    runtime=payload.get('runtime',summary.get('runtime'))
    if runtime is None:
        match=re.search(r'Runtime:\s*([0-9.]+)s',output)
        if match: runtime=float(match.group(1))
    return ok, wall, runtime, summary, payload


def main():
    p=argparse.ArgumentParser(); p.add_argument('--repeat',type=int,required=True); args=p.parse_args()
    if args.repeat<2: p.error('v2 requires repeat >=2')
    trial=f'r{args.repeat:02d}'
    out=ROOT/'experiments'/'v2-results'/trial; out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for k in list(env):
        if any(s in k.upper() for s in ('TOKEN','SECRET','API_KEY','PASSWORD','CREDENTIAL')): env.pop(k,None)
    env['PYTHONPATH']=str(ROOT/'vendor')
    env['PYTHONHASHSEED']='0'
    env['OMP_NUM_THREADS']='1'; env['OPENBLAS_NUM_THREADS']='1'; env['MKL_NUM_THREADS']='1'
    digest=hashlib.sha256((ROOT/'docs/acceptance-v2.md').read_bytes()).hexdigest()
    records=[]
    for step,skill in enumerate(SKILLS,1):
        for t in range(1,7):
            run=ROOT/'experiments'/'runs'/f't{t}'/trial/f'step{step:02d}'
            record={'protocol':'v2','repeat':args.repeat,'treatment':f'T{t}','step':step,'skill':skill,'individual_status':'MISSING','regression_status':'NOT_EVALUATED'}
            records.append(record)
            try:
                m=json.loads((run/'metrics.json').read_text())
            except (OSError,ValueError): continue
            u=m.get('usage') or {}
            record.update(generation_status=m.get('status'),new_loc=m.get('added_physical_lines'),removed_loc=m.get('removed_physical_lines'),total_loc=m.get('cumulative_physical_lines'),input_tokens=sum(u.get(k,0) or 0 for k in ('input_tokens','cache_creation_input_tokens','cache_read_input_tokens')),cache_creation_tokens=u.get('cache_creation_input_tokens'),cache_read_tokens=u.get('cache_read_input_tokens'),output_tokens=u.get('output_tokens'),estimated_cost_usd=m.get('estimated_cost_usd'),generation_seconds=m.get('generation_seconds'),assigned_library_imported=m.get('assigned_library_imported'),imports=';'.join(m.get('imports',[])))
            source=run/'generated.py'
            if m.get('status')!='generated': record['individual_status']='GENERATION_FAILED'; continue
            if not source.exists() or m.get('protocol')!='v2' or m.get('contract_sha256')!=digest or m.get('source_sha256')!=hashlib.sha256(source.read_bytes()).hexdigest():
                record['individual_status']='SNAPSHOT_MISMATCH'; continue
            folder=out/f't{t}'/f'step{step:02d}'; folder.mkdir(parents=True,exist_ok=True)
            checks={}
            for name in ([skill] if t%2 else SKILLS[:step]):
                ok,wall,runtime,summary,payload=check(source,name,folder,env)
                checks[name]='PASS' if ok else 'FAIL'
                print(f'{trial} T{t} Step {step} {name}: {checks[name]}',flush=True)
                if name==skill:
                    record.update(individual_status=checks[name],test_wall_seconds=wall,execution_seconds=runtime)
                    for k,v in summary.items():
                        if isinstance(v,(int,float,bool)): record['quality_'+k]=v
                    if skill=='race' and ok:
                        for metric,values in payload.get('comparisons',{}).items():
                            if isinstance(values,dict):
                                for k,v in values.items():
                                    if isinstance(v,(int,float,bool)): record[f'quality_{metric}_{k}']=v
            record['regression_status']=('PASS' if all(v=='PASS' for v in checks.values()) else 'FAIL') if t%2==0 else 'NOT_APPLICABLE'
            record['checks']=';'.join(f'{k}:{v}' for k,v in checks.items())
    dumpcsv(out/'all_steps.csv',records)
    totals=[]
    for t in range(1,7):
        rows=[r for r in records if r['treatment']==f'T{t}']
        evaluated=[r for r in rows if r['individual_status'] in ('PASS','FAIL')]
        def total(k):
            vals=[r[k] for r in rows if r.get(k) is not None]; return sum(vals) if vals else None
        totals.append({'treatment':f'T{t}','generated':sum(r.get('generation_status')=='generated' for r in rows),'evaluated':len(evaluated),'individual_passes':sum(r['individual_status']=='PASS' for r in evaluated),'regression_steps_passed':sum(r['regression_status']=='PASS' for r in rows) if t%2==0 else None,'final_loc':rows[-1].get('total_loc'),'total_input_tokens':total('input_tokens'),'total_output_tokens':total('output_tokens'),'estimated_total_cost_usd':total('estimated_cost_usd'),'total_generation_seconds':total('generation_seconds')})
    dumpcsv(out/'treatment_totals.csv',totals)
    print((out/'treatment_totals.csv').read_text())
    if any(r['individual_status'] not in ('PASS','FAIL') for r in records):
        raise SystemExit('At least one missing, failed-generation or invalid snapshot record; artifacts retained.')

if __name__=='__main__': main()
