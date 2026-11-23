#!/usr/bin/env python3
"""Reproduce CSC491 V2/T7 descriptive study from archived evidence.

Stdlib only. T7 is a single AI-assisted case, NOT an unaided human control.
No generated student program is executed by this script.
"""
import argparse
import csv
import io
import json
from pathlib import Path
from statistics import mean
from zipfile import ZipFile


def read_csv_from_zip(zip_path, suffix):
    with ZipFile(zip_path) as z:
        choices = [p for p in z.namelist() if p.endswith(suffix)]
        if len(choices) != 1:
            raise RuntimeError(f"Expected one match for {suffix}, found {choices}")
        return list(csv.DictReader(io.StringIO(z.read(choices[0]).decode('utf-8'))))


def read_json_from_zip(zip_path, suffix):
    with ZipFile(zip_path) as z:
        choices = [p for p in z.namelist() if p.endswith(suffix)]
        if len(choices) != 1:
            raise RuntimeError(f"Expected one match for {suffix}, found {choices}")
        return json.loads(z.read(choices[0]).decode('utf-8'))


def numeric(value):
    if value in ('', None):
        return None
    return float(value)


def upper_median(values):
    vals = sorted(v for v in values if v is not None)
    return vals[len(vals)//2] if vals else None


def form(v, nd=3):
    return f"{v:.{nd}f}" if v is not None else 'NA'


def write_csv(path, records, fields):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--claude-zip', required=True, type=Path)
    ap.add_argument('--t7-zip', required=True, type=Path)
    ap.add_argument('--out', type=Path, default=Path('reports/final-study/data'))
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    rows=read_csv_from_zip(args.claude_zip, '/v2-recorded-analysis/run_metrics.csv')
    assert len(rows)==24 and sorted(set(int(r['repeat']) for r in rows)) == [2,3,4,5]
    tally = {'passed':0, 'syntax_failed':0, 'blocked':0, 'test_failed':0}
    for r in rows:
        tally['passed'] += int(r['passes'])
        tally['syntax_failed'] += int(r['syntax_failures'])
        tally['blocked'] += int(r['blocked'])
        tally['test_failed'] += int(r['competency_failures'])
    assert sum(tally.values())==240, tally
    assert tally=={'passed':221,'syntax_failed':9,'blocked':5,'test_failed':5},tally
    summary=[]
    for t in range(1,7):
        subset=[r for r in rows if r['treatment']==f'T{t}']
        loc=[numeric(r['complete_loc']) for r in subset]
        x={
            'treatment':f'T{t}',
            'repetitions':4,
            'competency_passes':sum(int(r['passes']) for r in subset),
            'syntax_failures':sum(int(r['syntax_failures']) for r in subset),
            'test_failures':sum(int(r['competency_failures']) for r in subset),
            'blocked':sum(int(r['blocked']) for r in subset),
            'complete_loc_n':sum(v is not None for v in loc),
            'complete_loc_upper_median':form(upper_median(loc),0),
            'cost_usd_upper_median':form(upper_median([numeric(r['cost']) for r in subset]),4),
            'generation_seconds_upper_median':form(upper_median([numeric(r['generation_seconds']) for r in subset]),1),
        }
        summary.append(x)
    t7 = read_json_from_zip(args.t7_zip,'/summary.json')
    assert t7['passed']==10 and t7['required']==10
    assert len(t7['checks'])==10 and all(c['status']=='PASS' and c['returncode']==0 for c in t7['checks'])
    prov=t7['provenance']
    assert prov['not_human_only'] is True and prov['repetition']==1
    counts= [v['physical_lines'] for v in prov['files'].values()]
    assert sorted(counts)==[437,580,920]
    paired=[]
    for a,b in [(1,2),(3,4),(5,6)]:
        for rep in range(2,6):
            left=next(r for r in rows if r['treatment']==f'T{a}' and int(r['repeat'])==rep)
            right=next(r for r in rows if r['treatment']==f'T{b}' and int(r['repeat'])==rep)
            left_loc=numeric(left['complete_loc']);right_loc=numeric(right['complete_loc'])
            paired.append({
                'repetition':f'R{rep:02d}', 'pair':f'T{a}-T{b}',
                'no_reuse_complete_LOC':int(left_loc) if left_loc is not None else '',
                'reuse_complete_LOC':int(right_loc) if right_loc is not None else '',
                'reuse_LOC_change_pct':form(100*(right_loc/left_loc-1),2) if left_loc and right_loc else '',
                'no_reuse_cost_usd':form(numeric(left['cost']),5),
                'reuse_cost_usd':form(numeric(right['cost']),5),
                'cost_ratio_reuse_to_no_reuse':form(numeric(right['cost'])/numeric(left['cost']),3),
                'no_reuse_generation_s':form(numeric(left['generation_seconds']),2),
                'reuse_generation_s':form(numeric(right['generation_seconds']),2),
                'time_ratio_reuse_to_no_reuse':form(numeric(right['generation_seconds'])/numeric(left['generation_seconds']),3),
            })
    write_csv(args.out/'study_summary.csv',summary,list(summary[0].keys()))
    write_csv(args.out/'paired_reuse.csv',paired,list(paired[0].keys()))
    write_csv(args.out/'run_metrics.csv',rows,list(rows[0].keys()))
    metadata={
        'source_claude_zip':args.claude_zip.name,
        'source_t7_zip':args.t7_zip.name,
        'v2_repetitions':[2,3,4,5],
        'claude_outcomes':tally,
        'claude_estimated_api_equivalent_cost_usd':sum(float(r['cost']) for r in rows),
        'T7_assisted':{
            'passed':10,
            'repetitions':1,
            'new_adapter_LOC':437,
            'reused_homework_LOC':1500,
            'total_footprint_LOC':1937,
            'not_human_only':True,
            'generation_cost_reported':False,
            'development_time_reported':False,
        },
        'interpretation': 'Descriptive analysis only; seven-way statistical ranking not supported.',
    }
    (args.out/'audit.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Saved results to',args.out)
    print('Outcome tally:',tally)
    print('Estimated API-equivalent cost: $',round(metadata['claude_estimated_api_equivalent_cost_usd'],2))
    print('Matched complete LOC pairs:',sum(bool(p['reuse_LOC_change_pct']) for p in paired))
    for r in summary: print(r['treatment'],f"pass={r['competency_passes']}/40",f"LOC_n={r['complete_loc_n']}",f"LOC={r['complete_loc_upper_median']}")

if __name__=='__main__':main()
