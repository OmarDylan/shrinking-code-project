#!/usr/bin/env python3
"""Course-style ranking of independent V2 generation repetitions. Stdlib only."""
import argparse, csv, html, itertools, math, statistics as st
from pathlib import Path

SKILLS='remember guess certify flow blame justify choose fix spot race'.split()
TS=[f'T{x}' for x in range(1,7)]
NAMES=dict(zip(TS,['Python','Python + reuse','sklearn','sklearn + reuse','EZR available','EZR available + reuse']))
# labels, direction, minimum coverage handled by rank()
METRICS={
 'passes':('Individual skills passed (of 10)',True), 'generated':('Syntax-valid snapshots (of 10)',True),
 'final_passes':('Skills retained in final implementation (of 10)',True),
 'complete_loc':('Final physical LOC, complete generation only',False),
 'all_pass_loc':('Final physical LOC, all 10 passed and retained only',False),
 'added_loc_complete':('Gross added physical LOC, complete generation only',False),
 'cost':('Estimated API-equivalent USD, attempted calls',False),
 'tokens_in':('Total input tokens including cache, attempted calls',False),
 'tokens_out':('Total output tokens, attempted calls',False),
 'generation_seconds':('Sum of generation seconds, attempted calls',False),
 'complete_cost':('Estimated cost, all 10 generated only',False),
 'complete_generation_seconds':('Generation seconds, all 10 generated only',False),
 'test_wall_complete':('Sum of primary evaluator wall seconds, all ten passed only',False),
 'reported_execution_9':('Sum of reported execution seconds (skills 2–10), all ten passed only',False),
 'guess_mpg_mae':('Guess MPG MAE, passing snapshots only',False),
 'guess_disty_mae':('Guess distance-to-heaven MAE, passing snapshots only',False),
 'spot_f1':('Spot F1, passing snapshots only',True),
 'race_best10':('Race adaptive best at 10 labels, passing snapshots only',False),
 'race_random_best10':('Race random best at 10 labels, passing snapshots only',False),
 'race_gain':('Race random minus adaptive best at 10, passing snapshots only',True),
}

def num(x):
    if x in ('',None): return None
    try: f=float(x); return f if math.isfinite(f) else None
    except (TypeError,ValueError): return None

def med(xs): return sorted(xs)[len(xs)//2]  # course upper middle

def quant(xs,p): return sorted(xs)[min(len(xs)-1,int(len(xs)*p))]

def compare(xs,ys):
    if min(len(xs),len(ys))<2: return None
    n,m=len(xs),len(ys)
    pooled=math.sqrt(((n-1)*st.variance(xs)+(m-1)*st.variance(ys))/(n+m-2))
    cohen=abs(st.mean(xs)-st.mean(ys))/(pooled+1e-32)
    cliffs=abs(sum((x>y)-(x<y) for x in xs for y in ys))/(n*m)
    ks=max(abs(sum(x<=v for x in xs)/n-sum(y<=v for y in ys)/m) for v in set(xs+ys))/math.sqrt((n+m)/(n*m))
    return dict(cohen=cohen,cliffs=cliffs,ks=ks,same=cohen<=.35 and cliffs<=.195 and ks<=1.36)

def rank(bags,big,min_n=5):
    eligible={k:v for k,v in bags.items() if len(v)>=min_n}
    order=sorted(eligible,key=lambda k:((-1 if big else 1)*med(eligible[k]),k))
    out={}; anchor=None; tier=-1
    for k in order:
        if anchor is None or not compare(eligible[anchor],eligible[k])['same']:
            anchor=k; tier+=1
        out[k]=tier
    return out

def write(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        if keys:
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def total(rows,k):
    vals=[num(r.get(k)) for r in rows]
    return sum(vals) if rows and all(x is not None for x in vals) else None

def run_summary(rows):
    rows=sorted(rows,key=lambda r:int(r['step'])); t=rows[0]['treatment']; reuse=int(t[1:])%2==0
    generated=[r for r in rows if r['generation_status']=='generated']
    attempted=[r for r in rows if r['individual_status']!='BLOCKED']
    passes=sum(r['individual_status']=='PASS' for r in rows)
    final_checks=generated[-1].get('checks','') if generated else ''
    final_passes=sum(x.endswith(':PASS') for x in final_checks.split(';')) if reuse else passes
    complete=len(generated)==10; all_pass=complete and passes==10 and final_passes==10
    footprint=num(generated[-1].get('total_loc')) if reuse and generated else total(generated,'new_loc')
    if complete and footprint!=num(rows[-1].get('total_loc')): raise ValueError(f'LOC accounting mismatch {t} {rows[0]["repeat"]}')
    out=dict(repeat=int(rows[0]['repeat']),treatment=t,passes=passes,generated=len(generated),
             final_passes=final_passes,syntax_failures=sum(r['individual_status']=='GENERATION_FAILED' for r in rows),
             blocked=sum(r['individual_status']=='BLOCKED' for r in rows),
             competency_failures=sum(r['individual_status']=='FAIL' for r in rows),
             attempted=len(attempted),regression_passes=sum(r['regression_status']=='PASS' for r in rows) if reuse else None,
             complete_generation=complete,all_pass_and_retained=all_pass,partial_or_final_loc=footprint,
             complete_loc=footprint if complete else None,all_pass_loc=footprint if all_pass else None,
             added_loc_complete=total(generated,'new_loc') if complete else None,
             cost=total(attempted,'estimated_cost_usd'),tokens_in=total(attempted,'input_tokens'),
             tokens_out=total(attempted,'output_tokens'),generation_seconds=total(attempted,'generation_seconds'),
             test_wall_complete=total(rows,'test_wall_seconds') if all_pass else None,
             reported_execution_9=total(rows[1:],'execution_seconds') if all_pass else None,
             assigned_imports=sum(r.get('assigned_library_imported','').lower()=='true' for r in generated) if t not in ('T1','T2') else None)
    out['complete_cost']=out['cost'] if complete else None
    out['complete_generation_seconds']=out['generation_seconds'] if complete else None
    for name,step,key in [('guess_mpg_mae',2,'quality_mpg_error'),('guess_disty_mae',2,'quality_disty_error'),('spot_f1',9,'quality_f1'),('race_best10',10,'quality_best10_adaptive_mean'),('race_random_best10',10,'quality_best10_random_mean')]:
        row=rows[step-1]; out[name]=num(row.get(key)) if row['individual_status']=='PASS' else None
    a,b=out['race_best10'],out['race_random_best10']; out['race_gain']=b-a if a is not None and b is not None else None
    return out

def load(root,reps):
    allrows=[]; environments=[]
    for rep in reps:
        trial=f'r{rep:02d}'; paths=list(root.rglob(f'{trial}/all_steps.csv'))
        if len(paths)!=1: raise ValueError(f'{trial}: expected exactly one all_steps.csv, found {len(paths)}')
        p=paths[0]
        with p.open(newline='') as f: rows=list(csv.DictReader(f))
        seen=set()
        for r in rows:
            t,s=r['treatment'],int(r['step']); key=(t,s)
            if r['protocol']!='v2' or int(r['repeat'])!=rep or t not in TS or not 1<=s<=10 or r['skill']!=SKILLS[s-1] or key in seen: raise ValueError(f'Invalid record: {trial} {key}')
            seen.add(key)
            status=r['individual_status']; gen=r['generation_status']
            if (status,gen) not in {('PASS','generated'),('FAIL','generated'),('GENERATION_FAILED','generation_failed'),('BLOCKED','blocked_by_predecessor')}: raise ValueError(f'Unresolved/invalid outcome: {trial} {key} {status} / {gen}')
            if gen=='generated' and int(t[1:])%2==0 and (r['regression_status'] not in ('PASS','FAIL') or len(r['checks'].split(';'))!=s): raise ValueError(f'Invalid regression evidence {trial} {key}')
        if len(seen)!=60: raise ValueError(f'{trial}: missing records')
        env=p.parent.parent/'environment.txt'
        if not env.exists(): raise ValueError(f'{trial}: environment.txt missing')
        environments.append((trial,env.read_text()))
        allrows+=rows
    return allrows,environments

def svg(path,steps):
    W,H=960,460; L,R,T,B=78,170,45,65; w=W-L-R;h=H-T-B
    colors=['#2463a6','#25a0c3','#c45c28','#e3a130','#6c4c9b','#9a73c6']
    vals=[r['median_added_loc'] for r in steps if r['median_added_loc'] is not None]; top=max(50,math.ceil(max(vals,default=0)/50)*50)
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/><g font-family="Arial, sans-serif" font-size="13" fill="#27364a">', '<text x="78" y="25" font-size="19" font-weight="bold">Median new physical LOC per generated skill</text>']
    for i in range(6):
        v=top*i/5;y=T+h*(1-i/5); o.append(f'<line x1="{L}" y1="{y}" x2="{L+w}" y2="{y}" stroke="#dbe1e8"/><text x="{L-9}" y="{y+4}" text-anchor="end">{v:.0f}</text>')
    for i,s in enumerate(SKILLS):
        x=L+w*i/9; o.append(f'<text x="{x}" y="{T+h+24}" text-anchor="middle" transform="rotate(-30 {x} {T+h+24})">{s}</text>')
    for j,t in enumerate(TS):
        points=[]; rows=[r for r in steps if r['treatment']==t]
        for i,r in enumerate(rows):
            val=r['median_added_loc']
            if val is None:
                if points: o.append(f'<polyline points="{" ".join(points)}" stroke="{colors[j]}" fill="none" stroke-width="2"/>');points=[]
                continue
            x=L+w*i/9;y=T+h*(1-val/top);points.append(f'{x},{y}');o.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{colors[j]}"><title>{t}: {val:g} (n={r["generated_n"]})</title></circle>')
        if points: o.append(f'<polyline points="{" ".join(points)}" stroke="{colors[j]}" fill="none" stroke-width="2"/>')
        y=T+22*j;o.append(f'<line x1="{W-R+15}" y1="{y}" x2="{W-R+39}" y2="{y}" stroke="{colors[j]}" stroke-width="3"/><text x="{W-R+45}" y="{y+4}">{t}</text>')
    o.append('<text x="78" y="448" font-size="12">Conditional on syntax-valid generation; see skill_summary.csv for sample sizes.</text></g></svg>');path.write_text(''.join(o))

def overview(path,runs):
    keys=['passes','complete_loc','cost','generation_seconds','tokens_in','tokens_out']
    W,H=1200,950; o=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/><g font-family="Arial,sans-serif" font-size="13" fill="#22324a"><text x="35" y="30" font-size="22" font-weight="bold">V2 distributions by treatment</text><text x="35" y="52">Each dot is a generation repetition; thick line is upper median. LOC is complete-only.</text>']
    for panel,k in enumerate(keys):
        x0=35+(panel%2)*590; y0=78+(panel//2)*285; left=x0+48; right=x0+540; top=y0+43; bottom=y0+228
        xx={t:[r[k] for r in runs if r['treatment']==t and r[k] is not None] for t in TS}; vals=[v for a in xx.values() for v in a]; vmax=10 if k=='passes' else max(vals,default=1)*1.08 or 1
        o.append(f'<text x="{x0}" y="{y0+12}" font-size="15" font-weight="bold">{html.escape(METRICS[k][0])}</text>')
        for i in range(5):
            val=i*vmax/4; yy=bottom-(bottom-top)*i/4;o.append(f'<line x1="{left}" y1="{yy}" x2="{right}" y2="{yy}" stroke="#e1e6eb"/><text x="{left-7}" y="{yy+4}" text-anchor="end" font-size="11">{val:.3g}</text>')
        for j,t in enumerate(TS):
            cx=left+(right-left)*(j+.5)/6; a=xx[t]; col='#2463a6' if j%2==0 else '#c45c28';
            for i,v in enumerate(sorted(a)):
                yy=bottom-(bottom-top)*v/vmax; dx=((i%5)-2)*3;o.append(f'<circle cx="{cx+dx}" cy="{yy}" r="3" fill="{col}" fill-opacity="0.6"><title>{t}: {v:g}</title></circle>')
            if a:
                yy=bottom-(bottom-top)*med(a)/vmax;o.append(f'<line x1="{cx-16}" y1="{yy}" x2="{cx+16}" y2="{yy}" stroke="{col}" stroke-width="3"/>')
            o.append(f'<text x="{cx}" y="{bottom+18}" text-anchor="middle">{t}</text><text x="{cx}" y="{bottom+34}" text-anchor="middle" font-size="11">n={len(a)}</text>')
    o.append('</g></svg>');path.write_text(''.join(o))

def fmt(v): return '—' if v is None else f'{v:.4g}'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--first',type=int,default=2);ap.add_argument('--last',type=int,default=11);a=ap.parse_args();reps=list(range(a.first,a.last+1));out=a.output;out.mkdir(parents=True,exist_ok=True)
    if a.first<2 or not reps: ap.error('V2 repetitions >= 2 only')
    rows,envs=load(a.input,reps); write(out/'all_steps.csv',rows)
    runs=[run_summary([r for r in rows if int(r['repeat'])==rep and r['treatment']==t]) for rep in reps for t in TS];write(out/'run_metrics.csv',runs)
    stats=[]; pairs=[]
    variants=[('all_v2',runs)]
    if 2 in reps and len(reps)>1: variants.append(('exclude_validation_r02',[r for r in runs if r['repeat']!=2]))
    for cohort,rr in variants:
        for metric,(label,big) in METRICS.items():
            bags={t:[r[metric] for r in rr if r['treatment']==t and r[metric] is not None] for t in TS}; ranks=rank(bags,big)
            for t,x in bags.items():
                stats.append(dict(cohort=cohort,metric=metric,label=label,direction='higher' if big else 'lower',treatment=t,n=len(x),median=med(x) if x else None,q1=quant(x,.25) if x else None,q3=quant(x,.75) if x else None,mean=st.mean(x) if x else None,rank=ranks.get(t)))
            for t,u in itertools.combinations(TS,2):
                c=compare(bags[t],bags[u]) if min(len(bags[t]),len(bags[u]))>=5 else None
                pairs.append(dict(cohort=cohort,metric=metric,a=t,b=u,n_a=len(bags[t]),n_b=len(bags[u]),**(c or {'cohen':None,'cliffs':None,'ks':None,'same':None})))
    write(out/'rankings.csv',stats);write(out/'pairwise_same.csv',pairs)
    steps=[]
    for t in TS:
        for i,skill in enumerate(SKILLS,1):
            rr=[r for r in rows if r['treatment']==t and int(r['step'])==i];x=[num(r.get('new_loc')) for r in rr if r['generation_status']=='generated'];x=[v for v in x if v is not None]
            steps.append(dict(treatment=t,step=i,skill=skill,planned_n=len(rr),pass_n=sum(r['individual_status']=='PASS' for r in rr),fail_n=sum(r['individual_status']=='FAIL' for r in rr),syntax_failure_n=sum(r['individual_status']=='GENERATION_FAILED' for r in rr),blocked_n=sum(r['individual_status']=='BLOCKED' for r in rr),generated_n=len(x),median_added_loc=med(x) if x else None,q1_added_loc=quant(x,.25) if x else None,q3_added_loc=quant(x,.75) if x else None))
    write(out/'skill_summary.csv',steps);svg(out/'new_loc_by_skill.svg',steps)
    overview(out/'treatment_distributions.svg',runs)
    matched=[]
    for cohort,rr in variants:
        lookup={(r['repeat'],r['treatment']):r for r in rr}
        for t,u in [('T1','T2'),('T3','T4'),('T5','T6')]:
            for metric in ('passes','final_passes','complete_loc','all_pass_loc','cost','complete_cost','generation_seconds','complete_generation_seconds','test_wall_complete'):
                vals=[(lookup[i,t][metric],lookup[i,u][metric]) for i in sorted({r['repeat'] for r in rr}) if lookup[i,t][metric] is not None and lookup[i,u][metric] is not None];d=[b-a for a,b in vals];pct=[100*(b-a)/a for a,b in vals if a!=0]
                matched.append(dict(cohort=cohort,metric=metric,no_reuse=t,reuse=u,n_pairs=len(d),median_reuse_minus_no_reuse=med(d) if d else None,median_percent_change=med(pct) if pct else None,reuse_lower=sum(v<0 for v in d),equal=sum(v==0 for v in d),reuse_higher=sum(v>0 for v in d)))
    write(out/'matched_reuse.csv',matched)
    environments='\n\n'.join(f'=== {trial} ===\n{text}' for trial,text in envs);(out/'environments.txt').write_text(environments)
    # Ignore provenance lines (commit/image digest), not Python or package versions.
    consistent=len({tuple(x for x in text.splitlines() if x.startswith('Python ') or '==' in x) for _,text in envs})==1
    report=[f'# CSC491 — V2 statistical analysis (T1–T6)', '', f'Repetitions: {", ".join(f"r{x:02d}" for x in reps)}. {len(rows)} planned skill outcomes; {len(runs)} treatment runs.', f'Python and dependency manifests identical: **{consistent}**. Inspect environments.txt before interpreting timing.', '', '## Outcome accounting', '', '| Treatment | PASS / planned | Syntax failures | Blocked | Test FAIL | Complete generation | All skills passed and retained | Direct assigned library imports / generated |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for t in TS:
        r=[v for v in runs if v['treatment']==t];s=lambda k:sum(v[k] for v in r)
        imp='N/A' if t in ('T1','T2') else f'{s("assigned_imports")}/{s("generated")}'
        report.append(f'| {t}: {NAMES[t]} | {s("passes")}/{len(reps)*10} | {s("syntax_failures")} | {s("blocked")} | {s("competency_failures")} | {s("complete_generation")}/{len(reps)} | {s("all_pass_and_retained")}/{len(reps)} | {imp} |')
    report+=['', '## Course rankings', '', 'Entries are upper median [Q1, Q3]; n; rank. Rank 0 is best **for that metric only**. A dash means fewer than five observations (no rank). Conditional metrics may favor surviving programs; do not treat them as unconditional superiority.', '', '| Metric | '+' | '.join(TS)+' |','|---|'+'---:|'*6]
    for metric,(label,_) in METRICS.items():
        rs=[r for r in stats if r['cohort']=='all_v2' and r['metric']==metric]; report.append('| '+label+' | '+' | '.join(f'{fmt(r["median"])} [{fmt(r["q1"])}, {fmt(r["q3"])}]; n={r["n"]}; #{r["rank"] if r["rank"] is not None else "—"}' for r in rs)+' |')
    report+=['','## Matched reuse comparisons (same repetition)', '', '| Pair | Metric | Matched n | Median reuse − no reuse | Median % change | Reuse lower / equal / higher |','|---|---|---:|---:|---:|---:|']
    for r in matched:
        if r['cohort']=='all_v2' and r['metric'] in ('passes','complete_loc','cost','complete_cost','generation_seconds','complete_generation_seconds'):
            report.append(f'| {r["no_reuse"]}/{r["reuse"]} | {r["metric"]} | {r["n_pairs"]} | {fmt(r["median_reuse_minus_no_reuse"])} | {fmt(r["median_percent_change"])} | {r["reuse_lower"]}/{r["equal"]}/{r["reuse_higher"]} |')
    cost=total(runs,'cost');report+=['',f'Total attempted-call API-equivalent estimated cost: ${cost:.4f}.' if cost is not None else 'Total cost unavailable (missing measurements).','', '## Methods and limitations', '', '- Independent unit: one fresh treatment generation sequence on a fixed Auto93 split. Neither test rows, ten skills, nor overlapping Race episodes are independent repetitions.', '- Course glossary 6: https://txt.github.io/seai26f/gloss6.html. same() requires absolute pooled Cohen ≤ 0.35, absolute Cliff’s delta ≤ 0.195, scaled two-sample KS ≤ 1.36. Rank by upper median; compare each candidate to the current tier leader, starting at rank 0. Q1/Q3 are sorted order statistics at floor(n×0.25/0.75). No rank with n<5 (analysis convention). These are course-style descriptive tiers, not multiplicity-adjusted significance claims or proof of equality.', '- Final LOC uses only complete ten-skill generation sequences; all-pass LOC additionally requires individual and final retention success. Missing or blocked code is not zero LOC. All-outcome costs and time include failed attempts, but blocked steps have no call. Complete-only resource estimates are reported separately.', '- Primary test wall time includes evaluator overhead; it is not pure program execution time. Reported execution time sums skills 2–10 only when all nine values exist (Remember has no reported runtime); some timings come from generated program payloads and are not independently verified. Per-skill execution_seconds is retained in all_steps.csv when available. Different cloud hardware may affect timings.', '- r02 was the v2 validation run; sensitivity rankings and pairwise comparisons excluding it appear in CSVs. r01 used a different contract and is excluded. Model/library availability is not proof of library use; detected imports are a proxy.', '- Fixed dataset, narrowly specified reference tests and one model version limit generalization. No automatic repairs. Do not claim real-world fraud detection or causal repair.', '- T7 is not included: genuine manual prudence checks require separately collected human-authored evidence. EZR tuning is a different experiment; do not pool it with frozen V2. No overall seven-treatment ranking is claimed.', '', 'Files: run_metrics.csv, all_steps.csv, rankings.csv, pairwise_same.csv, matched_reuse.csv, skill_summary.csv, new_loc_by_skill.svg, treatment_distributions.svg, environments.txt.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n'); print(out/'REPORT.md')

if __name__=='__main__':main()
