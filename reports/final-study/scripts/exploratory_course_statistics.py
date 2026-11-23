#!/usr/bin/env python3
"""Exploratory 3-test comparisons, NO final ranks, n<=4 per treatment.

Thresholds from https://txt.github.io/seai26f/gloss6.html are
Cohen 0.35; abs Cliff 0.195; scaled KS 1.36.
This uses four recorded Claude generation repetitions only; it is
not a seven-treatment ranking or a claim of statistical significance.
"""
from pathlib import Path
import csv,math,statistics
root=Path(__file__).resolve().parents[1]
source=root/'data'/'run_metrics.csv'
rows=list(csv.DictReader(source.open()))

def xs(t,metric):
    vals=[]
    for r in rows:
        if r['treatment']!=t:continue
        v=r.get(metric)
        if v not in ('',None):vals.append(float(v))
    return vals

def cohen(x,y):
    m=len(x);n=len(y)
    vx=statistics.variance(x) if m>1 else 0
    vy=statistics.variance(y) if n>1 else 0
    sd=math.sqrt(((m-1)*vx+(n-1)*vy)/(m+n-2))
    return abs(statistics.mean(x)-statistics.mean(y))/(sd+1e-32)

def cliffs(x,y):
    gt=sum(a>b for a in x for b in y)
    lt=sum(a<b for a in x for b in y)
    return abs(gt-lt)/(len(x)*len(y))

def ks_scaled(x,y):
    x=sorted(x);y=sorted(y);nx=len(x);ny=len(y)
    p=q=0;d=0.
    # The course demonstration stops when either sample is exhausted.
    while p<nx and q<ny:
        v=min(x[p],y[q])
        while p<nx and x[p]==v:p+=1
        while q<ny and y[q]==v:q+=1
        d=max(d,abs(p/nx-q/ny))
    return d/math.sqrt((nx+ny)/(nx*ny))

result=[]
for a,b in [('T1','T2'),('T3','T4'),('T5','T6')]:
    for metric,label in [('cost','estimated_generation_cost_USD'),('generation_seconds','generation_wall_seconds'),('complete_loc','conditional_complete_LOC')]:
        x=xs(a,metric);y=xs(b,metric)
        if len(x)<2 or len(y)<2:
            c=cl=k=similar=None
        else:
            c=cohen(x,y);cl=cliffs(x,y);k=ks_scaled(x,y)
            similar=c<=.35 and cl<=.195 and k<=1.36
        result.append({'pair':f'{a}/{b}','metric':label,'n_without_reuse':len(x),'n_reuse':len(y),
            'cohen':round(c,4) if c is not None else '',
            'cliffs_abs':round(cl,4) if cl is not None else '',
            'ks_scaled':round(k,4) if k is not None else '',
            'same_under_lecture_cutoffs':str(similar) if similar is not None else 'NA',
            'interpretation':'exploratory only; incomplete samples; no statistical rank'})
p=root/'data'/'exploratory_same_diagnostics.csv'
with p.open('w',newline='') as f:
    w=csv.DictWriter(f,result[0].keys());w.writeheader();w.writerows(result)
print('Saved',p)
for r in result:print(r)
