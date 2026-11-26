#!/usr/bin/env python3
"""Read archived checker outputs only. No generated code execution.
Usage: python recompute_external.py REPLAY_ZIP OUTPUT_DIR
"""
import csv, json, math, statistics as st, itertools, sys, zipfile, re, hashlib
from pathlib import Path
from collections import defaultdict, Counter
A,O=map(Path,sys.argv[1:3]); O.mkdir(parents=True,exist_ok=True)
S='remember guess certify flow blame justify choose fix spot race'.split(); T=[f'T{i}' for i in range(1,8)]; C=T
z=zipfile.ZipFile(A); names=z.namelist()
def unique(suffix):
 ns=[n for n in names if n.endswith(suffix)]; assert len(ns)==1,(suffix,ns); return ns[0]
raw=z.read(unique('/all_540_outcomes_RECONSTRUCTED.csv'))
R=list(csv.DictReader(raw.decode().splitlines())); assert len(R)==540 and sum(r['individual_status']=='PASS' for r in R)==440
lookup={}
for n in names:
 m=re.search(r'/v2-results/r(\d+)/t(\d+)/step(\d+)/(\w+)\.(json|txt)$',n)
 if m:
  key=(int(m[1]),'T'+m[2],int(m[3]),m[4],m[5]);assert key not in lookup,key;lookup[key]=n
def output(r,t,s):
 ext='txt' if s=='remember' else 'json';i=S.index(s)+1
 if t=='T7':
  ns=[n for n in names if '/original_t7/' in n and n.endswith('/'+s+'.'+ext)]; assert len(ns)==1,(t,s,ns);n=ns[0]
 else:n=lookup[(r,t,i,s,ext)]
 b=z.read(n)
 if ext=='txt':assert re.search(rb'(?m)^PASS(?:\s|$)',b);return None
 return json.loads(b)
def write(n,rows):
 assert rows
 with (O/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def upper(v):v=sorted(v);return v[len(v)//2]
def mean(v):return st.mean(v) if v else None
def rate(n,d):return n/d if d else None
obs=[]; cases=defaultdict(lambda:defaultdict(list)); counts=Counter(); refs={}; repeats=defaultdict(list)
def add(r,t,s,m,v):
 if v is not None:obs.append(dict(repeat=r,condition=t,skill=s,metric=m,value=v))
def case(t,s,m,key,v,truth=None):
 if truth is not None:
  a=(s,m,key)
  if a in refs:assert refs[a]==truth,(a,refs[a],truth)
  else:refs[a]=truth
 cases[(t,s,m)][str(key)].append(float(v))
def analyze(r,t,s,a):
 counts[(t,s)]+=1; repeats[t,s].append(r)
 if s=='remember':
  for m in ['precision_at_10_proxy','recall_at_10_proxy']:add(r,t,s,m,1.)
  return
 z=a.get('summary',{}); e=a.get('errors',[])
 def metric(m,v):add(r,t,s,m,v)
 if s=='guess':
  assert len(e)==80
  for m,k in [('mpg_mae','mpg_error'),('disty_mae','disty_error')]:metric(m,mean([v[k] for v in e]));[case(t,s,m,v['row'],v[k]) for v in e]
  metric('hit_within_2_mpg',mean([v['mpg_error']<=2 for v in e]));[case(t,s,'hit_within_2_mpg',v['row'],v['mpg_error']<=2) for v in e]
 elif s=='certify':
  assert len(e)==80
  same=[v for v in e if v['expected']];diff=[v for v in e if not v['expected']];assert (len(same),len(diff))==(56,24)
  metric('accuracy',mean([v['correct'] for v in e]));metric('false_alarm_reference_same',mean([not v['actual'] for v in same]));metric('miss_reference_different',mean([v['actual'] for v in diff]))
  for v in e:case(t,s,'decision_error',v['case'],not v['correct'],v['expected'])
 elif s in ('flow','spot'):
  key='observed_alert' if s=='flow' else 'injected_anomaly';e=[v for v in e if v['window_ready']] if s=='flow' else e
  assert len(e)==(61 if s=='flow' else 80)
  tp=sum(v['alert'] and v[key] for v in e);fp=sum(v['alert'] and not v[key] for v in e);fn=sum(not v['alert'] and v[key] for v in e);tn=len(e)-tp-fp-fn
  for m,v in [('tp',tp),('fp',fp),('fn',fn),('tn',tn),('precision',rate(tp,tp+fp)),('recall',rate(tp,tp+fn)),('f1',rate(2*tp,2*tp+fp+fn)),('agreement',rate(tp+tn,len(e)))]:metric(m,v)
  for v in e:
   case(t,s,'decision_error',v['row'],v['alert']!=v[key],v[key]);case(t,s,'miss_on_positive' if v[key] else 'false_alert_on_negative',v['row'],not v['alert'] if v[key] else v['alert'],v[key])
 elif s=='blame':
  metric('contract_case_accuracy',z['correct']/z['cases']);metric('goal_error',z['goal_error']);metric('feature_error',z['feature_error'])
  # Passing checker verifies names, order and numeric evidence; no human causal validation.
  for v in e:case(t,s,'goal_contract_error_gt_1e_9',v['row'],v['goals_error']>1e-9)
 elif s=='justify':
  metric('contract_case_accuracy',z['correct']/z['cases']);metric('score_error',z['score_error'])
  for v in e:case(t,s,'decision_contract_error',v['row'],not v['correct'])
 elif s=='choose':
  for m in ('regret','top_mean','best','inspection_savings','inspections'):metric(m,z[m])
 elif s=='fix':
  metric('improvement_rate',rate(z['improved'],z['eligible']));metric('mean_gain',z['mean_gain']);metric('mean_edits',z['mean_edits']);metric('cutoff_reached_rate',rate(z['reached_cutoff'],z['eligible']))
  for v in e:case(t,s,'gain_all_rows',v['row'],round(v['gain'],10))
 elif s=='race':
  assert len(a['results'])==20
  for m in ('best5','best10','labels_to_target'):
   metric(m,mean([v['adaptive'][m] for v in a['results']]));metric('random_'+m,mean([v['random'][m] for v in a['results']]))
   for v in a['results']:case(t,s,m,v['episode'],round(v['adaptive'][m],12),round(v['random'][m],12))
  metric('target_reached_rate',a['target_reached']['adaptive']/20)
for v in R:
 if v['individual_status']=='PASS':analyze(int(v['repeat']),v['treatment'],v['skill'],output(int(v['repeat']),v['treatment'],v['skill']))
for t in C[6:]:
 for s in S:analyze('single',t,s,output(None,t,s))
assert sum(counts.values())==450
write('external_per_implementation.csv',obs)
g=defaultdict(list)
for v in obs:g[(v['skill'],v['metric'],v['condition'])].append(v['value'])
summ=[]
for (s,m,t),v in sorted(g.items(),key=lambda kv:(S.index(kv[0][0]),kv[0][1],C.index(kv[0][2]))):summ.append(dict(skill=s,metric=m,condition=t,planned=9 if t in T[:6] else 1,passing=counts[t,s],metric_n=len(v),upper_median=upper(v),minimum=min(v),maximum=max(v),aggregation='passing snapshots only' if t in T[:6] else 'single implementation'))
write('external_summary.csv',summ)
def same(x,y):
 n,m=len(x),len(y)
 if min(n,m)<3:return dict(cohen=None,cliffs=None,ks=None,same='INSUFFICIENT_N')
 c=abs(st.mean(x)-st.mean(y))/(math.sqrt(((n-1)*st.variance(x)+(m-1)*st.variance(y))/(n+m-2))+1e-32)
 d=abs(sum((i>j)-(i<j) for i in x for j in y))/(n*m)
 k=max(abs(sum(i<=v for i in x)/n-sum(j<=v for j in y)/m) for v in set(x+y))/math.sqrt(1/n+1/m)
 return dict(cohen=c,cliffs=d,ks=k,same=c<=.35 and d<=.195 and k<=1.36)
# Generation-level diagnostics; T7 deliberately appears as insufficient N, rather than pseudo-replication.
run=[]
primary={'remember':'precision_at_10_proxy','guess':'mpg_mae','certify':'accuracy','flow':'recall','blame':'contract_case_accuracy','justify':'contract_case_accuracy','choose':'regret','fix':'mean_gain','spot':'f1','race':'best10'}
for s,m in primary.items():
 for a,b in itertools.combinations(T,2):
  x,y=g[(s,m,a)],g[(s,m,b)];run.append(dict(skill=s,metric=m,a=a,b=b,n_a=len(x),n_b=len(y),upper_median_a=upper(x),upper_median_b=upper(y),**same(x,y)))
write('same_repetition_level_T1_T7.csv',run)
# Fixed-case profiles: one average loss/outcome per case per arm, averaging passing implementations.
# The same held-out items recur across arms; distribution comparisons descriptive, not independent inference.
profile=[];diag=[]
for (s,m) in sorted({(s,m) for t,s,m in cases},key=lambda v:(S.index(v[0]),v[1])):
 for t in C:
  for key,v in sorted(cases[t,s,m].items()):
   assert len(v)==counts[t,s],(t,s,m,key,len(v),counts[t,s]);profile.append(dict(condition=t,skill=s,metric=m,case_id=key,passing_implementations=len(v),case_mean=mean(v)))
 for a,b in itertools.combinations(T,2):
  aa,bb=cases[a,s,m],cases[b,s,m];assert aa.keys()==bb.keys(),(s,m,a,b)
  keys=sorted(aa);x=[mean(aa[k]) for k in keys];y=[mean(bb[k]) for k in keys]
  diag.append(dict(skill=s,metric=m,a=a,b=b,cases=len(keys),implementations_a=counts[a,s],implementations_b=counts[b,s],mean_a=mean(x),mean_b=mean(y),paired_mean_difference_b_minus_a=mean([v-u for u,v in zip(x,y)]),**same(x,y),scope='descriptive fixed-case profiles; no independent development inference'))
write('fixed_case_profiles.csv',profile);write('same_fixed_case_T1_T7.csv',diag)
# Each passing snapshot versus the single T7 on exactly the same cases. No pooling of repeated cases.
individual=[]
for s,m in sorted({(s,m) for t,s,m in cases},key=lambda v:(S.index(v[0]),v[1])):
 for t in T[:6]:
  aa,bb=cases[t,s,m],cases['T7',s,m];keys=sorted(aa);y=[bb[k][0] for k in keys]
  for j,r in enumerate(repeats[t,s]):
   x=[aa[k][j] for k in keys];individual.append(dict(skill=s,metric=m,condition=t,repeat=r,reference='T7',cases=len(keys),mean_treatment=mean(x),mean_T7=mean(y),paired_mean_difference=mean([u-v for u,v in zip(x,y)]),**same(x,y),scope='one snapshot vs single T7 on matched fixed cases'))
write('same_each_snapshot_vs_T7.csv',individual)
(O/'provenance.json').write_text(json.dumps({'replay_archive_sha256':hashlib.sha256(A.read_bytes()).hexdigest(),'outcomes_sha256':hashlib.sha256(raw).hexdigest(),'scope':'original R02-R05 and T7, replay R07-R10/R12; E1 excluded; saved passing outputs only','same_thresholds':[.35,.195,1.36]},indent=2))
print(len(obs),len(summ),len(run),len(profile),len(diag)); print('T7 fixed-case diagnostics:',Counter(str(r['same']) for r in diag if r['b']=='T7')); print('All fixed-case diagnostics:',Counter(str(r['same']) for r in diag))
