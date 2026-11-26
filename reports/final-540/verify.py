#!/usr/bin/env python3
"""Read-only final evidence and metric verification; no generated-code execution."""
import csv, json, zipfile, hashlib, collections, subprocess, tempfile, sys
from pathlib import Path
D=Path(__file__).resolve().parent
reps={2,3,4,5,7,8,9,10,12}; skills='remember guess certify flow blame justify choose fix spot race'.split()
with zipfile.ZipFile(D/'generation_records.zip') as z:
    h=json.loads((D/'generation_sha256.json').read_text())
    assert set(z.namelist())==set(h) and len(h)==988
    assert z.testzip() is None
    for name,digest in h.items(): assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    assert sum(n.endswith('/metrics.json') for n in h)==540
    assert sum(n.endswith('/generated.py') for n in h)==448
print('Generation records: 540 metadata + 448 source files; hashes match.')
with (D/'all_540_outcomes_RECONSTRUCTED.csv').open(newline='') as f: rows=list(csv.DictReader(f))
assert len(rows)==540
assert {(int(r['repeat']),r['treatment'],int(r['step'])) for r in rows}=={(n,f'T{t}',s) for n in reps for t in range(1,7) for s in range(1,11)}
assert all(r['skill']==skills[int(r['step'])-1] for r in rows)
assert collections.Counter(r['individual_status'] for r in rows)=={'PASS':440,'GENERATION_FAILED':32,'BLOCKED':60,'FAIL':8}
with zipfile.ZipFile(D/'evaluation_records.zip') as z:
    assert z.testzip() is None
    n=[n for n in z.namelist() if n.endswith('/all_540_outcomes_RECONSTRUCTED.csv')];assert len(n)==1
    assert z.read(n[0])==(D/'all_540_outcomes_RECONSTRUCTED.csv').read_bytes()
print('Outcomes: 540 unique records; counts match.')
keys={'external_per_implementation.csv':('repeat','condition','skill','metric'),'external_summary.csv':('skill','metric','condition'),'fixed_case_profiles.csv':('condition','skill','metric','case_id'),'same_repetition_level_T1_T7.csv':('skill','metric','a','b'),'same_fixed_case_T1_T7.csv':('skill','metric','a','b'),'same_each_snapshot_vs_T7.csv':('skill','metric','condition','repeat')}
def load(path,key):
    with path.open(newline='') as f: rows=[r for r in csv.DictReader(f) if r.get('condition')!='E1']
    result={tuple(r[x] for x in key):r for r in rows};assert len(result)==len(rows),path;return result
with tempfile.TemporaryDirectory(prefix='csc491-verify-') as tmp:
    subprocess.run([sys.executable,str(D/'recompute_external.py'),str(D/'evaluation_records.zip'),tmp],check=True)
    total=0
    for name,key in keys.items():
        a=load(D/'original_deliverables/results'/name,key); b=load(Path(tmp)/name,key); c=load(D/'reproduced_metrics'/name,key)
        assert a==b==c,name
        total+=len(a); print(name,len(a),'exact matches')
    assert total==9939
print('PASS: 9,939 T1–T7 analysis records match exactly. E1 is excluded.')
