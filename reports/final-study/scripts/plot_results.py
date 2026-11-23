#!/usr/bin/env python3
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
data=root/'data';out=root/'figures';out.mkdir(exist_ok=True)
with (data/'paired_reuse.csv').open() as f: pairs=list(csv.DictReader(f))
with (data/'study_summary.csv').open() as f: summary=list(csv.DictReader(f))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

fig,ax=plt.subplots(figsize=(3.27,2.05))
entries=[p for p in pairs if p['pair'] in ('T1-T2','T3-T4') and p['reuse_LOC_change_pct']]
y=range(len(entries));
for i,p in enumerate(entries):
    v0=float(p['no_reuse_complete_LOC']); v1=float(p['reuse_complete_LOC']);
    ax.plot([v1,v0],[i,i],color='#bac4d0',lw=2.4,zorder=1)
    ax.scatter([v0],[i],s=32,marker='s',color='#2d5779',zorder=2)
    ax.scatter([v1],[i],s=32,marker='o',color='#22967b',zorder=2)
ax.set_yticks(list(y),[f"{p['pair']} {p['repetition']}" for p in entries]);ax.invert_yaxis()
ax.set_xlim(350,1100);ax.set_xlabel('Complete implementation, physical LOC')
ax.tick_params(axis='both',length=0);ax.grid(axis='x',alpha=.2)
fig.text(.53,.99,'Squares: no reuse     Circles: reuse',ha='center',va='top',fontsize=7)
fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(out/'paired_LOC.pdf',bbox_inches='tight');fig.savefig(out/'paired_LOC.png',dpi=170,bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(3.27,1.95))
x=[r['treatment'] for r in summary];v=[float(r['cost_usd_upper_median']) for r in summary]
bars=ax.barh(x,v,height=.62,color=['#2d5779','#22967b']*3)
for b,value in zip(bars,v):
    ax.text(b.get_width()+0.025,b.get_y()+b.get_height()/2,f'${value:.2f}',va='center',fontsize=7)
ax.invert_yaxis();ax.set_xlim(0,1.65);ax.set_xlabel('Estimated USD / repetition (upper median)')
ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
ax.tick_params(length=0);fig.tight_layout();fig.savefig(out/'cost_median.pdf',bbox_inches='tight');fig.savefig(out/'cost_median.png',dpi=170,bbox_inches='tight');plt.close(fig)
print('Saved figure PDFs and PNGs to',out)
