"""Review figure: paired loss trajectories, per-step difference, fixed evaluation."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).parent
# Contract: descriptive, one fixed-seed run per arm, all 1000 dependent steps.
# No smoothing, confidence intervals, or significance claims. Python-only exports.
# Panel a = original data; b = signed relative discrepancy; c = fixed-eval trajectory.
contract={'claim':'Quantify U/P loss agreement over 1000 matched optimizer steps','archetype':'quantitative grid','backend':'python','dimensions_mm':[183,163],'outputs':['png','svg','pdf'],'seed':42,'runs_per_arm':1,'train_samples':1000,'eval_samples':128,'smoothing':'none','inferential_statistics':'none; steps are dependent, not independent replicates','source':'trajectory-comparison.json'}
(ROOT/'figure-contract.json').write_text(json.dumps(contract,indent=2))
data=json.loads((ROOT/'trajectory-comparison.json').read_text());steps=data['steps'];evaluation=data['evaluation']
x=np.array([r['step'] for r in steps]);u=np.array([r['U_loss'] for r in steps]);p=np.array([r['P_loss'] for r in steps]);delta=100*(p-u)/np.maximum(np.maximum(np.abs(p),np.abs(u)),1e-8)
assert len(x)==1000 and np.all(np.isfinite(u)) and np.all(np.isfinite(p))
with (ROOT/'loss-trajectories.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['optimizer_step','upstream_loss','pr_loss','signed_relative_difference_percent']);writer.writerows(zip(x,u,p,delta,strict=True))
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.7,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
fig,axes=plt.subplots(3,1,figsize=(7.2047,6.4173),layout='constrained')
colors={'U':'#777777','P':'#2166AC'}
axes[0].plot(x,u,color=colors['U'],linewidth=0.75,label='Upstream (U)')
axes[0].plot(x,p,color=colors['P'],linewidth=0.65,linestyle='--',label='PR (P)')
axes[0].set_ylabel('Training loss');axes[0].legend(loc='upper right',ncol=2)
axes[0].set_title('a  Raw training loss; identical initialization and batch order',loc='left',fontsize=9,fontweight='bold')
axes[1].axhline(0,color='#BBBBBB',linewidth=0.7)
axes[1].plot(x,delta,color=colors['P'],linewidth=0.7)
axes[1].set_ylabel('Relative difference (%)')
axes[1].set_title('b  (P - U) / max(|P|, |U|, 1e-8)',loc='left',fontsize=9,fontweight='bold')
weights=0.8**np.arange(7);weights/=weights.sum()
ex=[r['step'] for r in evaluation]
for arm in ['U','P']:
 values=[float(np.dot(r[f'{arm}_loss'],weights)) for r in evaluation]
 axes[2].plot(ex,values,color=colors[arm],linestyle='-' if arm=='U' else '--',marker='o' if arm=='U' else 'x',markersize=4,linewidth=0.9)
axes[2].set_ylabel('Fixed-evaluation loss')
axes[2].set_title('c  128 held-out conversations; 0.8^depth weighted loss',loc='left',fontsize=9,fontweight='bold')
for ax in axes:
 ax.set_xlim(0,1000);ax.set_xlabel('Optimizer step');ax.grid(axis='y',color='#EEEEEE',linewidth=0.6)
fig.savefig(ROOT/'loss-trajectories.svg')
fig.savefig(ROOT/'loss-trajectories.pdf')
fig.savefig(ROOT/'loss-trajectories.png',dpi=600)
plt.close(fig)
