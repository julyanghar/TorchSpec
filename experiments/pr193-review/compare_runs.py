import os
import json
from pathlib import Path
import numpy as np
from replay import ROOT

r=ROOT
for arm in ['U','P']:
 assert json.loads((r/f'train-{arm}-1000/complete.json').read_text())['steps']==1000
u=r/'train-U-1000';p=r/'train-P-1000'
ui=json.loads((u/'identity.json').read_text());pi=json.loads((p/'identity.json').read_text())
assert ui['initial']==pi['initial']
assert (u/'inputs.jsonl').read_bytes()==(p/'inputs.jsonl').read_bytes()
us=[json.loads(x) for x in (u/'metrics.jsonl').read_text().splitlines()];ps=[json.loads(x) for x in (p/'metrics.jsonl').read_text().splitlines()]
assert len(us)==len(ps)==1000
rows=[]
for a,b in zip(us,ps,strict=True):
 assert a['train/global_step']==b['train/global_step']
 av,bv=a['train/avg_loss'],b['train/avg_loss']
 rows.append({'step':a['train/global_step'],'U_loss':av,'P_loss':bv,'abs_diff':abs(av-bv),'relative_diff':abs(av-bv)/max(abs(av),abs(bv),1e-8),'U_grad_norm':a['train/grad_norm'],'P_grad_norm':b['train/grad_norm'],'per_depth_abs_diff':[abs(a[f'train/ploss_{i}']-b[f'train/ploss_{i}']) for i in range(7)]})
ue=[json.loads(x) for x in (u/'eval.jsonl').read_text().splitlines()];pe=[json.loads(x) for x in (p/'eval.jsonl').read_text().splitlines()];assert len(ue)==len(pe)==10
evals=[]
for a,b in zip(ue,pe,strict=True):
 assert a['step']==b['step'] and a['counts']==b['counts']
 evals.append({'step':a['step'],'U_loss':a['per_depth_loss'],'P_loss':b['per_depth_loss'],'relative_diff':[abs(x-y)/max(abs(x),abs(y),1e-8) for x,y in zip(a['per_depth_loss'],b['per_depth_loss'],strict=True)]})
max_rel=max(rows,key=lambda x:x['relative_diff'])
summary={'steps':1000,'initial_parameters_and_master_identical':True,'input_ids_and_masks_identical':True,'train_loss_bitwise_equal':all(x['abs_diff']==0 for x in rows),'mean_absolute_loss_difference':float(np.mean([x['abs_diff'] for x in rows])),'max_absolute_loss_difference':max(x['abs_diff'] for x in rows),'mean_relative_loss_difference':float(np.mean([x['relative_diff'] for x in rows])),'max_relative_loss_difference':max_rel['relative_diff'],'worst_relative_step':max_rel,'max_eval_relative_difference':max(max(e['relative_diff']) for e in evals),'upstream_CI_1percent_diagnostic_pass':max_rel['relative_diff']<=0.01,'final_state_hash_equal':json.loads((u/'final-state.json').read_text())==json.loads((p/'final-state.json').read_text())}
assert all(np.isfinite(x[k]) for x in rows for k in ['U_loss','P_loss','U_grad_norm','P_grad_norm'])
(r/'trajectory-comparison.json').write_text(json.dumps({'summary':summary,'steps':rows,'evaluation':evals},indent=2));print(json.dumps(summary,indent=2))
