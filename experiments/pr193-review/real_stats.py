import os
import hashlib
import json
import numpy as np
import torch
import torch.nn.functional as F
from replay import ROOT,Replay,tensor_hash
from torchspec.models.target.target_utils import TargetLMHead
from torchspec.utils.tensor import padding

model=os.environ['QWEN3_8B_PATH']
r=Replay();assert len(r.dataset.rows('train'))==1000 and len(r.dataset.rows('eval'))==128
mapping=torch.load(ROOT/'offline/vocab_mapping.pt',weights_only=True);t2d=mapping['t2d'].cuda()
head=TargetLMHead.from_pretrained(model,load_norm=False,device='cuda');rows=[]
with torch.no_grad(),(ROOT/'real-batch-stats.jsonl').open('w') as out:
 for split in ['train','eval']:
  for idx,data_id in enumerate(r.ids[split]):
   batch=r.batch(data_id);mask=batch['loss_mask'].cuda().bool();hs=padding(batch['last_hidden_states'],left=False).cuda()
   selected=hs[mask];V=0
   for start in range(0,len(selected),4096):V+=int(t2d[F.linear(selected[start:start+4096],head.lm_head.weight).argmax(-1)].sum())
   N=mask.numel();L=int(mask.sum());nonpad=int(batch['attention_mask'].sum());row={'id':data_id,'split':split,'N':N,'L':L,'V':V,'nonpadding_rows':nonpad,'rho_loss':L/N,'rho_coverage':V/L if L else None,'rho_final':V/N,'input_hash':tensor_hash(batch['input_ids']),'mask_hash':tensor_hash(batch['loss_mask'])}
   rows.append(row);out.write(json.dumps(row)+'\n');out.flush()
   if idx%100==0:print(split,idx,flush=True)
 summary={}
 for split in ['train','eval']:
  rr=[x for x in rows if x['split']==split];N=sum(x['N'] for x in rr);L=sum(x['L'] for x in rr);V=sum(x['V'] for x in rr)
  summary[split]={'samples':len(rr),'N':N,'L':L,'V':V,'nonpadding_rows':sum(x['nonpadding_rows'] for x in rr),'weighted_loss':L/N,'weighted_coverage':V/L,'weighted_final':V/N,'rho_loss_quantiles':np.quantile([x['rho_loss'] for x in rr],[0,.1,.5,.9,1]).tolist()}
 summary['vocab_mapping_sha256']=hashlib.sha256((ROOT/'offline/vocab_mapping.pt').read_bytes()).hexdigest()
 summary['target_head_hash']=tensor_hash(head.lm_head.weight)
 (ROOT/'real-stats-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
