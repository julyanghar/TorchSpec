import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from replay import ROOT,Replay
from variants import function
from torchspec.models.target.target_utils import TargetLMHead
from torchspec.utils.tensor import padding

r=Replay();meta=json.loads((ROOT/'data-manifest.json').read_text());head=TargetLMHead.from_pretrained(meta['tokenizer']).lm_head.weight
mapping=torch.load(ROOT/'offline/vocab_mapping.pt',weights_only=True);t2d=mapping['t2d'].cuda();w=head[t2d].float()
stats=[json.loads(x) for x in (ROOT/'real-batch-stats.jsonl').read_text().splitlines() if json.loads(x)['split']=='train'];order=sorted(stats,key=lambda x:(x['rho_loss'],x['id']));chosen=[order[round(q*(len(order)-1))] for q in np.linspace(0,1,12)]
torch.backends.cuda.matmul.allow_tf32=False;results=[]
def error(x,y):
 d=(x-y).abs();return {'max_abs':float(d.max()) if d.numel() else 0,'mean_abs':float(d.mean()) if d.numel() else 0,'relative_L2':float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(y).clamp_min(1e-20)),'elements':d.numel(),'exceeds_existing_elementwise_tolerance':int((d>2e-5+0.02*y.abs()).sum())}
with torch.no_grad():
 for index,info in enumerate(chosen):
  b=r.batch(info['id']);hs=padding(b['last_hidden_states'],left=False).cuda();mask=b['loss_mask'].cuda().bool();args=(hs,head,t2d,mask,7)
  outs={};coverage={};positions={}
  for arm in ['U','P','S']:
   o=function(arm)(*args);outs[arm]=o.target_p_padded[:,:info['N']][mask];coverage[arm]=o.coverage_padded;positions[arm]=o.position_mask
  ref=F.softmax(F.linear(hs[mask].float(),w),dim=-1)
  torch.testing.assert_close(outs['P'],outs['S'],atol=0,rtol=0)
  torch.testing.assert_close(coverage['U'],coverage['S'],atol=0,rtol=0)
  torch.testing.assert_close(positions['U'],positions['S'],atol=0,rtol=0)
  row={'id':info['id'],'N':info['N'],'L':info['L'],'P_vs_U':error(outs['P'],outs['U']),'S_vs_P':error(outs['S'],outs['P']),'U_vs_FP32':error(outs['U'],ref),'S_vs_FP32':error(outs['S'],ref),'target_argmax_changes':int((outs['S'].argmax(-1)!=outs['U'].argmax(-1)).sum())}
  results.append(row);(ROOT/'real-numerics.json').write_text(json.dumps(results,indent=2));print(row,flush=True)
