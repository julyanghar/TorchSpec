import os
import argparse
import json
import time
import numpy as np
import torch
from replay import ROOT,Replay
from variants import function
from torchspec.models.target.target_utils import TargetLMHead
from torchspec.utils.tensor import padding

p=argparse.ArgumentParser();p.add_argument('repeat',type=int);p.add_argument('--real',action='store_true');o=p.parse_args()
mode='real' if o.real else 'synthetic';results=[];functions={a:function(a) for a in ['U','P','S']}
torch.manual_seed(193)

def cases():
 if not o.real:
  head=torch.randn(65536,1024,device='cuda',dtype=torch.bfloat16)*0.02
  t2d=torch.arange(65536,device='cuda')%2==0
  for n in [2048,8192]:
   hs=torch.randn(1,n,1024,device='cuda',dtype=torch.bfloat16)
   for count in [0,204,1024,1843,1950,2028,2048]:
    count=count*(n//2048)
    for layout in ['suffix','dispersed']:
     mask=torch.zeros(1,n,device='cuda')
     if count:
      if layout=='suffix':mask[:,-count:]=1
      else:mask[:,torch.randperm(n,device='cuda')[:count]]=1
     yield {'N':n,'L':count,'H':1024,'V':65536,'draft_V':32768,'layout':layout},(hs,head,t2d,mask,7)
 else:
  replay=Replay();meta=json.loads((ROOT/'data-manifest.json').read_text());head=TargetLMHead.from_pretrained(meta['tokenizer']).lm_head.weight
  mapping=torch.load(ROOT/'offline/vocab_mapping.pt',weights_only=True);t2d=mapping['t2d'].cuda()
  stats=[json.loads(x) for x in (ROOT/'real-batch-stats.jsonl').read_text().splitlines() if json.loads(x)['split']=='train']
  ordered=sorted(stats,key=lambda x:(x['rho_loss'],x['id']))
  # Fixed quantiles across the entire training set, independent of timing results.
  chosen=[ordered[round(q*(len(ordered)-1))] for q in np.linspace(0,1,12)]
  for item in chosen:
   batch=replay.batch(item['id']);hs=padding(batch['last_hidden_states'],left=False).cuda();mask=batch['loss_mask'].cuda()
   yield {'N':item['N'],'L':item['L'],'H':head.shape[1],'V':head.shape[0],'draft_V':int(t2d.sum()),'id':item['id']},(hs,head,t2d,mask,7)

with torch.no_grad():
 for index,(info,args) in enumerate(cases()):
  gold=functions['U'](*args);valid=args[3].bool();vref=gold.target_p_padded[:,:info['N']][valid]
  error={}
  for arm in ['P','S']:
   out=functions[arm](*args);val=out.target_p_padded[:,:info['N']][valid]
   torch.testing.assert_close(out.position_mask,gold.position_mask,atol=0,rtol=0)
   torch.testing.assert_close(out.coverage_padded,gold.coverage_padded,atol=0,rtol=0)
   elementwise_pass=bool(torch.all((val-vref).abs() <= 2e-5+0.02*vref.abs()))
   error[arm]={'existing_elementwise_check_pass':elementwise_pass,'max_abs':float((val-vref).abs().max()) if val.numel() else 0,'rel_L2':float(torch.linalg.vector_norm(val-vref)/torch.linalg.vector_norm(vref).clamp_min(1e-20))}
   del out,val
  del gold,vref
  arms=['U','P','S'];offset=(index+o.repeat)%3;arms=arms[offset:]+arms[:offset]
  observations=[]
  for arm in arms:
   fn=functions[arm]
   for _ in range(10):warm=fn(*args);del warm
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
   start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
   start=time.perf_counter();start_event.record()
   for _ in range(20):out=fn(*args);del out
   end_event.record();torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000/20
   observations.append({'arm':arm,'wall_ms':wall,'event_ms':start_event.elapsed_time(end_event)/20,'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()})
  row={'repeat':o.repeat,'mode':mode,'case':info,'order':arms,'errors':error,'observations':observations};results.append(row)
  (ROOT/(f'performance-reviewed-{mode}-{o.repeat}.json')).write_text(json.dumps(results,indent=2))
  print(mode,o.repeat,index,info,{x['arm']:round(x['wall_ms'],3) for x in observations},flush=True)
print('COMPLETE',mode,o.repeat,len(results),flush=True)
