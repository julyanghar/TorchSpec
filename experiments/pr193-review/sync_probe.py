import os
import json
import torch
from replay import ROOT

mask=(torch.arange(8192,device='cuda')%7==0)
value=mask.sum()
for _ in range(5):mask.nonzero(as_tuple=True)
torch.cuda.synchronize()
observations=[]
torch.cuda.cudart().cudaProfilerStart()
for i in range(20):
    with torch.cuda.nvtx.range('existing_nonzero'):
        indices=mask.nonzero(as_tuple=True)[0]
    done=torch.cuda.Event()
    torch.cuda._sleep(30000000)
    done.record()
    with torch.cuda.nvtx.range('numel_branch'):
        all_valid=indices.numel()==mask.numel()
    observations.append({'iteration':i,'pending_gpu_work_after_numel':not done.query(),'all_valid':all_valid})
    with torch.cuda.nvtx.range('value_item_control'):
        count=value.item()
    assert count==indices.numel()
torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
(ROOT/'sync-probe.json').write_text(json.dumps(observations,indent=2))
print('pending GPU work after numel:',sum(x['pending_gpu_work_after_numel'] for x in observations),'/',len(observations))
