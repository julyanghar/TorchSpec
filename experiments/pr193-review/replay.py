import os
import hashlib
import json
from pathlib import Path
import torch
from torchspec.data.utils import DataCollatorWithPadding, resolve_loss_mask
from torchspec.offline.dataset import OfflineDataset

ROOT=Path(__file__).parent

def tensor_hash(tensor):
    a=tensor.detach().cpu().contiguous()
    return hashlib.sha256(a.view(torch.uint8).numpy().tobytes()).hexdigest()

class Replay:
    def __init__(self):
        self.dataset=OfflineDataset(ROOT/'offline')
        meta=json.loads((ROOT/'data-manifest.json').read_text())
        self.ids={'train':meta['train_ids'],'eval':['eval_'+x for x in meta['eval_ids']]}
        self.collate=DataCollatorWithPadding()
    def item(self,data_id):
        d=self.dataset.load(data_id)
        assert resolve_loss_mask(d) is not None
        for k,t in list(d.items()):
            if isinstance(t,torch.Tensor):
                if t.ndim==1:d[k]=t.unsqueeze(0)
                elif t.ndim==2 and k in ['hidden_states','last_hidden_states','target']:d[k]=t.unsqueeze(0)
        return d
    def batch(self,data_id):return self.collate([self.item(data_id)])
