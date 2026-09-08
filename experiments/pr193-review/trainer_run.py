"""Offline loss-trajectory runner using the unmodified native Eagle3Trainer loop.

Only the transport adapter and target-precompute implementation are selected here.
No standalone optimizer/loss implementation is used.
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torchspec.config.train_config import load_config,config_to_flat_args
from torchspec.models.draft import AutoDraftModelConfig
from torchspec.training.eagle3_trainer import Eagle3Trainer
from torchspec.utils.distributed import init_gloo_group
import torchspec.training.eagle3_trainer as trainer_module
from replay import ROOT,Replay,tensor_hash
from variants import function,source_for

p=argparse.ArgumentParser();p.add_argument('arm',choices=['U','P','S','E']);p.add_argument('name');p.add_argument('--steps',type=int,default=1000);p.add_argument('--eval-interval',type=int,default=100);p.add_argument('--no-save',action='store_true');opts=p.parse_args()
out=ROOT/opts.name;out.mkdir(exist_ok=False)
args=config_to_flat_args(load_config(str(ROOT/'train.yaml')))
args.lr_total_steps=1000;args.output_dir=str(out);args.checkpoint_dir=str(out/'checkpoints');args.rank=0;args.world_size=1;args.load_path=None
args.last_hidden_states_prenorm=False
args.num_train_steps=opts.steps
args.use_pytorch_profiler=False;args.enable_perf_metrics=True
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.set_device(0)
dist.init_process_group('nccl',init_method='file://'+str(out/'dist_init'),rank=0,world_size=1,timeout=timedelta(minutes=10));init_gloo_group()
trainer_module.compute_target_p_padded=function(opts.arm)
trainer=Eagle3Trainer(args)
trainer.init_model(AutoDraftModelConfig.from_file(args.draft_model_config),args.target_model_path)
mapping=torch.load(ROOT/'offline/vocab_mapping.pt',map_location='cpu',weights_only=True)
trainer.draft_model.set_vocab_buffers(mapping['d2t'],mapping['t2d'])
replay=Replay()

def state_hash():
    hashes={name:tensor_hash(p) for name,p in trainer.draft_model.named_parameters()}
    return {'parameters':hashes,'combined':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),'fp32_master':[tensor_hash(p) for p in trainer.optimizer.fp32_params]}

identity={'arm':opts.arm,'steps':opts.steps,'source_sha256':hashlib.sha256(source_for(opts.arm).encode()).hexdigest(),'torch':torch.__version__,'GPU':torch.cuda.get_device_name(),'input_manifest_sha256':hashlib.sha256((ROOT/'data-manifest.json').read_bytes()).hexdigest(),'initial':state_hash(),'scope':'actual Eagle3Trainer.train_from_queue + native BF16Optimizer, deterministic OfflineDataset transport adapter; not online pipeline throughput','args':vars(args)}
(out/'identity.json').write_text(json.dumps(identity,indent=2,default=str))
inputs_file=(out/'inputs.jsonl').open('w')
class Fetcher:
    def __init__(self):self.index=0
    def __iter__(self):return self
    def __next__(self):
        if self.index>=opts.steps:raise StopIteration
        data_id=replay.ids['train'][self.index];batch=replay.batch(data_id)
        row={'step':self.index+1,'id':data_id,'input':tensor_hash(batch['input_ids']),'mask':tensor_hash(batch['loss_mask'])}
        inputs_file.write(json.dumps(row)+'\n');inputs_file.flush();self.index+=1
        return batch
trainer.data_fetcher=Fetcher()
metrics=[]
def record(m):
    if not m:return
    assert math.isfinite(m['train/avg_loss']) and math.isfinite(m['train/grad_norm']),m
    metrics.append(m)
    with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(m)+'\n')
    if m['train/global_step']%10==0:print('STEP',m['train/global_step'],'loss',m['train/avg_loss'],'grad',m['train/grad_norm'],flush=True)

for step in range(opts.steps):
    record(trainer.train_from_queue(step,1))
    if opts.eval_interval and (step+1)%opts.eval_interval==0:
        record(trainer.flush_pending_metrics())
        trainer.model.eval();counts=[];numerators=[]
        for data_id in replay.ids['eval']:
            m=trainer.eval_forward(replay.batch(data_id));counts.append(m['acc_counts']);numerators.append(m['vlosses']*m['acc_counts'])
        count=torch.stack(counts).sum(0);per_depth=torch.stack(numerators).sum(0)/count.clamp_min(1)
        v={'step':step+1,'per_depth_loss':per_depth.cpu().tolist(),'counts':count.cpu().tolist()}
        with (out/'eval.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
        print('EVAL',json.dumps(v),flush=True)
record(trainer.flush_pending_metrics());inputs_file.close()
assert len(metrics)==opts.steps and sorted(x['train/global_step'] for x in metrics)==list(range(1,opts.steps+1))
(out/'final-state.json').write_text(json.dumps(state_hash(),indent=2))
if not opts.no_save:trainer.save_model(opts.steps-1,force_sync=True)
trainer.close();dist.destroy_process_group()
(out/'complete.json').write_text(json.dumps({'steps':opts.steps,'arm':opts.arm,'status':'COMPLETE'}))
print('COMPLETE',opts.name,flush=True)
