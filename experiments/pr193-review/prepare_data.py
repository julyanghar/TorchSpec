import os
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from torchspec.data.dataset import _init_tokenize_worker, _tokenize_single
from torchspec.data.utils import unpack_loss_mask

root=Path(__file__).parent
source=Path(os.environ['SHAREGPT_DATA'])
tokenizer=os.environ['QWEN3_8B_PATH']
offsets=[];digest=hashlib.sha256();pos=0
with source.open('rb') as stream:
 for line in stream:
  offsets.append(pos);pos+=len(line);digest.update(line)
order=list(range(len(offsets)));random.Random(42).shuffle(order)
_init_tokenize_worker(tokenizer,False,'qwen','auto')
records=[];skipped=[];stats=[]
with source.open('rb') as stream:
 for idx in order:
  stream.seek(offsets[idx]);raw=json.loads(stream.readline())
  item=_tokenize_single((raw['conversations'],None,None,4096,False))
  if item is None:
   skipped.append({'source_row':idx,'reason':'native preprocessing returned None'});continue
  ids=item['input_ids'].reshape(-1).tolist();mask=unpack_loss_mask(item['packed_loss_mask']).tolist()
  assert len(ids)==len(mask)
  if sum(mask)==0 or len(ids)<8:
   skipped.append({'source_row':idx,'reason':'no supervised row or fewer than 8 tokens'});continue
  sample={'data_id':f'sharegpt-row-{idx}','source_id':raw.get('id'),'source_row':idx,'input_ids':ids,'packed_loss_mask':item['packed_loss_mask'],'has_thinking':item.get('has_thinking',False)}
  records.append(sample)
  stats.append({'data_id':sample['data_id'],'N':len(ids),'L_raw':sum(mask),'rho_loss_raw':sum(mask)/len(ids),'last_turn_only':sample['has_thinking']})
  if len(records)%100==0:print('tokenized',len(records),flush=True)
  if len(records)==1128:break
assert len(records)==1128
for name,items in [('train',records[:1000]),('eval',records[1000:])]:
 with (root/(name+'.jsonl')).open('w') as f:
  for x in items:f.write(json.dumps(x,ensure_ascii=False)+'\n')
meta={'source':str(source),'source_sha256':digest.hexdigest(),'source_rows':len(offsets),'seed':42,'tokenizer':tokenizer,'max_length':4096,'last_turn_loss_only':'auto','native_parser':'torchspec.data.dataset._tokenize_single','selected_count':len(records),'rejected':skipped,'train_ids':[x['data_id'] for x in records[:1000]],'eval_ids':[x['data_id'] for x in records[1000:]]}
meta['hashes']={n:hashlib.sha256((root/(n+'.jsonl')).read_bytes()).hexdigest() for n in ['train','eval']}
(root/'data-manifest.json').write_text(json.dumps(meta,indent=2))
(root/'raw-mask-stats.json').write_text(json.dumps(stats,indent=2))
print('SUMMARY',sum(x['L_raw'] for x in stats)/sum(x['N'] for x in stats),np.quantile([x['rho_loss_raw'] for x in stats],[0,.1,.5,.9,1]).tolist(),flush=True)
