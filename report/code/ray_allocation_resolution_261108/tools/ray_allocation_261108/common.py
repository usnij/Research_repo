"""Frozen source, command and run definitions for the authorized 25 runs."""
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
CAMPAIGN='ray_allocation_resolution_261108'
STEPS=[2000,5000,7000,10000,15000,20000,30000]
SPECS={
 'counter':dict(cap=1229851,initial_N=155767,train_views=210,test_views=30,standard_ds=2,
                shapes={1:[3115,2076],2:[1558,1038],4:[779,519],8:[389,260]}),
 'garden':dict(cap=2044672,initial_N=138766,train_views=161,test_views=24,standard_ds=4,
               shapes={2:[2594,1681],4:[1297,840],8:[648,420]}),
 'bonsai':dict(cap=1209886,initial_N=206613,train_views=255,test_views=37,standard_ds=2,
               shapes={1:[3118,2078],4:[780,520],8:[390,260]}),
 'bicycle':dict(cap=3021939,initial_N=54275,train_views=169,test_views=25,standard_ds=4,
                shapes={4:[1237,822]})}
NODES=[('node01',59592874,'counter','47.186.21.5',54848,'ssh8.vast.ai',24229),
       ('node02',59592898,'garden','194.14.47.19',23008,'ssh5.vast.ai',30757),
       ('node03',59592915,'bonsai','194.14.47.19',24802,'ssh1.vast.ai',16631),
       ('node04',59592933,'bicycle','38.117.87.51',44799,'ssh9.vast.ai',14027)]
def read(p): return json.loads(Path(p).read_text())
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 temp=p.with_suffix(p.suffix+'.pending');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(p)
def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def signature(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def count(initial,cap,step):
 n=initial
 for t in range(600,min(step,25000),100):n=min(cap,int(n*1.05))
 return n

def train_command(root,run,python):
 root=Path(root)
 return [python,'-u','-m','sparse_grt.training','--scene',str(root/'data'/run['scene']),
         '--downsample',str(run['train_ds']),'--eval-downsample',str(run['eval_ds']),
         '--config',str(root/'fleet'/run['config_path']),'--output',str(root/'runs'/run['run_id']),
         '--budget','fixed','--growth','native_5percent','--budget-views','all','--cache-mib','1200',
         '--mode','train','--protocol','allocation_resolution','--structural-loss',run['loss'],
         '--allocation-arm',run['allocation_arm']]
def eval_command(root,run,step,python):
 root=Path(root);out=root/'runs'/run['run_id']
 cmd=[python,'-u',str(root/'code/eval_full.py'),'--checkpoint',str(out/f'ckpt_{step}.pt'),
      '--data-root',str(root/'data'/run['scene']),'--output',str(out/f'evaluation_{step}.json'),
      '--run-dir',str(out),'--run-id',run['run_id'],'--eval-downsample',str(run['eval_ds'])]
 if run['allocation_arm']=='uniform':cmd+=['--residual-maps',str(out/f'residual_maps_{step}')]
 return cmd

def verify_code(root,plan):
 for f in plan['code_files']:
  if sha(Path(root)/'code'/f['path'])!=f['sha256']:raise RuntimeError('Source changed: '+f['path'])
 for n in plan['nodes']:
  for r in n['runs']:
   if sha(Path(root)/'fleet'/r['config_path'])!=r['config_sha256']:raise RuntimeError('Config changed')
 body={k:v for k,v in plan.items() if k!='plan_signature'}
 if signature(body)!=plan['plan_signature']:raise RuntimeError('Plan changed')
