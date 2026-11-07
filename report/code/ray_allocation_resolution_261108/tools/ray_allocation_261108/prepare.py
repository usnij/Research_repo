import argparse,subprocess
from pathlib import Path
from common import *

def run(scene,experiment,arm,train_ds,eval_ds,K,block,sigma,loss='coordinate'):
 from sparse_grt.config import MethodConfig
 spec=SPECS[scene]
 method=vars(MethodConfig(views=K,block=block,sigma=sigma,l1_weight=.8,structure_weight=.2,
                         cap=spec['cap'],boundary='all'))
 rid=f'allocres_261108_{experiment}_{scene}_{arm}_seed777'
 w,h=spec['shapes'][train_ds];n=((h+block-1)//block)*((w+block-1)//block)
 if experiment=='E1':
  n6=((h+5)//6)*((w+5)//6);n8=((h+7)//8)*((w+7)//8)
  rays=([4*n8,4*(2*n6-n8)] if arm in ('schedule','combined') else [4*n6,4*n6])
 else:rays=[K*n,K*n]
 return dict(run_id=rid,experiment_id=experiment,scene=scene,arm=arm,train_ds=train_ds,eval_ds=eval_ds,
             allocation_arm=arm if experiment=='E1' else 'none',method=method,loss=loss,
             train_shape=[w,h],eval_shape=spec['shapes'][eval_ds],train_views=spec['train_views'],
             test_views=spec['test_views'],initial_N=spec['initial_N'],rays_first_second=rays,
             total_rays=15000*sum(rays),config_path=f'configs/{rid}.json',checkpoint_steps=STEPS,
             expected_G={str(s):count(spec['initial_N'],spec['cap'],s) for s in STEPS})
def main():
 p=argparse.ArgumentParser();p.add_argument('--fleet',type=Path,required=True);p.add_argument('--freeze',action='store_true');a=p.parse_args()
 code=Path(__file__).resolve().parents[2];nodes=[]
 for node,ident,scene,host,port,proxy,proxyport in NODES:
  alloc=[];res=[]
  if scene!='bonsai':
   alloc=[run(scene,'E1',x,SPECS[scene]['standard_ds'],SPECS[scene]['standard_ds'],4,6,6.)
          for x in ('uniform','schedule','residual','combined')]
  if scene!='bicycle':
   target=2 if scene=='garden' else 1
   res=[run(scene,'E2','highres_dense',target,target,1,1,6.,'dense'),
        run(scene,'E2','sparse_r16',target,target,4,8,6.)]
   if scene!='garden':
    res+=[run(scene,'E2','sparse_r64',target,target,4,16,12.),
          run(scene,'E2','lowres_dense_ds4',4,target,1,1,6.,'dense')]
   res+=[run(scene,'E2','lowres_dense_ds8',8,target,1,1,6.,'dense')]
  ordered=(res[:1]+alloc[:2]+res[1:]+alloc[2:])
  for i,r in enumerate(ordered,1):
   r['execution_order']=i
   write(a.fleet/r['config_path'],r['method']);r['config_sha256']=sha(a.fleet/r['config_path'])
  nodes.append(dict(node_id=node,instance_id=ident,scene=scene,direct=dict(host=host,port=port),
                    proxy=dict(host=proxy,port=proxyport),runs=ordered))
 assert sum(len(n['runs']) for n in nodes)==25
 for scene in ('counter','garden','bicycle'):
  armruns=[r for n in nodes for r in n['runs'] if r['scene']==scene and r['experiment_id']=='E1']
  assert len({r['total_rays'] for r in armruns})==1
 plan=dict(schema_version=1,campaign=CAMPAIGN,remote_root='/workspace/'+CAMPAIGN,
           python='/workspace/cvpr2027_env/venv/bin/python',nodes=nodes,
           frozen_commit=None,code_files=[],no_automatic_destroy=True,no_automatic_retry=True,
           first_run_id=nodes[3]['runs'][0]['run_id'],first_node_id='node04',
           changes_from_order=['Exact ray equality: thin block5 phase to 2*N6-N8 per view',
                              'E1/E2 launches follow complete code freeze; environment prep runs in parallel'],
           eval_metric='torchmetrics SSIM window11 sigma1.5; LPIPS VGG; full target-resolution test images',
           growth='native MCMC 1.05, 500<index<25000, every100, same original cap',
           renderer_settings=read(code/'src/sparse_grt/backend_config.json')['render'])
 if a.freeze:
  if (a.fleet/'plan.json').exists():raise FileExistsError('Already frozen')
  if subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True).strip():raise ValueError('Commit all code first')
  plan['frozen_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
  files=subprocess.check_output(['git','ls-files','-z'],cwd=code).decode().split('\0')
  plan['code_files']=[dict(path=x,sha256=sha(code/x)) for x in files if x and (code/x).is_file()]
 plan['plan_signature']=signature(plan)
 write(a.fleet/('plan.json' if a.freeze else 'plan.draft.json'),plan)
 print(json.dumps(dict(runs=25,nodes=4,commit=plan['frozen_commit'])))
if __name__=='__main__':main()
