"""CPU-only integration guards for preparation, freezing and the twelve-run queue."""
import copy
import csv
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from common import (ARM_SPECS, CAMPAIGN, NODE_SPECS, REQUIRED_CODE, SCENES, STEPS,
                    load_plan, native_count, read, sha, signature, train_command, write)
from prepare_plan import prepare, freeze
from worker import (gate, machine_lock, preflight, required_queue_disk_bytes, run_queue, validate_identity, verify_training)
from export_results import export


def draft_fixture(root):
    metadata, nodes, hosts = [], [], []
    for node, instance, scene in NODE_SPECS:
        spec=SCENES[scene];audit=root/scene/'audit.json'
        write(audit,dict(scene=scene,target_N=spec['cap']))
        metadata.append(dict(scene=scene,node_id=node,instance_id=instance,**spec,
                             cap_source=str(audit),cap_source_sha256=sha(audit)))
        nodes.append(dict(node_id=node,instance_id=instance,primary_scene=scene,runs=[
            dict(arm=arm,sigma=sigma if loss=='coordinate' else None,steps=30000,cap=spec['cap'])
            for arm,loss,sigma in ARM_SPECS]))
        hosts.append(dict(name=node,instance_id=instance,host='example.invalid',port=22))
    path=root/'draft.json';write(path,dict(campaign=CAMPAIGN,frozen=False,training_started=False,
                                         scene_metadata=metadata,nodes=nodes))
    hp=root/'hosts.json';write(hp,hosts)
    return path,hp


def sign(plan):
    plan=copy.deepcopy(plan);plan.pop('plan_signature',None)
    plan['frozen_commit']='fixture-commit'
    plan['code_files']=[dict(path=p,sha256='fixture') for p in sorted(REQUIRED_CODE)]
    plan['plan_signature']=signature(plan)
    return plan


def training_fixture(out,run):
    out.mkdir(parents=True,exist_ok=True)
    cap=run['method']['cap'];sigma=run['method']['sigma'];is_patch=run['structural_loss']=='patch'
    write(out/'training_complete.json',dict(passed=True,steps=30000,N=cap,from_scratch=True,
        final_optimizer_included=True,phase='training_complete',evaluation_seconds=0.,
        evaluation_steps=list(STEPS),training_seconds=300.,checkpoint_seconds=1.))
    protocol=dict(method=run['method'],train_views=run['train_views'],downsample=run['downsample'],
        growth='native_5percent',growth_end=None,structural_loss=run['structural_loss'],
        execution_protocol='micropatch_singleview',checkpoint_steps=list(STEPS),
        evaluation_phase='after_training_external_eval_full',sampler='micropatch',sampling_macro_tile=[6,6],
        observed_patch=[2,2],anchor_offsets=[0,2,4],anchor_probability=1/9,
        exact_ray_count='ceil(H/3)*ceil(W/3)',edge_policy='legacy_3x3_points_in_partial_6x6_tiles',
        observed_only=False,observed_RGB_only=True,full_model_regularization=True,multiview_start_step=0,
        native_5percent_growth_preserved=True,forced_cap_arrival=False,
        native_add_schedule=dict(original_add_method_preserved=True,start_exclusive=500,end_exclusive=25000,
            first_event=600,last_event=24900,frequency=100),
        coordinate_physical_support=None if is_patch else 3*sigma,coordinate_support_shape='axis_aligned_square',
        coordinate_sigma_config_unused=2. if is_patch else None,
        ssim_backend='torch_observed_patch_population_moments' if is_patch else 'coordinate_ssim_all')
    if is_patch:protocol['patch_loss']=dict(support='four_actual_observed_contiguous_pixels',weights=[.25]*4,
        moments='population',C1=.0001,C2=.0009,scale='1-SSIM',
        reduction='mean_over_complete_patches_and_RGB_channels',boundary='fallback_rays_L1_only')
    write(out/'protocol.json',protocol)
    write(out/'resolved_config.json',dict(render=run['renderer_settings'],strategy=dict(add=dict(
        start_iteration=500,end_iteration=25000,frequency=100,placement='opacity',max_n_gaussians=cap))))
    write(out/'growth_plan.json',dict(policy='native_5percent',audit_only=True,initial=run['initial_N'],
                                     target=cap,start=500,end=25000,frequency=100))
    write(out/'initial.json',dict(N=run['initial_N']))
    shape=next(iter(run['shapes']));w,h=map(int,shape.split('x'))
    g=dict(width=w,height=h,sampler='micropatch',macro_tile=[6,6],patch_shape=[2,2],
        total_rays_if_equal_views=run['rays_by_shape'][shape],patch_count=run['complete_2x2_patches_by_shape'][shape],
        patch_rays=4*run['complete_2x2_patches_by_shape'][shape],
        boundary_fallback_rays=run['edge_fallback_rays_by_shape'][shape],
        edge_policy='legacy_3x3_points_in_partial_6x6_tiles',physical_support_radius_pixels=3*sigma,
        structural_boundary_policy='exclude_all_fallback_rays' if is_patch else None,
        sigma_pixels=None if is_patch else sigma)
    write(out/'ray_geometry.json',{str(i):g for i in range(run['train_views'])})
    for step in STEPS:
        checkpoint=out/f'ckpt_{step}.pt';checkpoint.write_bytes(b'CPU fixture only')
        write(out/f'checkpoint_verified_{step}.json',dict(step=step,G=run['expected_G_by_checkpoint'][str(step)],
            load_verified=True,optimizer_present=True,optimizer_moment_shapes_verified=True,
            bytes=checkpoint.stat().st_size,training_seconds=step*.01))
    with (out/'step_times.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['step','mean_step_seconds_100','G'])
        for step in range(100,30001,100):writer.writerow([step,.01,native_count(run['initial_N'],cap,step)])


def evaluation_fixture(out,run,step):
    rows=[dict(view=i,image=f'{i}.jpg',PSNR=25.,SSIM=.8,LPIPS=.2) for i in range(run['test_views'])]
    write(out/f'evaluation_{step}.json',dict(step=step,run_id=run['run_id'],views=run['test_views'],
        checkpoint_sha256=sha(out/f'ckpt_{step}.pt'),optimizer_state_present=True,
        N=run['expected_G_by_checkpoint'][str(step)],PSNR=25.,SSIM=.8,LPIPS=.2,per_view=rows))


class MicropatchFleetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)/'campaign';self.root.mkdir()
        self.draft,self.hosts=draft_fixture(self.root)
        self.args=SimpleNamespace(draft=self.draft,hosts=self.hosts,output=self.root/'fleet',
            remote_root='/workspace/'+CAMPAIGN,python='/env/python')
        self.plan=sign(prepare(self.args));write(self.root/'fleet/plan.json',self.plan)

    def tearDown(self):self.tmp.cleanup()

    def test_exact_twelve_authorized_commands_and_config_placeholder(self):
        p=load_plan(self.root/'fleet/plan.json');self.assertEqual(sum(len(n['runs']) for n in p['nodes']),12)
        for n in p['nodes']:
            for run in n['runs']:
                cmd=train_command(self.root,run,'python')
                self.assertEqual(cmd[cmd.index('--protocol')+1],'micropatch_singleview')
                self.assertEqual(cmd[cmd.index('--mode')+1],'train');self.assertNotIn('--growth-end',cmd)
                self.assertEqual(run['method']['sampler'],'micropatch')
                self.assertEqual(read(self.root/'fleet'/run['config_path']),run['method'])
                self.assertEqual(run['sigma_config_unused'],run['structural_loss']=='patch')
        self.assertEqual(p['nodes'][-1]['runs'][0]['rays_by_shape'],{'1558x1039':180440})
        self.assertEqual(p['first_run_id'],'micropatch_N5_garden_k1_micropatch_coordinate_sigma1p5_r9_seed777')
        for node in p['nodes']:
            self.assertEqual([(r['structural_loss'],r['method']['sigma']) for r in node['runs']],
                             [('coordinate',1.5),('coordinate',2.),('patch',2.)])

    def test_signed_mutations_rejected(self):
        for mutation in ('instance','cap','downsample','extra_arm','point','sigma','loss','edge','renderer','seed','steps'):
            p=copy.deepcopy(self.plan);run=p['nodes'][0]['runs'][0]
            if mutation=='instance':p['nodes'][0]['instance_id']=99
            if mutation=='cap':run['method']['cap']+=1
            if mutation=='downsample':run['downsample']=2
            if mutation=='extra_arm':p['nodes'][0]['runs'].append(copy.deepcopy(run))
            if mutation=='point':run['method']['sampler']='point'
            if mutation=='sigma':run['method']['sigma']=5.
            if mutation=='loss':run['structural_loss']='grid'
            if mutation=='edge':run['edge_fallback_rays_by_shape']['1297x840']+=1
            if mutation=='renderer':run['renderer_settings']['particle_kernel_degree']=2
            if mutation=='seed':run['method']['ray_seed']+=1
            if mutation=='steps':run['checkpoint_steps'].remove(2000)
            p.pop('plan_signature');p['plan_signature']=signature(p);write(self.root/'bad.json',p)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):load_plan(self.root/'bad.json')

    def test_retired_sigma5_draft_or_mislabeled_sigma_is_rejected(self):
        original=read(self.draft)
        for stale_name in (True,False):
            draft=copy.deepcopy(original)
            for node in draft['nodes']:
                node['runs'][0]['sigma']=5.
                if stale_name:node['runs'][0]['arm']='micropatch_coordinate_sigma5'
            write(self.draft,draft)
            args=copy.copy(self.args);args.output=self.root/('stale_name' if stale_name else 'stale_sigma')
            with self.assertRaisesRegex(ValueError,'Draft arm list'):
                prepare(args)
            self.assertFalse(args.output.exists())

    def test_retired_sigma5_signed_plan_and_gate_are_rejected(self):
        plan=copy.deepcopy(self.plan)
        for node in plan['nodes']:
            run=node['runs'][0]
            run['arm']='micropatch_coordinate_sigma5'
            run['run_id']=run['run_id'].replace('sigma1p5','sigma5')
            run['config_path']=f"configs/{run['run_id']}.json"
            run['method']['sigma']=5.
        plan['first_run_id']=plan['nodes'][0]['runs'][0]['run_id']
        plan.pop('plan_signature');plan['plan_signature']=signature(plan)
        write(self.root/'retired_plan.json',plan)
        with self.assertRaisesRegex(ValueError,'configuration/geometry differs'):
            load_plan(self.root/'retired_plan.json')
        write(self.root/'.first_2k_gate.json',dict(plan_signature=plan['plan_signature'],
            first_run_id=plan['first_run_id'],checkpoint_verified=True,step=2000))
        with self.assertRaisesRegex(RuntimeError,'another plan'):
            gate(self.root,self.plan,self.plan['nodes'][1]['runs'][0])

    def test_sigma1p5_physical_support_must_be_exactly_4p5(self):
        run=self.plan['nodes'][0]['runs'][0];out=self.root/'support_probe'
        for artifact in ('protocol','geometry'):
            training_fixture(out,run)
            self.assertEqual(verify_training(out,run)['steps'],30000)
            if artifact=='protocol':
                p=out/'protocol.json';value=read(p)
                self.assertEqual(value['coordinate_physical_support'],4.5)
                value['coordinate_physical_support']=4.;write(p,value)
            else:
                p=out/'ray_geometry.json';value=read(p)
                self.assertEqual(value['0']['physical_support_radius_pixels'],4.5)
                value['0']['physical_support_radius_pixels']=4.;write(p,value)
            with self.subTest(artifact=artifact),self.assertRaisesRegex(RuntimeError,'support|bandwidth'):
                verify_training(out,run)

    def test_native_count_has_no_forced_arrival_or_horizon(self):
        self.assertEqual(native_count(100,1000,600),100);self.assertEqual(native_count(100,1000,601),105)
        self.assertEqual(native_count(1,1000,30000),1)
        self.assertEqual(native_count(138766,2044672,2000),274737)

    def test_all_twelve_completion_geometry_optimizer_and_timing(self):
        for n in self.plan['nodes']:
            for run in n['runs']:
                out=self.root/run['run_id'];training_fixture(out,run)
                self.assertEqual(verify_training(out,run)['steps'],30000)

    def test_incorrect_runtime_evidence_stops_queue(self):
        run=self.plan['nodes'][0]['runs'][2];out=self.root/'probe'
        for mutation in ('optimizer','timing','edge','renderer','growth','early_eval','patch_population'):
            training_fixture(out,run)
            if mutation=='optimizer':
                p=out/'checkpoint_verified_30000.json';x=read(p);x['optimizer_present']=False;write(p,x)
            elif mutation=='timing':
                p=out/'step_times.csv';p.write_text('\n'.join(p.read_text().splitlines()[:-1])+'\n')
            elif mutation=='edge':
                p=out/'ray_geometry.json';x=read(p);x['0']['boundary_fallback_rays']+=1;write(p,x)
            elif mutation=='renderer':
                p=out/'resolved_config.json';x=read(p);x['render']['max_consecutive_bvh_update']=16;write(p,x)
            elif mutation=='growth':
                p=out/'growth_plan.json';x=read(p);x['end']=5000;write(p,x)
            elif mutation=='early_eval':
                p=out/'training_complete.json';x=read(p);x['evaluation_seconds']=1;write(p,x)
            else:
                p=out/'protocol.json';x=read(p);x['patch_loss']['moments']='sample';write(p,x)
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):verify_training(out,run)

    def test_gate_and_cross_campaign_lock(self):
        first=self.plan['nodes'][0]['runs'][0];second=self.plan['nodes'][1]['runs'][0]
        gate(self.root,self.plan,first)
        with self.assertRaises(FileNotFoundError):gate(self.root,self.plan,second)
        write(self.root/'.first_2k_gate.json',dict(plan_signature=self.plan['plan_signature'],
            first_run_id=self.plan['first_run_id'],checkpoint_verified=True,step=2000))
        gate(self.root,self.plan,second)
        other=self.root.parent/'other';other.mkdir()
        with machine_lock(self.root),self.assertRaises(RuntimeError):
            with machine_lock(other):pass

    def test_identity_rejects_wrong_gpu_or_instance(self):
        node=self.plan['nodes'][0];hw=dict(uuid='GPU-A')
        ident=dict(node_id=node['node_id'],instance_id=node['instance_id'],gpu_uuid='GPU-A')
        validate_identity(ident,node,hw,{})
        with self.assertRaises(RuntimeError):validate_identity(ident,node,hw,{'CONTAINER_ID':'99'})
        with self.assertRaises(RuntimeError):validate_identity(ident,node,dict(uuid='GPU-B'),{})

    def test_unfrozen_or_untracked_source_cannot_launch_or_freeze(self):
        p=read(self.root/'fleet/plan.draft.json');write(self.root/'unfrozen.json',p)
        with self.assertRaises(ValueError):load_plan(self.root/'unfrozen.json')
        (self.root/'fleet/plan.json').unlink()
        args=SimpleNamespace(output=self.root/'fleet',code=self.root,commit='fixture')
        with patch('prepare_plan.subprocess.check_output',side_effect=['fixture\n','?? src/new_sampler.py\n']):
            with self.assertRaisesRegex(ValueError,'including new files'):freeze(args)

    def test_frozen_plan_refuses_regeneration(self):
        with self.assertRaises(FileExistsError):prepare(self.args)

    def test_train_then_seven_evaluations_per_arm_and_export(self):
        node=self.plan['nodes'][0];hardware=dict(uuid='GPU-X',name='NVIDIA GeForce RTX 4090',
            driver='test',memory_mib='24564',power_limit_w='450.00');events=[]
        byid={r['run_id']:r for r in node['runs']}
        def execute(cmd,log,code,env):
            if '--mode' in cmd:
                out=Path(cmd[cmd.index('--output')+1]);run=byid[out.name]
                events.append(('train',run['run_id'],30000));training_fixture(out,run)
                self.assertFalse(list(out.glob('evaluation_*.json')))
            else:
                rid=cmd[cmd.index('--run-id')+1];run=byid[rid];out=self.root/'runs'/rid
                self.assertTrue((out/'training_complete.json').exists())
                step=int(Path(cmd[cmd.index('--checkpoint')+1]).stem.split('_')[1])
                events.append(('eval',rid,step));evaluation_fixture(out,run,step)
        args=SimpleNamespace(root=self.root,node=node['node_id'],resume_evaluation=False)
        with patch('worker.preflight',return_value=hardware),patch('worker.gpu_inventory',return_value=hardware),\
             patch('worker.gate'),patch('worker.execute',side_effect=execute),\
             patch('worker.shutil.disk_usage',return_value=SimpleNamespace(free=1000*2**30)):
            run_queue(args)
            before=len(events);run_queue(args);self.assertEqual(len(events),before)
        self.assertEqual(len(events),24)
        for index,run in enumerate(node['runs']):
            self.assertEqual(events[8*index],('train',run['run_id'],30000))
            self.assertEqual([v[2] for v in events[8*index+1:8*index+8]],list(STEPS))
        export(self.plan,{node['node_id']:self.root},self.root/'export')
        with (self.root/'export/runs.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),12);self.assertEqual(sum(r['status']=='complete' for r in rows),3)
        patch_row=next(r for r in rows if r['scene']=='garden' and r['arm'].endswith('patch2x2_ssim'))
        self.assertEqual(patch_row['sigma_pixels'],'NA');self.assertEqual(patch_row['sampler'],'micropatch')

    def test_partial_training_never_automatically_resumes(self):
        node=self.plan['nodes'][0];run=node['runs'][0];(self.root/'runs'/run['run_id']).mkdir(parents=True)
        args=SimpleNamespace(root=self.root,node=node['node_id'],resume_evaluation=False)
        with patch('worker.preflight',return_value={}),patch('worker.gpu_inventory',return_value={}),patch('worker.gate'),patch('worker.shutil.disk_usage',return_value=SimpleNamespace(free=1000*2**30)):
            with self.assertRaisesRegex(RuntimeError,'Partial run'):run_queue(args)

    def test_freeze_binds_all_committed_files_and_configs(self):
        (self.root/'fleet/plan.json').unlink()
        code=self.root/'code'
        for name in REQUIRED_CODE:
            p=code/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture source')
        from common import RENDERER, verify_code
        write(code/'src/sparse_grt/backend_config.json',dict(render=RENDERER))
        args=SimpleNamespace(output=self.root/'fleet',code=code,commit='fixture')
        with patch('prepare_plan.subprocess.check_output',side_effect=[
                'fixture\n','',('\0'.join(sorted(REQUIRED_CODE))+'\0').encode()]):
            plan=freeze(args)
        self.assertEqual(load_plan(self.root/'fleet/plan.json')['plan_signature'],plan['plan_signature'])
        verify_code(self.root,plan)
        (code/'src/sparse_grt/sampling.py').write_text('modified after freeze')
        with self.assertRaisesRegex(ValueError,'Frozen source'):verify_code(self.root,plan)

    def test_preflight_binds_readiness_config_scene_and_identity(self):
        node=self.plan['nodes'][0];hardware=dict(uuid='GPU-X')
        write(self.root/'.environment_ready.json',dict(passed=True,decoder_verified=True,
            frozen_commit=self.plan['frozen_commit'],plan_signature=self.plan['plan_signature'],
            node_id=node['node_id'],instance_id=node['instance_id'],datasets_verified=['garden']))
        write(self.root/'.deployment_identity.json',dict(node_id=node['node_id'],
            instance_id=node['instance_id'],gpu_uuid='GPU-X'))
        source=self.root/'data/garden/sparse/0/images.bin';source.parent.mkdir(parents=True);source.write_bytes(b'fixture')
        with patch('worker.__file__',str(self.root/'code/tools'/CAMPAIGN/'worker.py')),\
             patch('worker.verify_code'),patch('worker.gpu_inventory',return_value=hardware),\
             patch.dict('worker.os.environ',{'GRT_GRAD_ACCUM_DOUBLE':'1'},clear=True):
            self.assertEqual(preflight(self.root,self.plan,node),hardware)
            config=self.root/'fleet'/node['runs'][0]['config_path'];x=read(config);x['sigma']=4.;write(config,x)
            with self.assertRaisesRegex(RuntimeError,'Frozen method config'):preflight(self.root,self.plan,node)

    def test_queue_disk_reserves_three_runs_without_pruning(self):
        runs=self.plan['nodes'][0]['runs'];required=required_queue_disk_bytes(runs)
        raw=3*712*sum(runs[0]['expected_G_by_checkpoint'].values())
        self.assertGreater(required,raw+4*2**30)
        self.assertEqual(required_queue_disk_bytes([]),0)

    def test_hardware_change_between_arms_rejected(self):
        args=SimpleNamespace(root=self.root,node='node04',resume_evaluation=False)
        with patch('worker.preflight',return_value={'uuid':'GPU-A'}),patch('worker.gpu_inventory',return_value={'uuid':'GPU-B'}),patch('worker.gate'),patch('worker.shutil.disk_usage',return_value=SimpleNamespace(free=1000*2**30)):
            with self.assertRaisesRegex(RuntimeError,'changed within'):run_queue(args)


if __name__=='__main__':unittest.main()
