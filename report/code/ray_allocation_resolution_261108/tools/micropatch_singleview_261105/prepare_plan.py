"""Build the twelve-run draft locally; freeze only a reviewed, committed source tree."""
import argparse
from pathlib import Path
import shlex
import subprocess

from common import (ARM_SPECS, CAMPAIGN, NODE_SPECS, RENDERER, REQUIRED_CODE, SCENES, STEPS,
                    eval_command, method_for, native_count, read, sha, shape_geometry,
                    signature, train_command, validate_arm_sequence, write)


def prepare(args):
    output = args.output.resolve()
    if (output/'plan.json').exists(): raise FileExistsError('Refusing to overwrite a frozen plan')
    draft = read(args.draft)
    if (draft.get('campaign') != CAMPAIGN or draft.get('frozen') is not False or
            draft.get('training_started') is not False):
        raise ValueError('Expected the local-only unfrozen micropatch preparation draft')
    hosts = read(args.hosts) if args.hosts else []
    if isinstance(hosts, dict): hosts = hosts['nodes']
    hosts = {h.get('name',h.get('node_id')):h for h in hosts}
    metadata = {s['scene']:s for s in draft['scene_metadata']}
    planned = {n['node_id']:n for n in draft['nodes']}
    if len(metadata)!=4 or len(planned)!=4: raise ValueError('Draft needs exactly four scenes/owners')
    nodes = []
    for node_id, instance, scene in NODE_SPECS:
        src = metadata[scene]; canonical = SCENES[scene]; assignment = planned[node_id]
        if (assignment['instance_id'] != instance or assignment['primary_scene'] != scene or
                len(assignment['runs']) != 3 or src['node_id'] != node_id or src['instance_id'] != instance):
            raise ValueError('Draft scene/instance/arm assignment differs')
        if any(src[k] != v for k,v in canonical.items()):
            raise ValueError('Draft cap, resolution or data count differs: '+scene)
        for proposed, (arm, loss, sigma) in zip(assignment['runs'], ARM_SPECS):
            if (proposed.get('arm') != arm or proposed.get('steps') != 30000
                    or proposed.get('cap') != canonical['cap']
                    or (loss == 'coordinate' and proposed.get('sigma') != sigma)):
                raise ValueError('Draft arm list differs from the authorized twelve runs')
        audit_path = Path(src['cap_source'])
        if sha(audit_path) != src['cap_source_sha256']:
            raise ValueError('Cap source changed: '+scene)
        audit = read(audit_path)
        if audit['scene'] != scene or audit['target_N'] != canonical['cap']:
            raise ValueError('Source audit cap differs: '+scene)
        host = hosts.get(node_id, {})
        if host and host['instance_id'] != instance:
            raise ValueError('Host inventory points at another instance: '+node_id)
        rays, patches, edges = shape_geometry(canonical['shapes'])
        runs = []
        for arm, loss, sigma in ARM_SPECS:
            method = method_for(scene,sigma)
            run_id = f'micropatch_N5_{scene}_k1_{arm}_r9_seed777'
            config_path = f'configs/{run_id}.json'
            runs.append(dict(run_id=run_id, experiment_id='MICROPATCH', scene=scene, arm=arm,
                structural_loss=loss, method=method, config_path=config_path, sigma_config_unused=(loss=='patch'),
                downsample=canonical['downsample'], train_views=canonical['train_views'], test_views=canonical['test_views'],
                initial_N=canonical['initial_N'], shapes=canonical['shapes'], rays_by_shape=rays,
                complete_2x2_patches_by_shape=patches, edge_fallback_rays_by_shape=edges,
                nominal_ray_fraction=1/9, schedule_scale=1, checkpoint_steps=list(STEPS), evaluation_steps=list(STEPS),
                renderer_settings=RENDERER,
                expected_G_by_checkpoint={str(s):native_count(canonical['initial_N'],canonical['cap'],s) for s in STEPS},
                cap_source=dict(rule='retained_fine_tracks_v1', source=str(audit_path), audit_sha256=sha(audit_path))))
        direct = host.get('direct') or (dict(host=host['host'],port=host['port']) if host.get('host') else None)
        proxy = host.get('proxy') or (dict(host=host['proxy_host'],port=host['proxy_port']) if host.get('proxy_host') else None)
        nodes.append(dict(node_id=node_id, instance_id=instance, primary_scene=scene,
                          assigned_scenes=[scene], direct=direct, proxy=proxy, runs=runs))
    plan = dict(schema_version=1,campaign=CAMPAIGN,allow_experiments=['MICROPATCH'],
        remote_root=args.remote_root,python=args.python,renderer='261103_exact_preserved',nodes=nodes,
        first_run_id=nodes[0]['runs'][0]['run_id'],first_node_id=nodes[0]['node_id'],first_2k_gate_required=True,
        automatic_checkpoint_deletion=False,no_automatic_retry=True,prior_campaign_records_untouched=True,
        evaluation_phase='after_training_external_eval_full',growth=dict(policy='native_5percent',
        start_exclusive=500,end_exclusive=25000,frequency=100,forced_cap_arrival=False,audit_only=True),
        source_draft_sha256=sha(args.draft),frozen_commit=None,code_files=[])
    validate_arm_sequence(plan)
    for node in nodes:
        for run in node['runs']:
            write(output/run['config_path'],run['method']);run['config_sha256']=sha(output/run['config_path'])
    write(output/'plan.draft.json',plan)
    return plan


def freeze(args):
    output = args.output.resolve(); code = args.code.resolve()
    if (output/'plan.json').exists(): raise FileExistsError('Refusing to overwrite a frozen plan')
    plan = read(output/'plan.draft.json');validate_arm_sequence(plan)
    actual_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
    if actual_commit != args.commit: raise ValueError('--commit must equal actual source HEAD')
    dirty = subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=code,text=True).strip()
    if dirty: raise ValueError('Commit the complete reviewed implementation, including new files, before freezing')
    names = subprocess.check_output(['git','ls-files','-z'],cwd=code).decode().split('\0')
    paths = sorted(name for name in names if name and (code/name).is_file())
    if not REQUIRED_CODE.issubset(paths): raise ValueError('Sampler, loss, training and runner must all be committed')
    if read(code/'src/sparse_grt/backend_config.json')['render'] != RENDERER:
        raise ValueError('Original [261103] renderer settings changed')
    for node in plan['nodes']:
        for run in node['runs']:
            p=output/run['config_path']
            if sha(p)!=run['config_sha256'] or read(p)!=run['method']:
                raise ValueError('Draft config changed before freezing')
    plan['code_files']=[dict(path=name,sha256=sha(code/name)) for name in paths]
    plan['frozen_commit']=actual_commit
    plan['plan_signature']=signature(plan)
    write(output/'plan.json',plan)
    lines=[f'# Frozen source: {actual_commit}',f'# Plan signature: {plan["plan_signature"]}',
           '# Training commands below are provenance; launch only through the guarded worker.']
    for node in plan['nodes']:
        lines.extend(['',f'# {node["node_id"]}, instance {node["instance_id"]}',
            shlex.join([plan['python'],str(Path(plan['remote_root'])/'code/tools'/CAMPAIGN/'worker.py'),
                        '--root',plan['remote_root'],'--node',node['node_id']])])
        for run in node['runs']:
            lines.extend([f'# {run["run_id"]}',shlex.join(train_command(plan['remote_root'],run,plan['python']))])
            lines.extend(shlex.join(eval_command(plan['remote_root'],run,s,plan['python'])) for s in STEPS)
    (output/'commands.txt').write_text('\n'.join(lines)+'\n')
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('prepare');a.add_argument('--draft',type=Path,required=True);a.add_argument('--hosts',type=Path)
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--remote-root',default='/workspace/'+CAMPAIGN)
    a.add_argument('--python',default='/workspace/cvpr2027_env/venv/bin/python')
    a=sub.add_parser('freeze');a.add_argument('--output',type=Path,required=True)
    a.add_argument('--code',type=Path,required=True);a.add_argument('--commit',required=True)
    args=p.parse_args();plan=prepare(args) if args.action=='prepare' else freeze(args)
    print(f'{args.action}: {len(plan["nodes"])} nodes, {sum(len(n["runs"]) for n in plan["nodes"])} runs')


if __name__=='__main__':main()
