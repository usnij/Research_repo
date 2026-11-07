"""Dependency-free commands and strict provenance for twelve micropatch runs."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

CAMPAIGN = 'micropatch_singleview_261105'
STEPS = (2000, 5000, 7000, 10000, 15000, 20000, 30000)
ARM_SPECS = (('micropatch_coordinate_sigma1p5', 'coordinate', 1.5),
             ('micropatch_coordinate_sigma2', 'coordinate', 2.),
             ('micropatch_patch2x2_ssim', 'patch', 2.))
NODE_SPECS = (('node04', 59233419, 'garden'), ('node02', 59233283, 'bicycle'),
              ('node01', 59233137, 'bonsai'), ('node03', 59233387, 'kitchen'))
SCENES = {
    'garden': dict(cap=2044672, initial_N=138766, downsample=4, train_views=161, test_views=24, shapes={'1297x840':185}),
    'bicycle': dict(cap=3021939, initial_N=54275, downsample=4, train_views=169, test_views=25, shapes={'1237x822':194}),
    'bonsai': dict(cap=1209886, initial_N=206613, downsample=2, train_views=255, test_views=37, shapes={'1559x1039':292}),
    'kitchen': dict(cap=1590685, initial_N=241367, downsample=2, train_views=244, test_views=35, shapes={'1558x1039':279}),
}
RENDERER = {'method':'3dgrt', 'pipeline_type':'reference', 'backward_pipeline_type':'referenceBwd',
            'particle_kernel_degree':4, 'particle_kernel_density_clamping':True,
            'particle_kernel_min_response':0.0113, 'particle_kernel_min_alpha':0.00392156862745098,
            'particle_kernel_max_alpha':0.99, 'particle_radiance_sph_degree':3,
            'primitive_type':'instances', 'min_transmittance':0.001, 'max_consecutive_bvh_update':15,
            'enable_normals':False, 'enable_hitcounts':True, 'enable_kernel_timings':False,
            'splat':{'medium_v2':False}}
REQUIRED_CODE = {'eval_full.py', 'src/sparse_grt/training.py', 'src/sparse_grt/config.py',
                 'src/sparse_grt/backend_config.json', 'src/sparse_grt/sampling.py',
                 'src/sparse_grt/contracts.py', 'src/sparse_grt/inputs.py',
                 'src/sparse_grt/micropatch.py', 'src/sparse_grt/micropatch_protocol.py',
                 'src/sparse_grt/losses/patch_ssim.py', 'src/sparse_grt/losses/coordinate_ssim_all.py',
                 'src/sparse_grt/losses/micropatch_coordinate.py',
                 'tools/'+CAMPAIGN+'/common.py', 'tools/'+CAMPAIGN+'/worker.py',
                 'tools/'+CAMPAIGN+'/prepare_plan.py', 'tools/'+CAMPAIGN+'/export_results.py',
                 'tools/'+CAMPAIGN+'/test_fleet.py'}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.pending')
    with pending.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    pending.replace(path)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*2**20), b''): digest.update(chunk)
    return digest.hexdigest()


def native_count(initial, cap, completed_steps):
    """Audit the untouched zero-based native add rule; never schedule births."""
    n = initial
    for _ in range(600, min(completed_steps, 25000), 100): n = min(cap, int(1.05*n))
    return n


def method_for(scene, sigma):
    return dict(views=1, block=3, sigma=sigma, l1_weight=.8, structure_weight=.2,
                opacity_weight=.01, scale_weight=.01, cap=SCENES[scene]['cap'], steps=30000,
                initialization_seed=777, view_seed=2610297100, ray_seed=910000000,
                boundary='all', grid_window=11, grid_sigma=1.5, sampler='micropatch')


def shape_geometry(shapes):
    rays, patches, edges = {}, {}, {}
    for shape in shapes:
        w,h = map(int, shape.split('x'))
        rays[shape] = ((w+2)//3)*((h+2)//3)
        patches[shape] = (w//6)*(h//6)
        edges[shape] = rays[shape]-4*patches[shape]
    return rays, patches, edges


def validate_arm_sequence(plan):
    if plan.get('campaign') != CAMPAIGN or plan.get('allow_experiments') != ['MICROPATCH']:
        raise ValueError('This launcher accepts only the independent micropatch campaign')
    nodes = plan.get('nodes', [])
    if len(nodes) != 4:
        raise ValueError('Require exactly four scene owners and twelve micropatch arms')
    ids = []
    for node, (node_id, instance, scene) in zip(nodes, NODE_SPECS):
        if (node.get('node_id'), node.get('instance_id'), node.get('primary_scene')) != (node_id, instance, scene):
            raise ValueError('Registered scene/node/instance ownership or queue order differs')
        if node.get('assigned_scenes') != [scene] or len(node.get('runs', [])) != 3:
            raise ValueError('Every node must own its scene and exactly three micropatch arms')
        metadata = SCENES[scene]
        rays, patches, edges = shape_geometry(metadata['shapes'])
        for run, (arm, loss, sigma) in zip(node['runs'], ARM_SPECS):
            run_id = f'micropatch_N5_{scene}_k1_{arm}_r9_seed777'
            expected = dict(run_id=run_id, experiment_id='MICROPATCH', scene=scene, arm=arm,
                            structural_loss=loss, method=method_for(scene, sigma), schedule_scale=1,
                            checkpoint_steps=list(STEPS), evaluation_steps=list(STEPS),
                            initial_N=metadata['initial_N'], downsample=metadata['downsample'],
                            train_views=metadata['train_views'], test_views=metadata['test_views'],
                            shapes=metadata['shapes'], rays_by_shape=rays,
                            complete_2x2_patches_by_shape=patches, edge_fallback_rays_by_shape=edges,
                            nominal_ray_fraction=1/9, renderer_settings=RENDERER,
                            config_path=f'configs/{run_id}.json',
                            sigma_config_unused=(loss == 'patch'),
                            expected_G_by_checkpoint={str(s):native_count(metadata['initial_N'],metadata['cap'],s) for s in STEPS})
            if any(run.get(key) != value for key,value in expected.items()):
                raise ValueError('Frozen micropatch configuration/geometry differs: '+str(run.get('run_id')))
            ids.append(run_id)
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate run IDs')
    if (plan.get('first_run_id') != nodes[0]['runs'][0]['run_id'] or
            plan.get('first_node_id') != nodes[0]['node_id'] or
            plan.get('first_2k_gate_required') is not True):
        raise ValueError('The first 2k gate must reference the authorized Garden sigma1.5 arm')
    if (plan.get('automatic_checkpoint_deletion') is not False or
            plan.get('no_automatic_retry') is not True or
            plan.get('prior_campaign_records_untouched') is not True):
        raise ValueError('Checkpoint preservation and isolation requirements differ')


def load_plan(path):
    plan = read(path)
    if plan.get('plan_signature') != signature({k:v for k,v in plan.items() if k!='plan_signature'}):
        raise ValueError('Plan signature mismatch')
    if not plan.get('frozen_commit') or not plan.get('code_files'):
        raise ValueError('Plan has not been frozen with code provenance')
    validate_arm_sequence(plan)
    names = [r.get('path') for r in plan['code_files']]
    if len(names) != len(set(names)) or not REQUIRED_CODE.issubset(names):
        raise ValueError('Frozen code inventory is incomplete or duplicated')
    for record in plan['code_files']:
        path = PurePosixPath(record['path'])
        if path.is_absolute() or '..' in path.parts or not record.get('sha256'):
            raise ValueError('Invalid frozen code path/hash')
    return plan


def run_signature(plan, run):
    return signature(dict(plan_signature=plan['plan_signature'], run=run))


def verify_code(root, plan):
    code = Path(root)/'code'
    for record in plan['code_files']:
        path = code/record['path']
        if not path.is_file() or sha(path) != record['sha256']:
            raise ValueError('Frozen source mismatch: '+str(path))


def selected_node(plan, node_id):
    matches = [n for n in plan['nodes'] if n['node_id']==node_id]
    if len(matches) != 1: raise ValueError('Unknown or duplicated node: '+node_id)
    return matches[0]


def train_command(root, run, python):
    root = Path(root)
    return [str(python), '-u', '-m', 'sparse_grt.training', '--scene', str(root/'data'/run['scene']),
            '--downsample', str(run['downsample']), '--config', str(root/'fleet'/run['config_path']),
            '--output', str(root/'runs'/run['run_id']), '--budget', 'fixed', '--growth', 'native_5percent',
            '--budget-views', 'all', '--cache-mib', '1200', '--mode', 'train',
            '--keep-checkpoints', ','.join(map(str,STEPS)), '--protocol', 'micropatch_singleview',
            '--structural-loss', run['structural_loss']]


def eval_command(root, run, step, python):
    if step not in STEPS: raise ValueError('Unregistered evaluation checkpoint')
    root = Path(root); out = root/'runs'/run['run_id']
    return [str(python), '-u', str(root/'code'/'eval_full.py'), '--checkpoint', str(out/f'ckpt_{step}.pt'),
            '--data-root', str(root/'data'/run['scene']), '--output', str(out/f'evaluation_{step}.json'),
            '--run-dir', str(out), '--run-id', run['run_id']]
