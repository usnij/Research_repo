"""Run one frozen scene queue, one GPU at a time; never retry training silently."""
import argparse
import csv
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

try:
    from .common import (STEPS, count, eval_command, now, read, sha, signature,
                         train_command, verify_code, write)
except ImportError:
    from common import (STEPS, count, eval_command, now, read, sha, signature,
                        train_command, verify_code, write)


def run_signature(plan, run):
    return signature(dict(plan_signature=plan['plan_signature'], run=run))


def load_plan(path):
    plan = read(path)
    body = {k: v for k, v in plan.items() if k != 'plan_signature'}
    if signature(body) != plan.get('plan_signature'):
        raise RuntimeError('Plan signature mismatch')
    if len([r for n in plan['nodes'] for r in n['runs']]) != 25:
        raise RuntimeError('Expected exactly 25 authorized runs')
    return plan


def selected_node(plan, node_id):
    nodes = [n for n in plan['nodes'] if n['node_id'] == node_id]
    if len(nodes) != 1:
        raise ValueError('Node is absent or duplicated in the frozen plan')
    return nodes[0]


@contextmanager
def machine_lock(root):
    path = root.parent / '.sparse-grt-gpu0.lock'
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another worker holds the shared GPU lock') from exc
        yield


def gpu_inventory():
    text = subprocess.check_output(['nvidia-smi',
        '--query-gpu=uuid,name,driver_version,memory.total,power.limit',
        '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
    lines = text.splitlines()
    if len(lines) != 1:
        raise RuntimeError('The scene queue requires exactly one GPU')
    fields = [v.strip() for v in lines[0].split(',')]
    if len(fields) != 5 or 'RTX 4090' not in fields[1]:
        raise RuntimeError('An RTX 4090 is required')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
        '--format=csv,noheader'], text=True, timeout=15).strip()
    if active and 'No running processes found' not in active:
        raise RuntimeError('GPU occupied; refusing overlapping workloads: '+active)
    return dict(zip(('uuid', 'name', 'driver', 'memory_mib', 'power_limit_w'), fields))


def validate_identity(identity, node, hardware, environment):
    expected = dict(node_id=node['node_id'], instance_id=node['instance_id'], gpu_uuid=hardware['uuid'])
    if any(identity.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Deployment node, instance or GPU does not match the scene owner')
    for key in ('CONTAINER_ID', 'VAST_CONTAINER_ID'):
        value = environment.get(key)
        if value and value.isdigit() and int(value) != node['instance_id']:
            raise RuntimeError('Vast instance ID differs from the frozen plan')


def preflight(root, plan, node):
    if Path(__file__).resolve() != (root/'code/tools/ray_allocation_261108/worker.py').resolve():
        raise RuntimeError('Execute the frozen worker from the campaign code directory')
    if not plan.get('frozen_commit') or not plan.get('code_files'):
        raise RuntimeError('The campaign has not been committed and frozen')
    ready = read(root/'.environment_ready.json')
    expected = dict(passed=True, decoder_verified=True, frozen_commit=plan['frozen_commit'],
                    plan_signature=plan['plan_signature'], node_id=node['node_id'],
                    instance_id=node['instance_id'])
    if any(ready.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Environment readiness does not match the frozen scene queue')
    if not {r['scene'] for r in node['runs']}.issubset(set(ready.get('datasets_verified', []))):
        raise RuntimeError('Assigned datasets have not been verified')
    verify_code(root, plan)
    for run in node['runs']:
        if read(root/'fleet'/run['config_path']) != run['method']:
            raise RuntimeError('Config content differs from the frozen run')
        if run['experiment_id'] not in ('E1', 'E2') or run['scene'] != node['scene']:
            raise RuntimeError('Run is not assigned to this scene owner')
        if not (root/'data'/run['scene']/'sparse/0/images.bin').is_file():
            raise RuntimeError('Scene data is absent')
    if os.environ.get('GRT_GRAD_ACCUM_DOUBLE') != '1':
        raise RuntimeError('GRT_GRAD_ACCUM_DOUBLE=1 is required')
    hardware = gpu_inventory()
    validate_identity(read(root/'.deployment_identity.json'), node, hardware, os.environ)
    return hardware


def gate_ready(root, plan, node, run):
    if run['run_id'] == plan['first_run_id'] and node['node_id'] == plan['first_node_id']:
        return True
    path = root/'.first_2k_gate.json'
    if not path.is_file():
        return False
    ack = read(path)
    expected = dict(plan_signature=plan['plan_signature'], first_run_id=plan['first_run_id'],
                    checkpoint_verified=True, step=2000)
    if any(ack.get(k) != v for k, v in expected.items()):
        raise RuntimeError('First 2k gate belongs to a different plan or is not verified')
    return True


def memory_sample():
    raw = subprocess.check_output(['nvidia-smi',
        '--query-gpu=uuid,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=5).strip().splitlines()
    if len(raw) != 1:
        raise RuntimeError('Expected one GPU while recording VRAM')
    uuid, used, total, utilization = [v.strip() for v in raw[0].split(',')]
    return dict(timestamp_utc=now(), gpu_uuid=uuid, used_mib=float(used),
                total_mib=float(total), utilization_percent=float(utilization))


def execute(command, log, memory_log, code, env, phase, hardware):
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists() or memory_log.exists():
        raise FileExistsError('Execution evidence already exists; refusing automatic retry')
    samples, monitor_errors = [], []
    started = time.monotonic()
    with log.open('x', encoding='utf-8') as output:
        child = subprocess.Popen(command, cwd=code, env=env, stdout=output,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        def stop(signum, _frame):
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass
            raise KeyboardInterrupt(f'Worker received signal {signum}')
        previous = {sig: signal.signal(sig, stop)
                    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        code_number = None
        try:
            while True:
                try:
                    sample = memory_sample()
                    if sample['gpu_uuid'] != hardware['uuid']:
                        raise RuntimeError('GPU UUID changed during execution')
                    samples.append(sample)
                except (subprocess.SubprocessError, ValueError) as exc:
                    monitor_errors.append(dict(timestamp_utc=now(), error=str(exc)))
                try:
                    code_number = child.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            write(memory_log, dict(phase=phase, gpu_uuid=hardware['uuid'],
                query_interval_seconds=2, measurement='nvidia-smi device-used VRAM; includes CUDA/OptiX allocations',
                peak_used_mib=max((s['used_mib'] for s in samples), default=None),
                samples=samples, monitor_errors=monitor_errors,
                process_exit_code=child.returncode, wall_seconds=time.monotonic()-started))
        if code_number:
            raise subprocess.CalledProcessError(code_number, command)
        if not samples:
            raise RuntimeError('No GPU VRAM samples were captured')


def verify_timing(out, run, completed):
    with (out/'step_times.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['step', 'mean_step_seconds_100', 'G']:
            raise RuntimeError('100-step timing schema differs')
        rows = list(reader)
    if [int(r['step']) for r in rows] != list(range(100, 30001, 100)):
        raise RuntimeError('100-step timing rows are missing or duplicated')
    seconds = 0.
    for row in rows:
        step, duration = int(row['step']), float(row['mean_step_seconds_100'])
        if not math.isfinite(duration) or duration < 0:
            raise RuntimeError('Invalid step duration')
        if int(row['G']) != count(run['initial_N'], run['method']['cap'], step):
            raise RuntimeError('Timing log Gaussian count differs from native growth')
        seconds += 100*duration
    if not math.isclose(seconds, completed.get('training_seconds', -1), rel_tol=1e-9, abs_tol=1e-6):
        raise RuntimeError('Pure training time differs from the 100-step timing sum')


def verify_training(out, run, plan):
    completed = read(out/'training_complete.json')
    expected = dict(passed=True, steps=30000, N=run['expected_G']['30000'], from_scratch=True,
        final_optimizer_included=True, phase='training_complete', evaluation_seconds=0,
        evaluation_steps=STEPS, cumulative_rays=run['total_rays'])
    if any(completed.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Training completion, ray count or evaluation ordering differs')
    protocol = read(out/'protocol.json')
    expected_protocol = dict(method=run['method'], train_views=run['train_views'],
        downsample=run['train_ds'], train_downsample=run['train_ds'], eval_downsample=run['eval_ds'],
        growth='native_5percent', growth_end=None, structural_loss=run['loss'],
        execution_protocol='allocation_resolution', checkpoint_steps=STEPS,
        evaluation_phase='after_training_external_eval_full', observed_only=False,
        multiview_start_step=0, allocation_arm=run['allocation_arm'], importance_loss_reweighting=False)
    if any(protocol.get(k) != v for k, v in expected_protocol.items()):
        raise RuntimeError('Actual protocol differs from the frozen run')
    resolved = read(out/'resolved_config.json')
    add = resolved['strategy']['add']
    expected_add = dict(start_iteration=500, end_iteration=25000, frequency=100,
                        placement='opacity', max_n_gaussians=run['method']['cap'])
    if any(add.get(k) != v for k, v in expected_add.items()):
        raise RuntimeError('MCMC growth recipe changed')
    if resolved['render'] != plan['renderer_settings']:
        raise RuntimeError('Renderer differs from the frozen plan')
    native = protocol.get('native_add_schedule', {})
    if native.get('original_add_method_preserved') is not True:
        raise RuntimeError('Native MCMC add operation was replaced')
    if read(out/'initial.json')['N'] != run['initial_N']:
        raise RuntimeError('Initial SfM point count differs')
    geometry = read(out/'ray_geometry.json')
    if len(geometry) != run['train_views']:
        raise RuntimeError('Training-view geometry coverage differs')
    for item in geometry.values():
        if [item['width'], item['height']] != run['train_shape']:
            raise RuntimeError('Training image shape differs from the manifest')
    for step in STEPS:
        checkpoint = out/f'ckpt_{step}.pt'
        proof = read(out/f'checkpoint_verified_{step}.json')
        expected_proof = dict(step=step, load_verified=True, optimizer_present=True,
                              optimizer_moment_shapes_verified=True, G=run['expected_G'][str(step)])
        if (not checkpoint.is_file() or checkpoint.stat().st_size <= 0 or
                any(proof.get(k) != v for k, v in expected_proof.items()) or
                proof.get('bytes') != checkpoint.stat().st_size):
            raise RuntimeError(f'Checkpoint reload/optimizer/G verification failed: {step}')
        if not math.isfinite(proof.get('training_seconds', float('nan'))):
            raise RuntimeError('Checkpoint cumulative time is absent')
    verify_timing(out, run, completed)
    return completed


def verify_evaluation(out, run, step):
    value = read(out/f'evaluation_{step}.json')
    expected = dict(step=step, views=run['test_views'], run_id=run['run_id'],
        N=run['expected_G'][str(step)], optimizer_state_present=True,
        train_downsample=run['train_ds'], eval_downsample=run['eval_ds'])
    if any(value.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Evaluation identity, DS, step or Gaussian count differs')
    if value.get('checkpoint_sha256') != sha(out/f'ckpt_{step}.pt'):
        raise RuntimeError('Evaluation checkpoint SHA256 mismatch')
    definition = value.get('metric_definition', {})
    if (definition.get('source') != 'sparse_grt.standalone_backend.evaluate' or
        definition.get('ssim') != 'torchmetrics StructuralSimilarityIndexMeasure(data_range=1.0), defaults' or
        definition.get('lpips') != "torchmetrics LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)"):
        raise RuntimeError('Full-view evaluation metrics changed')
    views = value.get('per_view', [])
    if len(views) != run['test_views'] or len({v['view'] for v in views}) != len(views):
        raise RuntimeError('Missing or repeated heldout views')
    for view in views:
        if [view.get('width'), view.get('height')] != run['eval_shape']:
            raise RuntimeError('A test view was rendered at the wrong target resolution')
    for metric in ('PSNR', 'SSIM', 'LPIPS'):
        if (not math.isfinite(value.get(metric, float('nan'))) or
            any(not math.isfinite(v.get(metric, float('nan'))) for v in views) or
            not math.isclose(value[metric], sum(v[metric] for v in views)/len(views),
                             rel_tol=1e-12, abs_tol=1e-12)):
            raise RuntimeError('Invalid or inconsistent full-view metrics')
    return value


def verify_artifacts(root, receipt):
    for record in receipt['files']:
        path = root/record['path']
        if not path.is_file() or sha(path) != record['sha256']:
            raise RuntimeError('Previously archived evidence changed: '+str(path))


def artifact_records(root, paths):
    return [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(set(paths)) if p.is_file()]


def receipt(root, plan, node, run, hardware, completed):
    out = root/'runs'/run['run_id']
    evaluations = {s: verify_evaluation(out, run, s) for s in STEPS}
    paths = [p for p in out.rglob('*') if p.is_file() and p.name != 'checkpoint_latest.pt']
    paths += list((root/'memory').glob(run['run_id']+'.*.json'))
    paths += list((root/'logs').glob(run['run_id']+'.*.log'))
    paths += [root/'run_manifests'/f"{run['run_id']}.json"]
    return dict(passed=True, run_id=run['run_id'], run_signature=run_signature(plan, run),
        plan_signature=plan['plan_signature'], frozen_commit=plan['frozen_commit'], hardware=hardware,
        node_id=node['node_id'], instance_id=node['instance_id'], training=completed,
        final_metrics={k: evaluations[30000][k] for k in ('PSNR', 'SSIM', 'LPIPS')},
        files=artifact_records(root, paths), complete_utc=now())


def cuda_oom_record(root, plan, run, hardware, phase, log, memory_log, evaluation_step=None):
    with log.open('rb') as handle:
        handle.seek(max(0, log.stat().st_size-128*1024))
        tail = handle.read().decode('utf-8', errors='replace')
    if not re.search(r'CUDA out of memory|CUDA_ERROR_OUT_OF_MEMORY|OPTIX_ERROR_MEMORY_ALLOCATION_FAILED', tail):
        raise RuntimeError('Non-CUDA-OOM failure; stop the queue for investigation')
    out = root/'runs'/run['run_id']
    structured = (out/'failure.json').is_file()
    if phase == 'training' and not structured:
        # CUDA can fail during model/dataset construction before the trainer's
        # guarded update loop exists. Preserve unknowns; do not invent step=0
        # or G=initial_N. Explicit CUDA traceback + measured VRAM is sufficient
        # to terminate this arm, never to retry or modify its settings.
        state = dict(passed=False, error_kind='cuda_out_of_memory', stage='init_unknown',
                     step=None, G=None, training_seconds=None,
                     state_evidence='No structured trainer failure; exact step/G unavailable')
    else:
        state = read(out/'failure.json') if phase == 'training' else read(out/'training_complete.json')
    if phase == 'training' and (state.get('error_kind') != 'cuda_out_of_memory' or state.get('passed') is not False):
        raise RuntimeError('CUDA OOM lacks structured training failure evidence')
    if phase == 'evaluation':
        if state.get('passed') is not True:
            raise RuntimeError('Evaluation OOM lacks a completed training record')
        state = dict(state, trained_steps=state['steps'], step=evaluation_step,
                     G=run['expected_G'][str(evaluation_step)])
    step, gaussians = state.get('step'), state.get('G', state.get('N'))
    unknown_state = state.get('stage') == 'init_unknown' and step is None and gaussians is None
    if not unknown_state and (type(step) is not int or not 0 <= step <= 30000 or type(gaussians) is not int or
            not run['initial_N'] <= gaussians <= run['method']['cap']):
        raise RuntimeError('OOM step/G is missing or inconsistent')
    memory = read(memory_log)
    paths = [log, memory_log, root/'run_manifests'/f"{run['run_id']}.json"]
    paths += [p for p in (out/'failure.json', out/'training_complete.json', out/'progress.json') if p.is_file()]
    return dict(passed=False, terminal=True, status='cuda_oom', phase=phase,
        run_id=run['run_id'], run_signature=run_signature(plan, run),
        plan_signature=plan['plan_signature'], frozen_commit=plan['frozen_commit'],
        step=step, attempted_step=state.get('attempted_step'), G=gaussians,
        stage=state.get('stage', phase),
        evaluation_step=evaluation_step, training_seconds=state.get('training_seconds'),
        peak_device_used_mib=memory['peak_used_mib'], hardware=hardware, failure=state,
        files=artifact_records(root, paths), recorded_utc=now())


def required_queue_disk_bytes(runs):
    sizes = [math.ceil(1.2*712*r['expected_G'][str(s)]) for r in runs for s in STEPS]
    return sum(sizes)+2*max(sizes)+4*2**30 if sizes else 0


def run_queue(args):
    root = args.root.resolve()
    plan = load_plan(root/'fleet/plan.json')
    node = selected_node(plan, args.node)
    hardware = preflight(root, plan, node)
    write(root/'hardware.json', dict(**hardware, node_id=node['node_id'], instance_id=node['instance_id'],
                                     recorded_utc=now()))
    env = dict(os.environ, PYTHONPATH=str(root/'code/src'), PYTHONUNBUFFERED='1',
               SPARSE_GRT_FROZEN_COMMIT=plan['frozen_commit'])
    remaining = [r for r in node['runs'] if not (root/'runs'/r['run_id']).exists()]
    required = required_queue_disk_bytes(remaining)
    if shutil.disk_usage(root).free < required:
        raise RuntimeError(f'Not enough disk to retain all checkpoints: need {required} bytes')
    done, failed = [], []
    for run in node['runs']:
        rid = run['run_id']; out = root/'runs'/rid
        success = root/'receipts'/f'{rid}.json'; failure = root/'failure_receipts'/f'{rid}.json'
        if success.exists() and failure.exists():
            raise RuntimeError('Both success and terminal failure receipts exist')
        existing_path = success if success.exists() else failure if failure.exists() else None
        if existing_path:
            existing = read(existing_path)
            if existing.get('run_signature') != run_signature(plan, run):
                raise RuntimeError('Existing receipt belongs to another frozen run')
            if success.exists() and existing.get('passed') is not True:
                raise RuntimeError('Invalid success receipt')
            if failure.exists() and (existing.get('status') != 'cuda_oom' or existing.get('terminal') is not True):
                raise RuntimeError('Invalid terminal failure receipt')
            verify_artifacts(root, existing)
            (done if success.exists() else failed).append(rid)
            continue
        while not gate_ready(root, plan, node, run):
            write(root/'status.json', dict(phase='waiting_first_2k_gate', run_id=rid,
                node_id=args.node, first_run_id=plan['first_run_id'], updated_utc=now()))
            time.sleep(15)
        verify_code(root, plan)
        if gpu_inventory() != hardware:
            raise RuntimeError('GPU, driver or power limit changed within the scene queue')
        manifest = root/'run_manifests'/f'{rid}.json'
        if out.exists():
            if not (out/'training_complete.json').is_file() or not manifest.is_file():
                raise RuntimeError('Partial training exists; no automatic restart is permitted')
            evidence = read(manifest)
            if evidence.get('run_signature') != run_signature(plan, run) or evidence.get('hardware') != hardware:
                raise RuntimeError('Evaluation continuation belongs to another run or hardware')
            completed = verify_training(out, run, plan)
        else:
            if manifest.exists():
                raise RuntimeError('An earlier launch was attempted; no automatic duplicate training')
            write(manifest, dict(run=run, run_signature=run_signature(plan, run),
                frozen_commit=plan['frozen_commit'], plan_signature=plan['plan_signature'], hardware=hardware,
                node_id=args.node, instance_id=node['instance_id'], launch_utc=now(),
                free_disk_bytes=shutil.disk_usage(root).free))
            write(root/'status.json', dict(phase='training', run_id=rid, completed=done,
                                          failed=failed, node_id=args.node, updated_utc=now()))
            log, memory = root/'logs'/f'{rid}.train.log', root/'memory'/f'{rid}.train.json'
            try:
                execute(train_command(root, run, plan['python']), log, memory, root/'code', env, 'training', hardware)
            except subprocess.CalledProcessError:
                record = cuda_oom_record(root, plan, run, hardware, 'training', log, memory)
                write(failure, record); failed.append(rid)
                write(root/'status.json', dict(phase='arm_cuda_oom', run_id=rid, node_id=args.node,
                    completed=done, failed=failed, step=record['step'], G=record['G'], updated_utc=now()))
                continue
            verify_code(root, plan)
            completed = verify_training(out, run, plan)
        write(root/'status.json', dict(phase='evaluation', run_id=rid, completed=done,
                                      failed=failed, node_id=args.node, updated_utc=now()))
        evaluation_failed = False
        for step in STEPS:
            if (out/f'evaluation_{step}.json').is_file():
                verify_evaluation(out, run, step)
                continue
            log, memory = root/'logs'/f'{rid}.eval{step}.log', root/'memory'/f'{rid}.eval{step}.json'
            try:
                execute(eval_command(root, run, step, plan['python']), log, memory, root/'code', env,
                        f'evaluation_{step}', hardware)
            except subprocess.CalledProcessError:
                record = cuda_oom_record(root, plan, run, hardware, 'evaluation', log, memory, step)
                write(failure, record); failed.append(rid); evaluation_failed = True
                break
            verify_evaluation(out, run, step)
        if evaluation_failed:
            continue
        verify_code(root, plan)
        write(success, receipt(root, plan, node, run, hardware, completed))
        done.append(rid)
        write(root/'status.json', dict(phase='run_complete', run_id=rid, completed=done,
                                      failed=failed, node_id=args.node, updated_utc=now()))
    write(root/'status.json', dict(phase='complete_with_failures' if failed else 'complete',
        completed=done, failed=failed, node_id=args.node, updated_utc=now(),
        checkpoint_policy='preserve_all', instance_policy='do_not_destroy'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    def stop_worker(signum, _frame):
        raise KeyboardInterrupt(f'Worker received signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop_worker)
    with machine_lock(args.root.resolve()):
        try:
            if args.check_only:
                plan = load_plan(args.root/'fleet/plan.json')
                print(json.dumps(preflight(args.root.resolve(), plan, selected_node(plan, args.node))))
            else:
                run_queue(args)
        except BaseException as error:
            write(args.root/'status.json', dict(phase='failed', node_id=args.node, updated_utc=now(),
                                               error=f'{type(error).__name__}: {error}'))
            raise


if __name__ == '__main__':
    main()
