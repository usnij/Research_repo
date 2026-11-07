"""Run a frozen scene queue on one GPU; wait for verified backup between jobs."""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

from benchmark_common import read, sha, signature, verify_files, verify_complete, wait_for_ack, write


def stamp():
    return datetime.now(timezone.utc).isoformat()


def execute(cmd, log):
    with Path(log).open('w') as f:
        subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, check=True)


def verify_run_protocol(run, audit, cfg):
    p, b = read(run/'protocol.json'), read(run/'budget.json')
    if p['method'] != cfg or p['train_views'] != audit['train_views']:
        raise ValueError('Actual method or split differs from frozen scene specification')
    if b['cap'] != audit['target_N'] or b['test_RGB_used']:
        raise ValueError('Input-derived cap differs or test RGB was used')
    if p['budget_method'] != 'fine_tracks_v1' or p['extra_training_rays'] != 0:
        raise ValueError('Wrong benchmark path')
    if p['growth'] != 'log_horizon' or p['growth_end'] != 5000:
        raise ValueError('Growth schedule differs')
    if read(run/'initial.json')['N'] != audit['initial_N']:
        raise ValueError('Initial point count differs')
    if {v['total_rays_if_equal_views'] for v in read(run/'ray_geometry.json').values()} != {audit['rays_per_update']}:
        raise ValueError('Actual ray budget differs')


def unpack_scene(root, name, audit):
    data = root/'data'/name
    files = read(root/'payload/scenes'/name/'files.json')
    if not data.exists():
        archive = root/audit['archive']['path']
        if sha(archive) != audit['archive']['sha256']:
            raise ValueError(f'Archive checksum mismatch: {name}')
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            expected = {f"{name}/{r['path']}" for r in files}
            if {m.name for m in members} != expected or any(not m.isfile() for m in members):
                raise ValueError('Unexpected archive contents')
            for m in members:
                path = Path(m.name)
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError('Unsafe archive member')
            tar.extractall(root/'data', members=members, filter='data')
    verify_files(data, files)
    return data


def training_command(root, scene, audit, arm, mode, out):
    return [sys.executable, '-u', '-m', 'sparse_grt.training',
            '--scene', str(root/'data'/scene), '--downsample', str(audit['downsample']),
            '--config', str(root/'payload/scenes'/scene/f'{arm}.json'),
            '--output', str(out), '--budget', 'fine_tracks_v1', '--budget-views', 'all',
            '--growth', 'log_horizon', '--growth-end', '5000',
            '--cache-mib', '1200', '--keep-checkpoints', '7000,30000', '--mode', mode]


def artifact(root, run, report):
    paths = sorted(p for p in run.rglob('*') if p.is_file() and
                   (p.suffix in ('.json', '.jsonl') or p.name in ('ckpt_7000.pt', 'ckpt_30000.pt')))
    if not all((run/f'ckpt_{step}.pt').is_file() for step in (7000, 30000)):
        raise ValueError('Required retained checkpoint is missing')
    records = [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=sha(p)) for p in paths]
    body = dict(run_id=run.name, files=records, report=report, completed_utc=stamp())
    return body | {'artifact_signature': signature(body)}


def work(root):
    node = read(root/'node.json')
    code = root/'code'
    logs = root/'logs'; logs.mkdir(exist_ok=True)
    frozen = read(root/'payload/frozen_files.json')
    verify_files(root, frozen)
    import sparse_grt
    from PIL import features
    if features.version('jpg') != '9.0' or features.check('libjpeg_turbo'):
        raise ValueError('The reference IJG JPEG decoder is required')
    installed = Path(sparse_grt.__file__).resolve().parent.parent
    if 'site-packages' not in installed.parts:
        raise ValueError('Training must import the installed package')
    for record in frozen:
        if record['path'].startswith('code/src/'):
            p = installed/record['path'][len('code/src/'):]
            if not p.is_file() or sha(p) != record['sha256']:
                raise ValueError(f'Installed package does not match frozen source: {p}')
    for package, version in read(root/'payload/dependencies.json').items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'Dependency mismatch: {package}, expected {version}')
    if os.environ.get('GRT_GRAD_ACCUM_DOUBLE') != '1':
        raise ValueError('The reference FP64 gradient accumulation must be enabled')
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:
        raise RuntimeError(f'GPU already has compute processes: {active}')
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,power.limit',
                                        '--format=csv,noheader'],text=True)
    if 'RTX 4090' not in hardware or len(hardware.strip().splitlines()) != 1:
        raise ValueError('This benchmark deployment expects exactly one RTX 4090')
    write(root/'hardware.json', dict(gpu=hardware, started_utc=stamp(), node=node['name'],
                                     python=sys.version, training_commit=read(root/'payload/spec.json')['training_commit']))
    write(root/'status.json', dict(phase='preflight', node=node['name'], updated_utc=stamp()))
    execute([sys.executable, '-m', 'unittest', 'discover', '-s', code/'tests', '-v'], logs/'cpu.log')
    execute([sys.executable, code/'tools/validate_paper_cuda.py', '--output', root/'loss_cuda.json'], logs/'cuda.log')
    # Download evaluation weights before any timed training process starts.
    execute([sys.executable, '-c', 'from torchvision.models import VGG16_Weights; VGG16_Weights.IMAGENET1K_V1.get_state_dict(progress=False, check_hash=True)'], logs/'evaluation_weights.log')
    completed = []
    for name in node['scenes']:
        audit = read(root/'payload/scenes'/name/'audit.json')
        unpack_scene(root, name, audit)
        for arm in node.get('arms', ['structure']):
            if arm not in ('structure', 'l1'):
                raise ValueError(f'Unknown arm: {arm}')
            run_id = f'{name}_{arm}_seed777'
            run = root/'runs'/run_id
            config = read(root/'payload/scenes'/name/f'{arm}.json')
            smoke = root/'smoke'/run_id
            receipt = root/'artifacts'/f'{run_id}.json'
            ack = root/'acks'/f'{run_id}.json'
            if receipt.exists():
                verify_complete(run)
                verify_run_protocol(run, audit, config)
                verify_files(root, read(receipt)['files'])
                wait_for_ack(ack, read(receipt)['artifact_signature'])
                completed.append(run_id)
                continue
            write(root/'status.json', dict(phase='smoke', scene=name, arm=arm, completed=completed, updated_utc=stamp()))
            if smoke.exists():
                if not (smoke/'complete.json').exists() or not read(smoke/'complete.json')['passed']:
                    raise RuntimeError('Incomplete smoke directory; retain it and investigate before restarting')
            else:
                execute(training_command(root, name, audit, arm, 'smoke', smoke), logs/f'{run_id}_smoke.log')
            verify_run_protocol(smoke, audit, config)
            if node.get('smoke_only', False):
                completed.append(run_id)
                continue
            if run.exists():
                raise RuntimeError(f'Existing run has no archive receipt: {run}; no silent restart or overwrite')
            if shutil.disk_usage(root).free < 12*1024**3 + 2200*audit['target_N']:
                raise RuntimeError('Insufficient checkpoint storage before starting scene')
            write(root/'status.json', dict(phase='training', scene=name, arm=arm, run_id=run_id,
                  completed=completed, started_utc=stamp(), updated_utc=stamp()))
            start = time.monotonic()
            execute(training_command(root, name, audit, arm, 'train', run), logs/f'{run_id}.log')
            report = verify_complete(run)
            verify_run_protocol(run, audit, config)
            if report['metrics']['30000']['views'] != audit['test_views']:
                raise ValueError('Final test-view coverage differs from frozen split')
            peer = root/'runs'/f"{name}_{'l1' if arm=='structure' else 'structure'}_seed777"
            if (peer/'complete.json').exists():
                for filename in ('initial.json', 'training_manifest.json', 'growth_plan.json'):
                    if read(peer/filename) != read(run/filename):
                        raise ValueError(f'Paired comparison differs: {filename}')
                if read(peer/'first_input.json')['hash'] != read(run/'first_input.json')['hash']:
                    raise ValueError('Paired input rays/RGB differ')
            report.update(scene=name, arm=arm, dataset=audit['dataset'], node=node['name'],
                          wall_seconds=time.monotonic()-start, input_preparation_seconds=read(run/'budget.json')['seconds'],
                          timing_note='Training excludes evaluation/save/setup; wall time includes child initialization and all work.')
            write(run/'benchmark_result.json', report)
            record = artifact(root, run, report)
            write(receipt, record)
            write(root/'status.json', dict(phase='awaiting_backup', run_id=run_id, completed=completed, updated_utc=stamp()))
            wait_for_ack(ack, record['artifact_signature'])
            completed.append(run_id)
    write(root/'status.json', dict(phase='smoke_complete' if node.get('smoke_only') else 'complete',
                                  completed=completed, updated_utc=stamp()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args(); root = a.root.resolve()
    # A per-machine lock also prevents two different benchmark roots using this GPU.
    lock = (root.parent/'.sparse-grt-gpu0.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        work(root)
    except Exception as e:
        write(root/'status.json', dict(phase='failed', error=f'{type(e).__name__}: {e}', updated_utc=stamp()))
        raise


if __name__ == '__main__':
    main()
