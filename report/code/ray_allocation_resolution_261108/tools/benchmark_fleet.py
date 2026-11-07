"""Freeze, deploy, collect and summarize a one-GPU-per-node benchmark fleet.

SSH keys and Drive OAuth remain on the operator's computer. This tool never rents,
destroys, or silently restarts a failed training run.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time

from benchmark_common import read, relative_file, sha, signature, verify_files, write


def ssh(node):
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4',
            '-o', 'StrictHostKeyChecking=accept-new', '-p', str(node['port']),
            f"{node.get('user','root')}@{node['host']}"]


def remote(node, script, payload=None):
    code = ('payload = '+repr(payload)+'\n' if payload is not None else '') + script
    p = subprocess.run(ssh(node)+['python3 -'], input=code, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return p.stdout


def rsync(node, source, destination, *, pull=False, files=None):
    # SSH transport uses the same options, without the final destination argument.
    transport = shlex.join(ssh(node)[:-1])
    host = ssh(node)[-1]
    args = ['rsync', '-a', '--partial', '--append-verify', '--protect-args', '-e', transport]
    if files is not None:
        args += ['--files-from', str(files)]
    if pull:
        args += [f'{host}:{source}', str(destination)]
    else:
        args += [str(source), f'{host}:{destination}']
    subprocess.run(args, check=True)


def freeze(code, prepared, output):
    if output.exists():
        raise FileExistsError(output)
    if not read(prepared/'ready.json')['passed']:
        raise ValueError('Dataset preparation is not complete')
    output.mkdir(parents=True)
    source = output/'release'
    names = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],
                                    cwd=code).decode().split('\0')
    for name in filter(None, names):
        p = relative_file(code, name)
        if not p.is_file():
            continue
        q = source/'code'/name; q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p,q)
    payload = source/'payload'; payload.mkdir()
    for name in ('spec.json', 'scene_index.json', 'allocation.json', 'ready.json'):
        shutil.copy2(prepared/name, payload/name)
    shutil.copytree(prepared/'scenes', payload/'scenes')
    dependencies = {}
    for line in (code/'releases/standalone_remote_20261102/requirements-installed.txt').read_text().splitlines():
        if '==' in line:
            name, version = line.split('==',1); dependencies[name] = version
    write(payload/'dependencies.json', dependencies)
    (payload/'constraints.txt').write_text(''.join(f'{k}=={v}\n' for k,v in sorted(dependencies.items())))
    files = [dict(path=str(p.relative_to(source)),bytes=p.stat().st_size,sha256=sha(p))
             for p in sorted(source.rglob('*')) if p.is_file()]
    write(payload/'frozen_files.json', files)
    archive = output/'release.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for p in sorted(source.rglob('*')):
            if p.is_file():
                tar.add(p, arcname=str(p.relative_to(source)), recursive=False)
    write(output/'delivery.json', dict(archive_sha256=sha(archive),
          source_signature=signature(files), git_commit=subprocess.check_output(
          ['git','rev-parse','HEAD'],cwd=code,text=True).strip(),
          prepared=str(prepared.resolve()), release=str(output.resolve())))


def nodes_from(fleet, selected=None):
    nodes = fleet['nodes']
    if len({n['name'] for n in nodes}) != len(nodes):
        raise ValueError('Duplicate node names')
    if len({(n['host'],n['port']) for n in nodes}) != len(nodes):
        raise ValueError('Same endpoint assigned twice')
    for n in nodes:
        if not re.fullmatch(r'[a-zA-Z0-9_.-]+',n['host']) or not re.fullmatch(r'[a-zA-Z0-9_-]+',n['name']):
            raise ValueError('Invalid host or node name')
        if not 1 <= int(n['port']) <= 65535 or n.get('user','root') != 'root':
            raise ValueError('Expected a root SSH endpoint with a valid port')
        if not re.fullmatch(r'/workspace/[a-zA-Z0-9_/-]+', n['root']):
            raise ValueError('Use a simple absolute /workspace deployment path')
        if not n['scenes'] or len(n['scenes']) != len(set(n['scenes'])):
            raise ValueError('Empty or duplicate scene assignment')
    jobs = [(s,a) for n in nodes for s in n['scenes'] for a in n.get('arms',['structure'])]
    if len(jobs) != len(set(jobs)):
        raise ValueError('A scene/arm/seed was assigned to more than one node')
    if selected:
        wanted = set(selected.split(','))
        if not wanted <= {n['name'] for n in nodes}:
            raise ValueError('Unknown node selection')
        nodes = [n for n in nodes if n['name'] in wanted]
    return nodes


BOOTSTRAP = r'''#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=__ROOT__
cd "$ROOT"
trap 'python3 -c "import json,pathlib; pathlib.Path(\"$ROOT/bootstrap_failed.json\").write_text(json.dumps({\"failed\":True}))"' ERR
if [[ ! -f "$ROOT/environment.ready" ]]; then
  EXISTING=__EXISTING__
  if [[ -n "$EXISTING" ]]; then
    [[ -f "$EXISTING" ]]
    printf 'source %q\n' "$EXISTING" > "$ROOT/activate.sh"
  else
    apt-get update
    apt-get install -y --no-install-recommends gcc-11 g++-11 make curl git zlib1g-dev rsync
    export PIP_CONSTRAINT="$ROOT/payload/constraints.txt"
    MAMBA_EXE=${MAMBA_EXE:-/opt/miniforge3/bin/mamba} \
      bash "$ROOT/code/tools/setup_reference_env.sh" "$ROOT/env"
    printf 'source %q\n' "$ROOT/env/activate.sh" > "$ROOT/activate.sh"
  fi
  touch "$ROOT/environment.ready"
fi
source "$ROOT/activate.sh"
while [[ ! -f "$ROOT/uploads_ready.json" ]]; do sleep 3; done
exec python -u "$ROOT/code/tools/benchmark_worker.py" --root "$ROOT"
'''


def deploy(node, delivery):
    prepared, release = Path(delivery['prepared']), Path(delivery['release'])
    index = {s['scene']:s for s in read(prepared/'scene_index.json')}
    if not set(node['scenes']) <= index.keys():
        raise ValueError('Scene is missing from prepared inputs')
    root = node['root']
    node_config = {k:node[k] for k in ('name','scenes','arms','smoke_only') if k in node}
    remote(node, "from pathlib import Path\np=Path(payload); p.mkdir(parents=True,exist_ok=True)\n", root)
    rsync(node, release/'release.tar.gz', root+'/release.tar.gz')
    bootstrap = BOOTSTRAP.replace('__ROOT__',shlex.quote(root)).replace(
        '__EXISTING__',shlex.quote(node.get('existing_activation','')))
    program = 'sparse-grt-'+node['name']
    conf = (f'[program:{program}]\ncommand=/bin/bash {root}/bootstrap.sh\ndirectory={root}\n'
            'autostart=false\nautorestart=false\nstartsecs=2\nstopasgroup=true\nkillasgroup=true\n'
            f'redirect_stderr=true\nstdout_logfile={root}/supervisor.log\n'
            'stdout_logfile_maxbytes=10MB\nstdout_logfile_backups=2\n')
    remote(node, r'''
from pathlib import Path
import json,hashlib,tarfile,subprocess
r=Path(payload['root']); expected=payload['config']; config=r/'node.json'
if config.exists() and json.loads(config.read_text()) != expected: raise ValueError('Refuse changed node assignment')
if hashlib.sha256((r/'release.tar.gz').read_bytes()).hexdigest()!=payload['sha']: raise ValueError('Source archive mismatch')
marker=r/'source.sha256'
if marker.exists() and marker.read_text()!=payload['sha']: raise ValueError('Refuse changed release in an existing deployment')
if not marker.exists():
 with tarfile.open(r/'release.tar.gz') as tar:
  for m in tar.getmembers():
   if not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts: raise ValueError('Unsafe source member')
  tar.extractall(r,filter='data')
 marker.write_text(payload['sha'])
config.write_text(json.dumps(expected,indent=2)+'\n')
(r/'bootstrap.sh').write_text(payload['bootstrap'])
conf=Path('/etc/supervisor/conf.d')/(payload['program']+'.conf')
if conf.exists() and conf.read_text()!=payload['conf']: raise ValueError('Supervisor program already used')
conf.write_text(payload['conf'])
subprocess.run(['supervisorctl','reread'],check=True,stdout=subprocess.DEVNULL)
subprocess.run(['supervisorctl','update',payload['program']],check=True,stdout=subprocess.DEVNULL)
state=subprocess.run(['supervisorctl','status',payload['program']],capture_output=True,text=True).stdout
if not any(x in state for x in ('RUNNING','STARTING')):
 if (r/'status.json').exists() or (r/'bootstrap_failed.json').exists(): raise ValueError('Inspect previous result before manually restarting')
 subprocess.run(['supervisorctl','start',payload['program']],check=True,stdout=subprocess.DEVNULL)
print('bootstrap started')
''', dict(root=root,config=node_config,sha=delivery['archive_sha256'],bootstrap=bootstrap,conf=conf,program=program))
    remote(node, "from pathlib import Path\nPath(payload).mkdir(exist_ok=True)\n",root+'/assets')
    for name in node['scenes']:
        rsync(node, prepared/index[name]['archive']['path'], root+'/assets/')
    remote(node, "from pathlib import Path\nimport json\np=Path(payload)/'uploads_ready.json'; t=p.with_suffix('.tmp'); t.write_text(json.dumps({'ready':True})); t.replace(p)\n",root)
    print(node['name'], 'deployed', ','.join(node['scenes']), flush=True)


def status(node):
    raw = remote(node, r'''
from pathlib import Path
import json
r=Path(payload); d={}
if (r/'status.json').exists(): d=json.loads((r/'status.json').read_text())
elif (r/'bootstrap_failed.json').exists(): d={'phase':'failed','error':'bootstrap failed; inspect supervisor.log'}
else: d={'phase':'installing_or_transferring'}
if d.get('run_id'):
 p=r/'runs'/d['run_id']/'progress.json'
 if p.exists(): d['progress']=json.loads(p.read_text())
d['artifacts']=[json.loads(p.read_text()) for p in sorted((r/'artifacts').glob('*.json'))]
print(json.dumps(d))
''',node['root'])
    return json.loads(raw)


def archive_one(node, record, fleet, local):
    target = local/node['name']; target.mkdir(parents=True,exist_ok=True)
    receipt = target/'receipts'/f"{record['run_id']}.json"
    if receipt.exists():
        previous = read(receipt)
        if previous['artifact_signature'] != record['artifact_signature']:
            raise ValueError('Archive identity changed')
    else:
        need = sum(f['bytes'] for f in record['files'])
        if shutil.disk_usage(target).free < need+5*1024**3:
            raise RuntimeError('Not enough local space to safely stage checkpoints')
        for f in record['files']:
            relative_file(target,f['path'])
        with tempfile.NamedTemporaryFile(mode='w',dir=target,delete=False) as f:
            f.write('\n'.join(r['path'] for r in record['files'])+'\n'); listing=Path(f.name)
        try:
            rsync(node, node['root']+'/', target, pull=True, files=listing)
        finally:
            listing.unlink(missing_ok=True)
        verify_files(target,record['files'])
        drive = fleet.get('drive')
        if drive:
            remote_path=drive['destination'].rstrip('/')+'/'+node['name']+'/'+record['run_id']
            flags=['--drive-root-folder-id',drive['root_id'],'--transfers','2','--checkers','2',
                   '--tpslimit','2','--retries','3','--low-level-retries','5']
            subprocess.run(['rclone','copy',str(target/'runs'/record['run_id']),remote_path,*flags],check=True)
            subprocess.run(['rclone','check',str(target/'runs'/record['run_id']),remote_path,
                            '--one-way',*flags],check=True)
        previous = dict(verified=True, artifact_signature=record['artifact_signature'],
                        run_id=record['run_id'], drive_verified=bool(drive),
                        report=record['report'], collected_utc=datetime.now(timezone.utc).isoformat())
        write(receipt,previous)
    remote(node,"from pathlib import Path\nimport json\np=Path(payload['root'])/'acks'/(payload['ack']['run_id']+'.json'); p.parent.mkdir(exist_ok=True); t=p.with_suffix('.tmp'); t.write_text(json.dumps(payload['ack'])); t.replace(p)\n",
           dict(root=node['root'],ack=previous))
    print(node['name'],record['run_id'],'backup verified',flush=True)


def summarize(local, scenes):
    rows = []
    for p in sorted(local.glob('*/receipts/*.json')):
        d=read(p)
        if d.get('verified'):
            rows.append(d['report'])
    write(local/'results.json',rows)
    lines=['# 최종 방법 벤치마크', '', '30,000스텝 완료·평가·보관 검증을 통과한 실행만 표시한다.', '',
           '| 씬 | 조건 | Gaussian | PSNR | SSIM | LPIPS | 학습(분) | 전체 실행(분) |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        m=r['metrics']['30000'];c=r['complete']
        lines.append(f"| {r['scene']} | {r['arm']} | {c['N']:,} | {m['PSNR']:.4f} | {m['SSIM']:.5f} | {m['LPIPS']:.5f} | {c['training_seconds']/60:.2f} | {r['wall_seconds']/60:.2f} |")
    lines += ['', '학습시간은 동기화한 update 합계다. 전체 실행시간은 초기화·입력 cap 산출·평가·저장을 포함하며 설치·전송·백업은 제외한다.', '',
              '전체 씬이 완료되기 전에는 데이터셋 전체 평균으로 표시하지 않는다.']
    (local/'results.md').write_text('\n'.join(lines)+'\n')
    averages=[]
    for dataset in sorted({s['dataset'] for s in scenes}):
        wanted={s['scene'] for s in scenes if s['dataset']==dataset}
        for arm in ('structure','l1'):
            group=[r for r in rows if r['dataset']==dataset and r['arm']==arm]
            if {r['scene'] for r in group}==wanted:
                averages.append(dict(dataset=dataset,arm=arm,scenes=len(wanted),
                    **{k:sum(r['metrics']['30000'][k] for r in group)/len(group) for k in ('PSNR','SSIM','LPIPS')}))
    write(local/'dataset_averages.json',averages)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    f=sub.add_parser('freeze'); f.add_argument('--code',type=Path,required=True)
    f.add_argument('--prepared',type=Path,required=True); f.add_argument('--output',type=Path,required=True)
    for action in ('deploy','status','watch','collect'):
        q=sub.add_parser(action);q.add_argument('--fleet',type=Path,required=True);q.add_argument('--nodes')
        q.add_argument('--workers',type=int,default=2)
        if action=='watch':q.add_argument('--interval',type=int,default=60)
    a=p.parse_args()
    if a.action=='freeze':
        freeze(a.code.resolve(),a.prepared.resolve(),a.output.resolve()); return
    fleet=read(a.fleet);nodes=nodes_from(fleet,a.nodes)
    delivery=read(Path(fleet['release'])/'delivery.json')
    local=Path(fleet['collected']);local.mkdir(parents=True,exist_ok=True)
    if a.action=='deploy':
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            list(pool.map(lambda n:deploy(n,delivery),nodes))
        return
    lock=(local/'.collector.lock').open('a')
    if a.action in ('watch','collect'):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        if a.action == 'watch':
            fresh = read(a.fleet)
            if fresh['release'] != fleet['release'] or fresh['collected'] != fleet['collected']:
                raise ValueError('Cannot replace the release or collection root of an active watcher')
            fleet = fresh
            nodes = nodes_from(fleet,a.nodes)
        snapshots={}
        for node in nodes:
            try:
                d=status(node);snapshots[node['name']]=d
                step=d.get('progress',{}).get('step','-')
                print(node['name'],d.get('phase'),d.get('run_id',''),step,flush=True)
            except (subprocess.CalledProcessError,OSError,ValueError) as e:
                snapshots[node['name']]={'phase':'unreachable','error':str(e)}
        write(local/'fleet_status.json',snapshots)
        if a.action in ('collect','watch'):
            pending=[(n,r) for n in nodes for r in snapshots[n['name']].get('artifacts',[])
                     if snapshots[n['name']].get('phase') in ('awaiting_backup','complete','failed')]
            def attempt(pair):
                try:
                    archive_one(*pair,fleet,local)
                    return None
                except Exception as e:
                    return dict(node=pair[0]['name'],run_id=pair[1]['run_id'],error=str(e))
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                errors=[e for e in pool.map(attempt,pending) if e is not None]
            write(local/'collection_errors.json',errors)
            for e in errors:
                print('Backup pending; will retry:',e,flush=True)
            summarize(local,read(Path(delivery['prepared'])/'scene_index.json'))
        assigned={(s,arm) for n in nodes for s in n['scenes'] for arm in n.get('arms',['structure'])}
        expected={tuple(j) for j in fleet.get('expected_jobs',list(assigned))}
        terminal=all(v.get('phase') in ('complete','smoke_complete','failed') for v in snapshots.values())
        if a.action!='watch' or (terminal and expected <= assigned):
            break
        time.sleep(max(10,a.interval))


if __name__=='__main__':main()
