"""Export all 25 planned/observed campaign rows without inventing measurements."""
import argparse
import csv
import json
import math
from pathlib import Path
import shlex

try:
    from .common import STEPS, eval_command, read, train_command, write
    from .worker import load_plan, run_signature
except ImportError:
    from common import STEPS, eval_command, read, train_command, write
    from worker import load_plan, run_signature


RUN_COLUMNS = ['run_id', 'campaign', 'experiment_id', 'scene', 'arm', 'status', 'execution_order',
    'densification', 'loss', 'K', 'block_size', 'R_total', 'rays_first_half', 'rays_second_half',
    'cumulative_rays', 'planned_cumulative_rays', 'train_DS', 'eval_DS', 'resolution', 'eval_resolution',
    'renderer_settings', 'cap', 'final_G', 'seed', 'observed_only', 'MV_start_step', 'instance',
    'GPU', 'driver', 'power_limit', 'commit', 'training_seconds', 'mean_step_seconds',
    'PSNR', 'SSIM', 'LPIPS', 'sigma_pixels', 'dense_window', 'dense_sigma',
    'peak_training_device_used_mib', 'peak_evaluation_device_used_mib',
    'peak_torch_allocated_bytes', 'peak_torch_reserved_bytes', 'plan_signature',
    'failure_step', 'failure_G', 'failure_phase', 'failure_stage', 'failure_evaluation_step', 'settings_changes']
CURVE_COLUMNS = ['run_id', 'step', 'training_seconds', 'G', 'PSNR', 'SSIM', 'LPIPS']
VIEW_COLUMNS = ['run_id', 'view', 'image', 'width', 'height', 'PSNR', 'SSIM', 'LPIPS']
LOG_COLUMNS = ['step', 'mean_step_seconds_100', 'G']
SUMMARY_COLUMNS = ['scene', 'experiment_id', 'arm', 'eval_resolution', 'final_G',
    'training_seconds', 'mean_step_seconds', 'PSNR', 'SSIM', 'LPIPS', 'instance',
    'peak_training_device_used_mib', 'peak_evaluation_device_used_mib', 'status']


def maybe(path):
    return read(path) if path is not None and path.is_file() else {}


def csv_file(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix+'.pending')
    with pending.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='raise')
        writer.writeheader()
        for row in rows:
            writer.writerow({c: 'NA' if row.get(c) is None else row[c] for c in columns})
    pending.replace(path)


def metrics(value):
    result = {k: value.get(k) for k in ('PSNR', 'SSIM', 'LPIPS')}
    if value and any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in result.values()):
        raise ValueError('A metric is missing or non-finite')
    return result


def checked_evaluation(path, run, step):
    value = maybe(path)
    if not value:
        return {}
    expected = dict(run_id=run['run_id'], step=step, views=run['test_views'],
                    train_downsample=run['train_ds'], eval_downsample=run['eval_ds'],
                    N=run['expected_G'][str(step)])
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('Evaluation identity/count/resolution differs: '+str(path))
    per_view = value.get('per_view', [])
    if len(per_view) != run['test_views'] or len({v['view'] for v in per_view}) != len(per_view):
        raise ValueError('Missing or duplicated test view')
    for view in per_view:
        metrics(view)
        if [view.get('width'), view.get('height')] != run['eval_shape']:
            raise ValueError('A test view has the wrong target resolution')
    for key, actual in metrics(value).items():
        expected_mean = sum(view[key] for view in per_view)/len(per_view)
        if not math.isclose(actual, expected_mean, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Aggregate metric is not the mean of all heldout views')
    return value


def peak_memory(root, rid, training):
    if root is None:
        return None
    paths = [root/'memory'/f'{rid}.train.json'] if training else list((root/'memory').glob(f'{rid}.eval*.json'))
    values = [maybe(p).get('peak_used_mib') for p in paths]
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return max(values) if values else None


def log_rows(path):
    if not path.is_file():
        return []
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != LOG_COLUMNS:
            raise ValueError('Unexpected 100-step timing columns')
        rows = list(reader)
    good = []
    for index, row in enumerate(rows):
        if any(row.get(k) in (None, '') for k in LOG_COLUMNS):
            if index == len(rows)-1:
                break  # Collector may have copied an in-flight final line.
            raise ValueError('Incomplete nonfinal timing row')
        step = int(row['step'])
        duration = float(row['mean_step_seconds_100'])
        if step != (len(good)+1)*100 or not math.isfinite(duration) or duration < 0:
            raise ValueError('Timing rows are duplicated, unordered or invalid')
        good.append({k: row[k] for k in LOG_COLUMNS})
    return good


def summary_markdown(rows, changes):
    names = ['씬', '실험', 'arm', '평가 해상도', '최종 G', '학습시간(s)', 'step당(s)',
             '30k PSNR', 'SSIM', 'LPIPS', '인스턴스', '최대 학습 VRAM(MiB)', '최대 평가 VRAM(MiB)', '상태']
    lines = ['| '+' | '.join(names)+' |', '| '+' | '.join(['---']*len(names))+' |']
    for row in rows:
        cells = []
        for key in SUMMARY_COLUMNS:
            value = row.get(key)
            if value is None:
                cell = '—'
            elif isinstance(value, float):
                cell = format(value, '.6f').rstrip('0').rstrip('.')
            else:
                cell = str(value)
            cells.append(cell.replace('|', '\\|'))
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '시간: CUDA 동기화 전체 학습 step의 합이며 평가·저장 시간 제외. '
              'step당 시간=총 학습시간/30,000. VRAM은 2초 간격 nvidia-smi 관측 최대값이며 '
              '학습과 평가를 별도로 표시합니다. 원시 CSV/JSON 지표는 반올림하지 않습니다.', '',
              '| 명시한 설정 차이 |', '| --- |']
    lines += ['| '+str(value).replace('|', '\\|')+' |' for value in changes]
    return '\n'.join(lines)+'\n'


def export(plan, roots, output):
    output.mkdir(parents=True, exist_ok=True)
    runs, curves, views, commands = [], [], [], []
    for node in plan['nodes']:
        root = roots.get(node['node_id'])
        hardware = maybe(root/'hardware.json') if root else {}
        for run in node['runs']:
            rid, method = run['run_id'], run['method']
            out = root/'runs'/rid if root else None
            evidence = maybe(root/'run_manifests'/f'{rid}.json') if root else {}
            success = maybe(root/'receipts'/f'{rid}.json') if root else {}
            failure = maybe(root/'failure_receipts'/f'{rid}.json') if root else {}
            for item in (evidence, success, failure):
                if item and item.get('run_signature') != run_signature(plan, run):
                    raise ValueError('Collected evidence belongs to another frozen plan')
            if success and failure:
                raise ValueError('Both completion and failure receipts exist')
            if success and success.get('passed') is not True:
                raise ValueError('Completion receipt does not report success')
            if failure and (failure.get('terminal') is not True or failure.get('status') != 'cuda_oom'):
                raise ValueError('Failure is not a verified terminal CUDA OOM')
            protocol = maybe(out/'protocol.json') if out else {}
            progress = maybe(out/'progress.json') if out else {}
            training = maybe(out/'training_complete.json') if out else {}
            final = checked_evaluation(out/'evaluation_30000.json', run, 30000) if out else {}
            status = 'planned'
            if evidence or (out and out.is_dir()): status = 'started'
            if progress: status = f"training_{progress['step']}"
            if training: status = 'training_complete_evaluation_pending'
            if final: status = 'final_evaluated_receipt_pending'
            if success: status = 'complete'
            if failure: status = 'cuda_oom'
            if success and not final:
                raise ValueError('Completion receipt has no collected final evaluation')
            training_seconds = training.get('training_seconds', failure.get('training_seconds'))
            rfirst, rsecond = run['rays_first_second']
            row = dict(run_id=rid, campaign=plan['campaign'], experiment_id=run['experiment_id'],
                scene=run['scene'], arm=run['arm'], status=status, execution_order=run['execution_order'],
                densification='MCMC 5%/100step;500<index<25000;original_cap',
                loss=f"0.8 L1 + 0.2 {run['loss']} SSIM; opacity/scale reg 0.01 each",
                K=method['views'], block_size=method['block'], R_total=rfirst if rfirst==rsecond else f'{rfirst};{rsecond}',
                rays_first_half=rfirst, rays_second_half=rsecond,
                cumulative_rays=training.get('cumulative_rays'), planned_cumulative_rays=run['total_rays'],
                train_DS=run['train_ds'], eval_DS=run['eval_ds'],
                resolution='x'.join(map(str, run['train_shape'])),
                eval_resolution='x'.join(map(str, run['eval_shape']))+f" (DS{run['eval_ds']})",
                renderer_settings=json.dumps(plan['renderer_settings'], sort_keys=True, separators=(',', ':')),
                cap=method['cap'], final_G=final.get('N', training.get('N')),
                seed=method['initialization_seed'], observed_only=protocol.get('observed_only'),
                MV_start_step=protocol.get('multiview_start_step'), instance=node['instance_id'],
                GPU=hardware.get('name'), driver=hardware.get('driver'), power_limit=hardware.get('power_limit_w'),
                commit=plan['frozen_commit'], training_seconds=training_seconds,
                mean_step_seconds=training_seconds/30000 if training and training_seconds is not None else None,
                **metrics(final), sigma_pixels=method['sigma'] if run['loss']=='coordinate' else None,
                dense_window=11 if run['loss']=='dense' else None, dense_sigma=1.5 if run['loss']=='dense' else None,
                peak_training_device_used_mib=peak_memory(root, rid, True),
                peak_evaluation_device_used_mib=peak_memory(root, rid, False),
                peak_torch_allocated_bytes=training.get('peak_torch_allocated_bytes'),
                peak_torch_reserved_bytes=training.get('peak_torch_reserved_bytes'),
                plan_signature=plan['plan_signature'], failure_step=failure.get('step'), failure_G=failure.get('G'),
                failure_phase=failure.get('phase'), failure_stage=failure.get('stage'),
                failure_evaluation_step=failure.get('evaluation_step'),
                settings_changes=protocol.get('ray_budget_correction'))
            runs.append(row)
            for step in STEPS:
                proof = maybe(out/f'checkpoint_verified_{step}.json') if out else {}
                evaluation = checked_evaluation(out/f'evaluation_{step}.json', run, step) if out else {}
                if not proof and not evaluation:
                    continue
                if proof and (proof.get('step') != step or proof.get('load_verified') is not True or
                              proof.get('G') != run['expected_G'][str(step)]):
                    raise ValueError('Checkpoint verification receipt differs from the run')
                curves.append(dict(run_id=rid, step=step, training_seconds=proof.get('training_seconds'),
                                   G=proof.get('G', evaluation.get('N')), **metrics(evaluation)))
            if final:
                views.extend(dict(run_id=rid, view=v['view'], image=v['image'], width=v['width'],
                                  height=v['height'], **metrics(v)) for v in final['per_view'])
            if out and (out/'step_times.csv').is_file():
                csv_file(output/'logs'/f'{rid}.csv', LOG_COLUMNS, log_rows(out/'step_times.csv'))
            commands += [f"# {rid}; order {run['execution_order']}; {node['node_id']}; instance {node['instance_id']}",
                shlex.join(train_command(plan['remote_root'], run, plan['python']))]
            commands += [shlex.join(eval_command(plan['remote_root'], run, s, plan['python'])) for s in STEPS]
    csv_file(output/'runs.csv', RUN_COLUMNS, runs)
    csv_file(output/'curves.csv', CURVE_COLUMNS, curves)
    csv_file(output/'views.csv', VIEW_COLUMNS, views)
    (output/'commands.txt').write_text('\n'.join(commands)+'\n', encoding='utf-8')
    summary = [{k: r.get(k) for k in SUMMARY_COLUMNS} for r in runs]
    write(output/'summary.json', dict(campaign=plan['campaign'], plan_signature=plan['plan_signature'], rows=summary))
    (output/'summary.md').write_text(summary_markdown(summary, plan.get('changes_from_order', [])), encoding='utf-8')
    write(output/'export_scope.json', dict(campaign=plan['campaign'], plan_signature=plan['plan_signature'],
        runs=len(runs), curves=len(curves), views=len(views), missing_value='NA',
        precision='Unrounded evaluator floats in CSV/JSON; Markdown is display-only',
        historical_reused_runs_included=False, expected_runs=25,
        step_time_definition='synchronized training_seconds / 30000; setup/checkpoint/evaluation excluded'))
    print(json.dumps(dict(runs=len(runs), curves=len(curves), views=len(views), output=str(output))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--node-root', action='append', default=[], metavar='NODE=/LOCAL/COLLECTED/ROOT')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    roots = {}
    for item in args.node_root:
        node, path = item.split('=', 1)
        if node in roots:
            raise ValueError('Repeated node root')
        roots[node] = Path(path).resolve()
    plan = load_plan(args.plan)
    if set(roots)-{n['node_id'] for n in plan['nodes']}:
        raise ValueError('Collected root has an unknown node ID')
    export(plan, roots, args.output.resolve())


if __name__ == '__main__':
    main()
