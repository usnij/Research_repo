"""Train one COLMAP scene with explicit budget and growth policies.

The renderer and MCMC extensions are built from the included source. No Counter-only paths,
counts, image dimensions or validation-view counts are assumed here.
"""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

from sparse_grt.config import MethodConfig
from sparse_grt.budget import estimate
from sparse_grt.growth import GrowthPlan, NativeGrowthAudit
from sparse_grt import standalone_backend as backend
from sparse_grt.inputs import SelectedInputs
from sparse_grt.engine import update
from sparse_grt.contracts import make_manifest, validate_manifest, read_manifest
from sparse_grt.provenance import loaded_runtime


def save(runtime, model, strategy, out, step, times, manifest_hash, keep, input_state=None):
    if shutil.disk_usage(out).free < 4*2**30:
        raise RuntimeError('Less than 4 GiB free before checkpoint save')
    torch,S=runtime.torch,runtime.S
    state=model.get_model_parameters()
    state.update(global_step=step,training_seconds=times[0],evaluation_seconds=times[1],
                 checkpoint_seconds=times[2],from_scratch=True,manifest_hash=manifest_hash,
                 torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                 numpy_rng=runtime.np.random.get_state(),python_rng=random.getstate(),
                 paired_step=getattr(strategy,'paired_step',None))
    state['N'] = model.num_gaussians
    if input_state is not None:
        state['allocation_input_state'] = input_state
    pending=out/'checkpoint.pending.pt'
    torch.save(S.cpu(state),pending);pending.replace(out/'checkpoint_latest.pt')
    if step in keep:
        dest=out/f'ckpt_{step}.pt'
        if dest.exists():raise FileExistsError(dest)
        os.link(out/'checkpoint_latest.pt',dest)


def main(a):
    # Legacy imports may change cwd; resolve every supplied path beforehand.
    out, scene, config = [p.resolve() for p in (a.output,a.scene,a.config)]
    manifest_path = a.manifest.resolve() if a.manifest else None
    if out.exists():
        raise FileExistsError(out)
    method = MethodConfig.read(config)
    protocol_name = getattr(a, 'protocol', 'legacy')
    cvpr = protocol_name == 'cvpr2027'
    native5 = protocol_name == 'native5_singleview'
    capacity_duration = protocol_name == 'native5_capacity_duration'
    micropatch = protocol_name == 'micropatch_singleview'
    allocation = protocol_name == 'allocation_resolution'
    recorded = cvpr or native5 or capacity_duration or micropatch or allocation
    structural_loss = getattr(a, 'structural_loss', 'coordinate')
    if not micropatch and (method.sampler != 'point' or structural_loss == 'patch'):
        raise ValueError('Micropatch sampling/loss require their separate execution protocol')
    if cvpr:
        from sparse_grt import protocol as P
        P.validate(method, a)
    elif native5:
        from sparse_grt import native5_protocol as P
        P.validate(method, a)
    elif capacity_duration:
        from sparse_grt import capacity_duration_protocol as P
        P.validate(method, a)
    elif micropatch:
        from sparse_grt import micropatch_protocol as P
        P.validate(method, a)
    elif allocation:
        from sparse_grt import allocation_protocol as P
        P.validate(method, a)
    paper_guard = getattr(a, 'paper_guard', None)
    if paper_guard is not None:
        paper_guard.verify()
        paper_guard.validate_method(method, a.paper_preset)
    keep=(set(P.checkpoint_steps(method)) if capacity_duration else set(P.CHECKPOINTS)
          if recorded else {int(x) for x in a.keep_checkpoints.split(',') if x})
    if any(x<=0 or x>method.steps or x%1000 for x in keep):
        raise ValueError('Retained checkpoints must be valid 1000-step boundaries')
    if micropatch:
        from sparse_grt.losses import micropatch_coordinate as C
    elif method.boundary == 'all':
        from sparse_grt.losses import coordinate_ssim_all as C
    else:
        from sparse_grt.losses import coordinate_ssim as C
    T = backend.load()
    torch, S = T.torch, T.S
    gpu_name = P.validate_gpu(torch) if recorded else None
    cfg = backend.configure(T, scene, a.downsample, method)
    if native5 or micropatch or allocation:
        P.configure_native_add(cfg)
    if capacity_duration:
        schedule_record = P.configure(cfg, method)
    raw_data = backend.dataset(T, cfg)
    if a.budget != 'fixed':
        budget = estimate(raw_data.image_paths, method=a.budget,
                          images_bin=scene/'sparse/0/images.bin',
                          view_limit=None if a.budget_views == 'all' else 16)
        method = replace(method, cap=budget['cap'])
        cfg.strategy.add.max_n_gaussians = method.cap
    else:
        budget = dict(method='explicit_fixed_reference', cap=method.cap, seconds=0.)
    if paper_guard is not None:
        # Recomputed input budget must equal the registered scene value.
        paper_guard.validate_method(method, a.paper_preset)
    out.mkdir(parents=True)
    S.write(out/'budget.json', budget)
    data = T.B.CachedDataset(raw_data)
    growth_factory = None
    if a.growth == 'log_horizon':
        if a.growth_end is None:
            raise ValueError('An explicit common growth end step is required')
        growth_factory = lambda n: GrowthPlan(n, method.cap, 600, a.growth_end)
    elif a.growth_end is not None:
        raise ValueError('growth-end only applies to log_horizon')
    model, strategy, runner, plan = backend.initialize(T, cfg, method, data, growth_factory)
    if native5 or micropatch or allocation:
        # This is count validation only. Keep the native strategy's add method.
        plan = NativeGrowthAudit(model.num_gaussians, method.cap)
    if capacity_duration:
        plan = P.count_audit(model.num_gaussians, method, cfg)
    if paper_guard is not None:
        runtime_check = paper_guard.validate_runtime(T, cfg, method, scene, a.downsample)
        S.write(out/'paper_release.json', dict(**paper_guard.report, runtime=runtime_check,
                                             mode=a.mode, preset=a.paper_preset))
    initial = dict(N=model.num_gaussians, parameters=S.digest({k:getattr(model,k) for k in S.KEYS}),
                   optimizer_states=len(model.optimizer.state), from_scratch=True)
    S.write(out/'initial.json',initial)
    if plan:
        S.write(out/'growth_plan.json',plan.record())
    S.write(out/'resolved_config.json',S.OmegaConf.to_container(cfg,resolve=True))
    input_type, objective_backend = SelectedInputs, None
    if structural_loss == 'patch':
        from sparse_grt.micropatch import PatchInputs, patch_objective
        input_type, objective_backend = PatchInputs, patch_objective
    if structural_loss in ('grid', 'dense'):
        from sparse_grt.controls import GridInputs, DenseInputs, grid_objective, dense_objective
        input_type, objective_backend = ((GridInputs, grid_objective) if structural_loss == 'grid'
                                         else (DenseInputs, dense_objective))
        # Import before starting training, so a missing extension fails at setup.
        if structural_loss == 'dense' or method.grid_window == 11:
            from fused_ssim import fused_ssim
    input_options = {}
    if allocation and a.allocation_arm != 'none':
        from sparse_grt.allocation_inputs import AllocationInputs
        input_type, objective_backend = AllocationInputs, P.point_objective
        input_options['allocation_arm'] = a.allocation_arm
    inputs = input_type(data, S.Batch, method, resident_budget_bytes=int(a.cache_mib*2**20),
                        weight_fn=C.weights, **input_options)
    schedule = (read_manifest(manifest_path,method,len(data)) if manifest_path
                else make_manifest(method,len(data)))
    rows = validate_manifest(schedule,method,len(data))
    mh = S.digest(rows)
    S.write(out/'training_manifest.json',schedule)
    S.write(out/'ray_geometry.json',inputs.geometry)
    if allocation and a.allocation_arm != 'none':
        from sparse_grt.allocation_inputs import allocation_budget
        S.write(out/'ray_allocation_phases.json', {str(vid): {str(step): allocation_budget(
            g['height'],g['width'],a.allocation_arm,step) for step in (1,15001)}
            for vid,g in inputs.geometry.items()})
    S.write(out/'runtime_manifest.json',loaded_runtime())
    protocol = dict(method=vars(method),scene=str(scene),downsample=a.downsample,
                    boundary=method.boundary,growth=a.growth,growth_end=a.growth_end,
                    budget_method=a.budget,manifest_hash=mh,train_views=len(data),
                    input_mode='resident_uint8' if inputs.resident else 'selective_host',
                    image_cache_bytes=inputs.resident_bytes,
                    test_RGB_used_for_budget=False,extra_training_rays=0,
                    timing='Synchronized full steps; budget/setup/evaluation/checkpoint time separately recorded')
    protocol.update(execution_protocol=protocol_name,
                    structural_loss=structural_loss, gpu_name=gpu_name,
                    checkpoint_steps=sorted(keep),
                    evaluation_phase='after_training_external_eval_full' if recorded else 'during_training',
                    observed_only=False, multiview_start_step=0,
                    multiview_start_completed_step=1 if method.views > 1 else None)
    if native5 or micropatch or allocation:
        protocol.update(native_add_schedule=dict(start_exclusive=500, end_exclusive=25000,
                        first_event=600, last_event=24900, frequency=100,
                        target_rule='min(cap, int(1.05 * current_G))',
                        original_add_method_preserved=True),
                        compact_window=method.grid_window if structural_loss == 'grid' else None,
                        compact_sigma=method.grid_sigma if structural_loss == 'grid' else None,
                        compact_padding='valid' if structural_loss in ('grid', 'dense') else None,
                        ssim_backend=('torch_separable_population_moments' if structural_loss == 'grid'
                                      and method.grid_window == 4 else 'fused_ssim_11x11'
                                      if structural_loss in ('grid', 'dense') else 'coordinate_ssim_all'),
                        native_count_audit=plan.record())
    if micropatch:
        protocol.update(P.record(method, structural_loss))
    if capacity_duration:
        protocol.update(schedule_record,
                        native_add_schedule=dict(start_exclusive=plan.start,
                            end_exclusive=plan.end, first_event=plan.events[0],
                            last_event=plan.events[-1], frequency=plan.frequency,
                            target_rule='min(cap, int(1.05 * current_G))',
                            original_add_method_preserved=True),
                        evaluation_steps=list(P.evaluation_steps(method)),
                        native_count_audit=plan.record(),
                        ssim_backend=('fused_ssim_11x11' if structural_loss == 'dense'
                                      else 'coordinate_ssim_all'),
                        compact_padding='valid' if structural_loss == 'dense' else None)
    if allocation:
        protocol.update(campaign='ray_allocation_resolution_261108',
            allocation_arm=a.allocation_arm, train_downsample=a.downsample,
            eval_downsample=a.eval_downsample,
            ray_budget_correction='late_target=2*N6-N8; uniform thinning of block5 samples'
                if a.allocation_arm in ('schedule','combined') else None,
            score_grid=8 if a.allocation_arm != 'none' else None,
            importance_loss_reweighting=False, memory_fraction_limit=.82)
        if a.allocation_arm in ('schedule','residual','combined'):
            protocol['ssim_backend'] = 'coordinate_ssim_points_exact_multiplicity'
    S.write(out/'protocol.json',protocol)
    limit = 610 if a.mode == 'smoke' else method.steps
    train_s=eval_s=save_s=0.
    growth_events=0
    completed_step=0
    completed_count=model.num_gaussians
    checkpoint_saved_step=0
    attempted_step=0
    phase='before_first_update'
    model.train()
    total_rays = 0
    if allocation:
        torch.cuda.reset_peak_memory_stats()
    try:
        with (out/'steps.jsonl').open('w') as fp, (P.StepLog(out) if recorded else nullcontext()) as timing_log:
            for row in rows[:limit]:
                attempted_step=row['step']+1
                phase='training_update'
                t=row['step']; torch.cuda.synchronize(); tick=time.perf_counter()
                if allocation and a.allocation_arm != 'none':
                    inputs.set_step(t+1)
                batch,_,plans=inputs.make(row['views'],row['ray_seed'])
                runner.native(); T.M.clear(runner)
                update_options = {}
                if allocation and a.allocation_arm != 'none':
                    update_options['observation_callback'] = inputs.update_residual
                values=update(model,strategy,data,batch,plans,t,method,loss_backend=C.loss,
                              feature_step_condition=T.B.check_step_condition,
                              objective_backend=objective_backend, **update_options)
                torch.cuda.synchronize(); elapsed=time.perf_counter()-tick; train_s+=elapsed
                if not all(math.isfinite(v) for v in values.values()) or not 0<values['N']<=method.cap:
                    raise FloatingPointError(values)
                if plan and values['N']!=plan.count_after(t):
                    raise RuntimeError('Actual growth differs from the precomputed schedule')
                if allocation:
                    drawn = sum(p['end']-p['start'] for p in plans)
                    total_rays += drawn
                    if a.allocation_arm != 'none':
                        values.update(base_rays=sum(p['base_count'] for p in plans),
                                      bonus_rays=sum(p['bonus_count'] for p in plans))
                        if all(p['unique_ray_count'] is not None for p in plans):
                            values['unique_rays'] = sum(p['unique_ray_count'] for p in plans)
                    values.update(rays=drawn, cumulative_rays=total_rays,
                                  peak_torch_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                                  peak_torch_reserved_bytes=int(torch.cuda.max_memory_reserved()))
                completed_step=t+1
                completed_count=values['N']
                growth_events+=int(values['delta_N']>0)
                fp.write(json.dumps(dict(step=t+1,seconds=elapsed,**values))+'\n')
                if timing_log is not None:
                    timing_log.append(t+1, elapsed, values['N'])
                if t==0:
                    S.write(out/'first_input.json',dict(hash=S.digest(
                        {k:getattr(batch,k) for k in ('rays_ori','rays_dir','rgb_gt')}), losses=values))
                del batch,plans
                if (t+1)%50==0 or t+1==limit:
                    fp.flush(); progress=dict(step=t+1,total_steps=limit,training_seconds=train_s,**values)
                    S.write(out/'progress.json',progress); print('PROGRESS',json.dumps(progress),flush=True)
                if not recorded and a.mode=='train' and t+1 in (7000,15000,method.steps):
                    model.eval(); gc.collect(); torch.cuda.empty_cache(); tick=time.perf_counter()
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        result=backend.evaluate(T,model,cfg)
                    eval_s+=time.perf_counter()-tick
                    S.write(out/f'evaluation_{t+1}.json',result); model.train()
                checkpoint_due = ((t+1 in keep) if recorded else ((t+1)%1000==0))
                if (a.mode=='train' and checkpoint_due) or t+1==limit:
                    phase='checkpoint_save'
                    tick=time.perf_counter()
                    if allocation and a.allocation_arm != 'none':
                        save(T,model,strategy,out,t+1,(train_s,eval_s,save_s),mh,keep,
                             input_state=inputs.state_dict())
                    else:
                        save(T,model,strategy,out,t+1,(train_s,eval_s,save_s),mh,keep)
                    if recorded and t+1 in keep:
                        receipt = P.verify_checkpoint(T, out/f'ckpt_{t+1}.pt', t+1,
                                                      train_s, model.num_gaussians)
                        S.write(out/f'checkpoint_verified_{t+1}.json', receipt)
                    checkpoint_saved_step=t+1
                    save_s+=time.perf_counter()-tick
    except Exception as error:
        if capacity_duration or micropatch or allocation:
            # Do not synchronize/allocate CUDA memory after an OOM. A partially
            # completed update may have changed G, so retain both counts.
            message=str(error).lower().replace('_',' ')
            oom=isinstance(error, torch.cuda.OutOfMemoryError) or (
                any(token in message for token in ('out of memory', 'memory allocation failed'))
                and any(token in message for token in ('cuda','optix')))
            failure=dict(passed=False, phase=phase,
                error_kind='cuda_out_of_memory' if oom else type(error).__name__,
                error=str(error), step=completed_step, attempted_step=attempted_step,
                N=model.num_gaussians, G=model.num_gaussians,
                last_completed_G=completed_count, planned_steps=method.steps,
                training_seconds=train_s, checkpoint_seconds=save_s,
                final_checkpoint_saved_step=checkpoint_saved_step,
                from_scratch=True, final_optimizer_included=False)
            for name in ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved'):
                try:
                    failure['cuda_'+name+'_bytes']=int(getattr(torch.cuda,name)())
                except Exception:
                    pass
            S.write(out/'failure.json',failure)
        raise
    first_growth_step=(plan.events[0]+1 if capacity_duration else 601)
    if limit>=first_growth_step and initial['N']<method.cap and not growth_events:
        raise RuntimeError('No growth occurred')
    completion = dict(passed=True,steps=limit,N=model.num_gaussians,
            training_seconds=train_s,evaluation_seconds=eval_s,checkpoint_seconds=save_s,
            growth_events=growth_events,from_scratch=True)
    if allocation:
        completion.update(cumulative_rays=total_rays,
                          peak_torch_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                          peak_torch_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    if recorded:
        completion.update(phase='training_complete', evaluation_pending=True,
                          evaluation_steps=(list(P.evaluation_steps(method)) if capacity_duration
                                            else sorted(keep)) if a.mode == 'train' else [],
                          final_optimizer_included=True)
        S.write(out/'training_complete.json', completion)
    S.write(out/'complete.json', completion)


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--downsample',type=int,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--budget',choices=('fixed','fine_tracks_v1','haar_v1'),required=True)
    p.add_argument('--budget-views',choices=('16','all'),default='16')
    p.add_argument('--growth',choices=('native_5percent','log_horizon'),required=True)
    p.add_argument('--growth-end',type=int)
    p.add_argument('--cache-mib',type=int,default=1200)
    p.add_argument('--mode',choices=('smoke','train'),default='smoke')
    p.add_argument('--manifest',type=Path)
    p.add_argument('--keep-checkpoints',default='7000,15000,30000')
    p.add_argument('--protocol',choices=('legacy','cvpr2027','native5_singleview','native5_capacity_duration','micropatch_singleview','allocation_resolution'),default='legacy',
                   help='CVPR: seven retained checkpoints, update-only timing, external post-training evaluation')
    p.add_argument('--structural-loss',choices=('coordinate','grid','dense','patch'),default='coordinate',
                   help='Established coordinate loss, historical compact-grid SSIM, or historical dense SSIM')
    p.add_argument('--allocation-arm', choices=('none','uniform','schedule','residual','combined'), default='none')
    p.add_argument('--eval-downsample', type=int)
    main(p.parse_args())


if __name__ == "__main__":
    cli()
