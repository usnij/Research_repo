"""Run the 610-update CLI, then compare live versus restored continuation.

The extra 100 updates include the next MCMC relocation/ADD event. This validates
checkpoint contents and reconstruction; it does not add a resume flag to the CLI.
Only load checkpoints created by this script in the same invocation.
"""
import argparse
import gc
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from sparse_grt import standalone_backend as B
from sparse_grt import training
from sparse_grt.config import MethodConfig
from sparse_grt.contracts import validate_manifest
from sparse_grt.engine import update
from sparse_grt.growth import GrowthPlan, install
from sparse_grt.inputs import SelectedInputs
from sparse_grt.losses import coordinate_ssim_all as C
from sparse_grt.provenance import loaded_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('training_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    training_args = args.training_args
    if training_args[:1] == ['--']:
        training_args = training_args[1:]
    captured = {}
    original_save = training.save

    def capture(runtime, model, strategy, out, step, times, manifest_hash, keep):
        original_save(runtime, model, strategy, out, step, times, manifest_hash, keep)
        captured.update(runtime=runtime, model=model, strategy=strategy, out=out, step=step)

    training.save = capture
    sys.argv = ['sparse-grt-train', *training_args]
    training.cli()
    training.save = original_save
    out = captured['out']
    assert captured['step'] == 610, 'Use --mode smoke for this check'
    runtime, model, strategy = (captured[k] for k in ('runtime', 'model', 'strategy'))
    cfg = model.conf
    protocol = json.loads((out/'protocol.json').read_text())
    method = MethodConfig(**protocol['method'])
    assert method.boundary == 'all'
    data = B.CachedDataset(B.dataset(runtime, cfg))
    inputs = SelectedInputs(data, B.Batch, method, resident_budget_bytes=1200*2**20,
                            weight_fn=C.weights)
    schedule = json.loads((out/'training_manifest.json').read_text())
    rows = validate_manifest(schedule, method, len(data))[610:710]
    checkpoint = torch.load(str(out/'checkpoint_latest.pt'), map_location='cpu',
                            mmap=True, weights_only=False)
    assert B.digest({k:getattr(model,k) for k in B.KEYS}) == B.digest(
        {k:checkpoint[k] for k in B.KEYS})
    assert B.digest(model.optimizer.state_dict()) == B.digest(checkpoint['optimizer'])

    def restore_rng():
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        random.setstate(checkpoint['python_rng'])

    def run_continuation(current_model, current_strategy):
        runner = B.Runner(current_model)
        current_model.train()
        restore_rng()
        values = []
        for row in rows:
            step = row['step']
            batch, _, plans = inputs.make(row['views'], row['ray_seed'])
            runner.native(); B.clear(runner)
            result = update(current_model, current_strategy, data, batch, plans, step,
                            method, loss_backend=C.loss,
                            feature_step_condition=B.check_step_condition)
            values.append(dict(step=step+1, **result))
        torch.cuda.synchronize()
        state = B.cpu({k:getattr(current_model,k) for k in B.KEYS})
        hashes = dict(parameters=B.digest(state), optimizer=B.digest(current_model.optimizer.state_dict()),
                      rng=B.digest((torch.get_rng_state(), torch.cuda.get_rng_state_all(),
                                    np.random.get_state(), random.getstate())))
        return values, state, hashes

    expected_rows, expected_state, expected_hashes = run_continuation(model, strategy)
    B.write(out/'continuation_live.json', dict(rows=expected_rows, hashes=expected_hashes))
    cfg = checkpoint['config']
    restored = B.MixtureOfGaussians(cfg)
    for name in B.KEYS:
        checkpoint[name] = torch.nn.Parameter(checkpoint[name].cuda())
    restored.init_from_checkpoint(checkpoint, setup_optimizer=True)
    restored.cuda()
    resumed_strategy = B.mc.MCMCStrategy(cfg, restored)
    resumed_strategy.init_densification_buffer(checkpoint)
    plan_data = json.loads((out/'growth_plan.json').read_text())
    plan = GrowthPlan(**{k:plan_data[k] for k in
                        ('initial','target','first_event','last_event','frequency')})
    install(resumed_strategy, plan)
    B.keyed_mcmc(resumed_strategy)
    resumed_strategy.paired_step = checkpoint['paired_step']
    restored.build_acc(rebuild=True)
    actual_rows, actual_state, actual_hashes = run_continuation(restored, resumed_strategy)
    errors = {k:float((actual_state[k]-expected_state[k]).abs().max()) for k in B.KEYS}
    forbidden = [n for n in sys.modules if n.startswith(
        ('threedgrut','threedgrt_tracer','area_tracer512','lib3dgrt','lib_mcmc'))]
    provenance = loaded_runtime()
    package_paths = {n:v['path'] for n,v in provenance['modules'].items()
                     if n.startswith(('sparse_grt.', 'sparse_grt_backend.', 'sparse_grt_tracer.'))}
    installed_only = all('/site-packages/' in p for p in package_paths.values())
    passed = (expected_rows == actual_rows and expected_hashes == actual_hashes
              and not forbidden and installed_only)
    result = dict(passed=passed, from_scratch_steps=610, continuation_steps=100,
                  continuation_end=710, final_N=restored.num_gaussians,
                  relocation_and_growth_event_crossed=any(r['delta_N']>0 for r in actual_rows),
                  step_losses_counts_bit_equal=expected_rows==actual_rows,
                  live_hashes=expected_hashes, restored_hashes=actual_hashes,
                  parameter_max_abs_errors=errors, forbidden_modules=forbidden,
                  installed_package_only=installed_only, package_paths=package_paths,
                  scope='Same-GPU live versus restored continuation; not a full-training quality check')
    B.write(args.report, result)
    B.write(out/'continuation_restored.json', dict(rows=actual_rows, hashes=actual_hashes))
    B.write(out/'installed_runtime_after_training.json', provenance)
    print('CONTINUATION', json.dumps({k:v for k,v in result.items() if k!='package_paths'}), flush=True)
    if not passed:
        raise SystemExit('Checkpoint continuation differs; inspect the report')


if __name__ == '__main__':
    main()
