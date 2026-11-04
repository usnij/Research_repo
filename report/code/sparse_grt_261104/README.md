# Current sparse GRT training code

This directory contains the complete standalone GRT/MCMC training implementation,
coordinate structural loss, input cap estimator, sampling, and checkpoint evaluator.
The source is copied from frozen commit
`f472ab69eb6102a0fa7a6a948c4379be335a9c1f`, based on the established training commit
`7fa0939af55467194d5959f5582ee7b6a9ffca5b`. The file `eval_full.py` calls the existing
`standalone_backend.evaluate` without changing the metric definitions.
See `source_manifest.json` for source hashes and the exact SSIM definitions.

The new execution protocol adds checkpoint verification, timing records, post-training
evaluation, and explicit switches for the existing grid/dense control losses. It retains
the renderer, MCMC, coordinate loss, sampler, cap estimator, and growth rule. The
renderer uses the existing reference/referenceBwd configuration with density clamping
and instance primitives. It does not silently switch to the official performance preset.

This source snapshot contains no checkpoint, training result, deployment address, or
instance assignment. Actual GPU validation and run status belong to the result files.

## Installation

Use Python 3.11, PyTorch 2.1.2/cu118 and the pinned dependencies in `pyproject.toml`.
`tools/setup_reference_env.sh` creates the tested CUDA/JPEG environment from source.
It requires uv, mamba, gcc-11, g++-11, make, curl, git and zlib development headers.
It does not replace the host GPU driver.

```bash
bash tools/setup_reference_env.sh /absolute/new/environment
source /absolute/new/environment/activate.sh
```

The package includes GRT and MCMC sources. OptiX headers are fetched from a pinned
public commit during setup. No research-directory imports or prebuilt local binaries
are required by `python -m sparse_grt.training`.

## Existing training recipe

Explicit benchmark configurations are in `configs/benchmark_261103/`.
The following documents the existing Counter recipe, not authorization to run it:

```bash
python -m sparse_grt.training --scene /data/counter --downsample 2 \
  --config configs/benchmark_261103/counter_structure.json \
  --budget fine_tracks_v1 --budget-views all \
  --growth log_horizon --growth-end 5000 --cache-mib 1200 \
  --keep-checkpoints 7000,15000,30000 --output /results/counter --mode train
```

The cap uses all training images and training SfM tracks. Growth reaches the cap at
5,000 steps. Opacity and scale regularization use all Gaussians (`observed_only=false`).
MV begins with the first update. `historical_controls/` contains the actual compact-grid
and dense SSIM adapters; these depend on fused-ssim commit
`1272e21a282342e89537159e4bad508b19b34157`, whose source remains in `../ssim_261007`.

## Frozen execution protocol

Use `--protocol cvpr2027` and `--structural-loss coordinate|grid|dense` with an explicit
method config. The protocol requires 30,000 steps, 0.8 L1 + 0.2 SSIM, opacity/scale
regularization of 0.01 each, and log growth ending at step 5,000. Dense controls require
K=1 and block=1; the fixed-scale coordinate loss requires sigma=6 and all observed centres.
The grid and dense losses require the same fused-ssim dependency as the historical controls.

The trainer retains optimizer checkpoints at 2k, 5k, 7k, 10k, 15k, 20k and 30k. Each is
reloaded to verify its step, count, cumulative update time, and Adam state. Timed updates
include CUDA synchronization and exclude checkpoint/evaluation time. `step_times.csv`
records the mean duration and Gaussian count every 100 updates. Training does not run
evaluation in this mode; `eval_full.py` is called after training finishes.

`tools/cvpr2027/` provides a GPU-locked E1/E2 worker, immutable run signatures, duplicate
run prevention, and unrounded CSV export. The worker requires a successful reference
reevaluation and a verified first 2k checkpoint before releasing the remaining queues.
Private hosts and scene assignments are supplied to `prepare_plan.py prepare --hosts`;
the schema is documented in that script. Deployment plans and readiness acknowledgements
are external inputs, not published source or substitutes for measured results.

## Full-view checkpoint evaluation

```bash
python eval_full.py --checkpoint /results/counter/ckpt_30000.pt \
  --data-root /data/counter --output /results/reevaluated_counter.json \
  --reference /results/counter/evaluation_30000.json
```

The output preserves full-precision means and per-view PSNR/SSIM/VGG-LPIPS.
The optional reference comparison checks PSNR to three decimals and SSIM/LPIPS to
four, plus Gaussian count, validation-view count and filename order. Existing output
files are never overwritten. A reference mismatch saves its result and exits with code 2.
A CUDA GPU is required for actual evaluation.
