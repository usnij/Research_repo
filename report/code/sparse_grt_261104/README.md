# Current sparse GRT training code

This directory contains the complete standalone GRT/MCMC training implementation,
coordinate structural loss, input cap estimator, sampling, and checkpoint evaluator.
The training implementation is copied unchanged from commit
`7fa0939af55467194d5959f5582ee7b6a9ffca5b`. The new file `eval_full.py` calls the existing
`standalone_backend.evaluate` without changing the metric definitions.
See `source_manifest.json` for source hashes and the exact SSIM definitions.

This snapshot is the existing method for review, not the final new-campaign freeze.
GPU reevaluation has not been run; only the evaluator CLI and CPU output/comparison
logic have been checked. Required checkpoint timing/logging changes and renderer
preset interpretation must be resolved before the new campaign starts.

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
