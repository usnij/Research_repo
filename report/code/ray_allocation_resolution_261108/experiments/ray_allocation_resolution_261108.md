# Ray allocation and target-resolution evaluation: 2026-11-08

The 25 new runs are E1 (3 scenes ×4 allocation arms) and E2 (Counter/Bonsai ×5 resolution arms, Garden ×3). They do not replace historical experiments carrying E1/E2 labels: campaign and run IDs distinguish them. TurboGS and Table 2 are not in this queue.

## Shared contract

- 30,000 completed updates, initialization seed777; private view/ray streams retained from the paired recipe.
- Unmodified native MCMC add: 5% every100, backend indices 500<step<25000; no forced cap-arrival step. Original cap Counter1229851, Bonsai1209886, Garden2044672, Bicycle3021939 at every resolution.
- 0.8L1+0.2SSIM, whole-model opacity/scale penalties each0.01, observed_only=false, MV begins at update1.
- Dense training: fused SSIM 11x11 sigma1.5 valid. Coordinate training: all observed centers, square support ±3sigma, normalized observed weights, C1=.0001,C2=.0009, population moments; all contributing samples receive gradients.
- `[261103]` renderer configuration and native optimizer/MCMC operations are unchanged.
- Every step is CUDA synchronized. Selection, sorting, EMA, loss, backward, optimizer and acceleration rebuild count as training. Setup, checkpoint saving, evaluation and diagnostics do not.
- Checkpoints with optimizer at2k/5k/7k/10k/15k/20k/30k; reload-verify each. Post-training full target-resolution evaluation, standard torchmetrics SSIM(kernel11,sigma1.5) and VGG LPIPS. No resizing of a low-resolution prediction.
- One frozen source commit, exact per-file hashes and a signed run/config plan. A machine lock enforces one job/GPU. No automatic instance destruction, checkpoint pruning or training retry.

## E1 and disclosed quota correction

K4, standard training/evaluation DS, sigma6. Uniform uses existing block6 samples and existing grid-indexed coordinate kernel unchanged. Temporal schedule uses block8 for completed updates1..15000 and block5 for15001..30000. However .5*(1/64+1/25) exceeds1/36 by0.125%, with further ceil-edge differences. We uniformly thin the late block5 sample list to **2*N6-N8 per view**. N_b=ceil(H/b)*ceil(W/b). Thus full30k drawn-ray totals are exactly equal; no unreported nominal-budget mismatch.

Residual: base block8, targetN6. Combined: baseblock11 targetN8 early; baseblock7 target2*N6-N8 late. Per-training-view score maps always have8x8 blocks. Initialize luminance(.2126,.7152,.0722) sample standard deviation(correction1; singleton=0), matching August EVER. Choose bonus blocks proportional to current scores with replacement, then uniformly choose a valid pixel in each block. Zero scores fall back to uniform blocks. Repeated pixels remain rendered samples. Base+bonus are stably sorted by8x8 block. No inverse-probability loss weights.

After the forward pass, compute RGB-mean absolute residual and block means, then update each observed block once using0.9E+0.1mean. Unobserved blocks stay unchanged. Store scores, counts, last update and exact keyed RNG scheme with checkpoints. Additional sampler RNG never changes view/MCMC randomness.

`coordinate_ssim_points` implements arbitrary physical neighborhoods. Repeated coordinates are internally aggregated with multiplicity and within-pixel moments, preserving the exact loss and each rendered sample's derivative. The renderer's ray budget is not deduplicated. Existing-grid loss/gradient parity, duplicates and boundary tests precede deployment. General-kernel overhead is part of measured training time.

`ray_allocation_phases.json`, steps.jsonl and final cumulative_rays record exact quotas. Unique-pixel counts are sampled at step1 and each100 updates. Arm1 saves sampled EMA state plus exact8x8 test residual maps at the seven evaluations; test maps never influence training. Concentration sorts blocks by mean absolute residual and reports pixel-area-weighted residual mass in the top ceil(20%) blocks. This differs from August sparse-minus-dense squared-error deficit concentration.

## E2

Counter/Bonsai evaluateDS1: trainDS1dense K1; trainDS1sparse K4 block8sigma6; trainDS1sparse K4 block16sigma12; trainDS4dense K1; trainDS8dense K1. Garden evaluatesDS2: trainDS2dense K1; trainDS2sparse K4block8sigma6; trainDS8dense K1. Existing image files are byte-preserved, with SHA and decoder verification. Train/eval image sizes and camera intrinsics are verified independently. Sparse/lowres dense ray counts are close rather than identical due image rounding; report exact counts.

## Predeclared assessment

LPIPS gap closure=(L_arm1-L_candidate)/(L_arm1-L_dense). Undefined if denominator<=0. The same candidate must reach>=30% on both Bicycle/Garden, PSNR loss<=0.05dB and training time<=1.05×arm1, plus Counter LPIPS degradation<=0.002. Historical N5 dense provides provisional quality only, not matched timing. Later same-commit/same-instance dense at the matching evaluation resolution provides final comparison.
