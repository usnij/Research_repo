"""Transitional runner using an explicitly supplied existing renderer bootstrap.

This keeps the validated renderer/Adam/MCMC configuration while replacing the
duplicated training loop. This adapter is not a standalone public backend.
"""
from pathlib import Path
import argparse
import gc
import importlib.util
import functools
import json
import math
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from sparse_grt.config import MethodConfig
from sparse_grt.engine import update
from sparse_grt.inputs import SelectedInputs
from sparse_grt.losses import coordinate_ssim as C
from sparse_grt.contracts import make_manifest, read_manifest, validate_manifest, ray_geometry
from sparse_grt.provenance import loaded_runtime


def main(args):
    cfg = MethodConfig.read(args.config)
    if cfg.boundary != 'valid':
        raise ValueError("Use train_scene.py for the explicit all-centre objective")
    entry, out = args.legacy_entry.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    # The external compatibility adapter currently represents exactly this
    # benchmark. Reject other settings rather than silently mislabel a run.
    if (cfg.views != 4 or cfg.block != 8 or cfg.cap != 1131968
            or cfg.initialization_seed != 777 or cfg.steps != 30000):
        raise ValueError("The legacy Counter adapter is MV4/block8/cap1131968/seed777/30k only")
    sys.path.insert(0, str(entry.parent))
    spec = importlib.util.spec_from_file_location("explicit_legacy_backend", entry)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    torch, S = T.torch, T.S
    model, backend_config, strategy, dataset, runner, _ = T.setup(out)
    S.write(out/"runtime_manifest.json",loaded_runtime())
    backend_config.loss.lambda_l1 = cfg.l1_weight
    backend_config.loss.lambda_ssim = cfg.structure_weight
    backend_config.loss.lambda_opacity = cfg.opacity_weight
    backend_config.loss.lambda_scale = cfg.scale_weight
    weight_fn = functools.partial(C.weights, sigma=cfg.sigma)
    if args.input_mode == "legacy":
        T.M.A.spatial_weights = weight_fn if cfg.structure_weight else lambda *a,**k:(None,7)
        inputs = T.M.Inputs(dataset)
    else:
        inputs = SelectedInputs(dataset, S.Batch, cfg,
                                resident_budget_bytes=int(args.gpu_image_cache_mib*2**20)
                                if args.input_mode == "resident_uint8" else 0)
    schedule = (read_manifest(args.manifest,cfg,len(dataset)) if args.manifest
                else make_manifest(cfg,len(dataset)))
    manifest = validate_manifest(schedule,cfg,len(dataset))
    manifest_hash = S.digest(manifest)
    protocol = dict(method=vars(cfg), input_mode=args.input_mode, mode=args.mode,
                    manifest_hash=manifest_hash, legacy_bootstrap=str(entry),
                    resident_image_bytes=getattr(inputs,"resident_bytes",0),
                    runtime_modules={k:getattr(sys.modules.get(k),"__file__",None) for k in
                                     ("threedgrut", "threedgrut.strategy.mcmc", "threedgrt_tracer")},
                    timing="Synchronized complete steps, excludes setup/evaluation/checkpoint I/O")
    S.write(out/"protocol.json", protocol)
    S.write(out/"inputs.json", manifest)
    S.write(out/"training_manifest.json", schedule)
    # This compatibility adapter is explicitly Counter DS2; other datasets
    # must provide their actual per-view dimensions through a separate adapter.
    S.write(out/"ray_geometry.json", ray_geometry(1038,1558,cfg))
    total_steps = 610 if args.mode == "smoke" else cfg.steps
    train_s = eval_s = save_s = 0.
    evaluations = {}
    growth_events = 0
    model.train()
    with (out/"steps.jsonl").open("w") as fp:
        for row in manifest[:total_steps]:
            t = row["step"]
            torch.cuda.synchronize()
            tick = time.perf_counter()
            batch, _, plans = inputs.make(row["views"], row["ray_seed"], "point_ssim")
            # Keep even the redundant baseline hooks until parity is measured.
            runner.native(); T.M.clear(runner)
            values = update(model, strategy, dataset, batch, plans, t, cfg,
                            loss_backend=C.loss, feature_step_condition=T.B.check_step_condition)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-tick
            train_s += elapsed
            if not all(math.isfinite(v) for v in values.values()) or not 0 < values["N"] <= cfg.cap:
                raise FloatingPointError(values)
            growth_events += int(values["delta_N"] > 0)
            fp.write(json.dumps(dict(step=t+1, seconds=elapsed, **values))+"\n")
            if t == 0:
                S.write(out/"first_input.json", dict(hash=S.digest(
                    {k:getattr(batch,k) for k in ("rays_ori","rays_dir","rgb_gt")})))
            del batch, plans
            if (t+1)%50 == 0 or t+1 == total_steps:
                fp.flush()
                progress = dict(step=t+1,total_steps=total_steps,training_seconds=train_s,**values)
                S.write(out/"progress.json",progress)
                print("PROGRESS",json.dumps(progress),flush=True)
            if args.mode == "train" and t+1 in (7000,15000,30000):
                model.eval();gc.collect();torch.cuda.empty_cache();tick=time.perf_counter()
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    ev=S.full_test(model,backend_config)
                eval_s += time.perf_counter()-tick
                evaluations[str(t+1)] = ev
                S.write(out/f"evaluation_{t+1}.json",ev);model.train()
            if args.mode == "train" and (t+1)%1000 == 0:
                tick=time.perf_counter()
                T.save(model,strategy,out,t+1,train_s,eval_s,save_s,manifest_hash)
                save_s += time.perf_counter()-tick
    if args.mode == "smoke" and not growth_events:
        raise RuntimeError("Smoke check did not exercise MCMC growth")
    S.write(out/"complete.json",dict(passed=True,steps=total_steps,N=model.num_gaussians,
            training_seconds=train_s,evaluation_seconds=eval_s,checkpoint_seconds=save_s,
            evaluations=evaluations,growth_events=growth_events))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--legacy-entry",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--mode",choices=("smoke","train"),default="smoke")
    parser.add_argument("--input-mode",choices=("legacy","selective_host","resident_uint8"),default="legacy")
    parser.add_argument("--gpu-image-cache-mib",type=int,default=1200)
    parser.add_argument("--manifest",type=Path,help="Reuse a saved full-horizon training_manifest.json for paired runs")
    main(parser.parse_args())
