#!/usr/bin/env python3
"""Reevaluate a saved GRT checkpoint with the production full-view metrics.

This entry point only restores and evaluates a model. It deliberately delegates
metric computation to sparse_grt.standalone_backend.evaluate, the evaluator used
for the existing benchmark, rather than defining another SSIM or LPIPS variant.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True,
                                       stderr=subprocess.DEVNULL, timeout=15).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def compare_reference(result: dict, reference: dict) -> dict:
    """Use the professor's decimal precision; retain all original values."""
    digits = {"PSNR": 3, "SSIM": 4, "LPIPS": 4}
    checks = {}
    for key, places in digits.items():
        actual, expected = float(result[key]), float(reference[key])
        if not math.isfinite(actual) or not math.isfinite(expected):
            raise ValueError(f"Non-finite reference or result: {key}")
        checks[key] = dict(actual=actual, expected=expected,
                           difference=actual - expected, decimal_places=places,
                           actual_rounded=f"{actual:.{places}f}",
                           expected_rounded=f"{expected:.{places}f}",
                           matched=f"{actual:.{places}f}" == f"{expected:.{places}f}")
    counts_match = all(result[key] == reference[key] for key in ("N", "views"))
    # A metric coincidence must not conceal a different validation split/order.
    actual_images = [Path(row["image"]).name for row in result["per_view"]]
    expected_images = [Path(row["image"]).name for row in reference["per_view"]]
    view_order_match = actual_images == expected_images
    return dict(passed=counts_match and view_order_match and
                all(item["matched"] for item in checks.values()),
                counts_match=counts_match, view_order_match=view_order_match,
                metrics=checks)


def write_new_json(path: Path, value: dict) -> None:
    """Publish a complete JSON without replacing any existing result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic publication; fails if path exists.
    finally:
        temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--data-root", type=Path, required=True,
                        help="COLMAP scene root containing images_<DS> and sparse/0")
    result.add_argument("--output", type=Path, required=True,
                        help="New JSON file; existing results are never overwritten")
    result.add_argument("--config", type=Path,
                        help="Optional backend YAML/JSON overrides merged into checkpoint config")
    result.add_argument("--run-dir", type=Path,
                        help="Original run directory for provenance; defaults to checkpoint parent")
    result.add_argument("--run-id", help="Result run ID; defaults to run directory name")
    result.add_argument("--reference", type=Path,
                        help="Existing evaluation JSON for rounded metric/split comparison")
    result.add_argument("--expected-checkpoint-sha256")
    result.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    result.add_argument("--eval-downsample", type=int,
                        help="Render saved Gaussians directly at this dataset resolution")
    result.add_argument("--residual-maps", type=Path,
                        help="Save exact 8x8 test RGB absolute-residual maps, outside training time")
    return result


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in ("checkpoint", "data_root", "output", "config", "run_dir", "reference"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    for name in ("checkpoint", "config", "reference"):
        value = getattr(args, name)
        if value is not None and not value.is_file():
            raise ValueError(f"{name} is not a file: {value}")
    if not args.data_root.is_dir():
        raise ValueError(f"data-root is not a directory: {args.data_root}")
    if args.run_dir is None:
        args.run_dir = args.checkpoint.parent
    if not args.run_dir.is_dir():
        raise ValueError(f"run-dir is not a directory: {args.run_dir}")
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite output: {args.output}")
    if args.device < 0:
        raise ValueError("device must be nonnegative")
    if args.eval_downsample is not None and args.eval_downsample < 1:
        raise ValueError("eval-downsample must be positive")
    if args.expected_checkpoint_sha256 is not None:
        value = args.expected_checkpoint_sha256.lower()
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("expected-checkpoint-sha256 must contain 64 hexadecimal characters")
        args.expected_checkpoint_sha256 = value
    return args


def evaluate(args: argparse.Namespace) -> dict:
    # Lazy imports keep --help and argument validation usable without CUDA or
    # the renderer installed. Prefer the source tree next to this entry point.
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    import torch
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("Full-view GRT evaluation requires a CUDA GPU")
    if args.device >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {args.device} is unavailable")
    torch.cuda.set_device(args.device)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not OmegaConf.has_resolver("int_list"):
        OmegaConf.register_new_resolver("int_list", lambda values: list(map(int, values)))

    from sparse_grt import standalone_backend as backend
    from sparse_grt.provenance import loaded_runtime

    checkpoint_sha = sha256(args.checkpoint)
    if args.expected_checkpoint_sha256 and checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError("Checkpoint SHA256 differs from --expected-checkpoint-sha256")
    # Only use checkpoints produced by the trusted training pipeline: they
    # contain OmegaConf and RNG state, so weights_only=True is not compatible.
    saved = torch.load(str(args.checkpoint), map_location="cpu", mmap=True, weights_only=False)
    required = set(backend.KEYS) | {"config", "global_step", "scene_extent", "background",
                                    "n_active_features", "max_n_features"}
    if required - saved.keys():
        raise ValueError(f"Checkpoint is missing fields: {sorted(required - saved.keys())}")
    cfg = OmegaConf.create(OmegaConf.to_container(saved["config"], resolve=False))
    OmegaConf.set_struct(cfg, False)
    checkpoint_config = OmegaConf.to_container(cfg, resolve=True)
    override = None
    if args.config is not None:
        override = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(cfg, override)
    training_downsample = int(saved['config']['dataset']['downsample_factor'])
    if args.eval_downsample is not None:
        cfg.dataset.downsample_factor = args.eval_downsample
    cfg.path = str(args.data_root)
    cfg.render.enable_kernel_timings = False
    if cfg.render.method != "3dgrt" or cfg.dataset.type != "colmap":
        raise ValueError("This evaluator supports GRT checkpoints and COLMAP datasets")
    # Match the production restore used by the existing frozen-checkpoint
    # probes. The optimizer stays on CPU and is not restored for evaluation.
    for key in backend.KEYS:
        saved[key] = torch.nn.Parameter(saved[key].cuda(), requires_grad=False)
    model = backend.MixtureOfGaussians(cfg, scene_extent=saved["scene_extent"])
    model.init_from_checkpoint(saved, setup_optimizer=False)
    model.cuda().eval()
    model.requires_grad_(False)
    runner = backend.Runner(model)
    model.build_acc(rebuild=True)
    step = int(saved["global_step"])
    has_optimizer = bool(saved.get("optimizer", {}).get("state"))
    del saved
    gc.collect()
    runtime = backend.load()
    torch.cuda.synchronize()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    result = backend.evaluate(runtime, model, cfg, residual_map_dir=args.residual_maps)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    # Keep the production keys at the top level for existing table collectors.
    result.update(
        schema_version=1, run_id=args.run_id or args.run_dir.name,
        run_dir=str(args.run_dir), checkpoint=str(args.checkpoint),
        checkpoint_sha256=checkpoint_sha, checkpoint_bytes=args.checkpoint.stat().st_size,
        step=step, optimizer_state_present=has_optimizer,
        evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
        evaluation_seconds=elapsed, data_root=str(args.data_root),
        train_downsample=training_downsample,
        eval_downsample=int(cfg.dataset.downsample_factor),
        peak_torch_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_torch_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        checkpoint_config=checkpoint_config,
        config_override=None if override is None else dict(
            path=str(args.config), sha256=sha256(args.config),
            values=OmegaConf.to_container(override, resolve=True)),
        evaluation_config=OmegaConf.to_container(cfg, resolve=True),
        metric_definition=dict(
            source="sparse_grt.standalone_backend.evaluate",
            aggregation="arithmetic mean of per-view metrics",
            rgb_clamp=[0, 1], psnr_mse_floor=1e-12,
            ssim="torchmetrics StructuralSimilarityIndexMeasure(data_range=1.0), defaults",
            lpips="torchmetrics LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)"),
        environment=dict(
            hostname=platform.node(), python=sys.version, executable=sys.executable,
            platform=platform.platform(), gpu=torch.cuda.get_device_name(args.device),
            cuda_device=args.device, torch=str(torch.__version__), torch_cuda=torch.version.cuda,
            gpu_inventory=command_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,power.limit",
                                          "--format=csv,noheader,nounits"]),
            git_commit=(command_output(["git", "rev-parse", "HEAD"], root)
                        or os.environ.get("SPARSE_GRT_FROZEN_COMMIT")),
            git_status=command_output(["git", "status", "--porcelain"], root),
            entrypoint_sha256=sha256(Path(__file__).resolve()),
            cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False,
            runtime=loaded_runtime()),
    )
    if args.reference is not None:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        result["reference_comparison"] = dict(
            path=str(args.reference), sha256=sha256(args.reference),
            **compare_reference(result, reference))
    # Hold renderer buffer ownership until all evaluation work has completed.
    del runner
    return result


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    try:
        args = validate_args(cli.parse_args(argv))
    except ValueError as error:
        cli.error(str(error))
    result = evaluate(args)
    write_new_json(args.output, result)
    print(f"{result['run_id']}: G={result['N']:,} PSNR={result['PSNR']:.3f} "
          f"SSIM={result['SSIM']:.4f} LPIPS={result['LPIPS']:.4f}", flush=True)
    print(f"Full-precision results: {args.output}", flush=True)
    comparison = result.get("reference_comparison")
    if comparison is not None:
        print(f"Reference comparison: {'PASS' if comparison['passed'] else 'FAIL'}", flush=True)
        return 0 if comparison["passed"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
