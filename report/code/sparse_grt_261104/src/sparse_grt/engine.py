"""One update, with the exact established ordering of optimizer and MCMC.

Backend setup, evaluation and checkpoint I/O belong to an adapter. Keeping
the core update here prevents coefficient experiments from copying trainers.
"""
import torch
from .objective import objective


PARAMETER_NAMES = ("positions", "rotation", "scale", "density", "features_albedo", "features_specular")


def update(model, strategy, dataset, batch, plans, step, config, *, loss_backend,
           feature_step_condition, check_finite=True, objective_backend=None):
    model.optimizer.zero_grad(set_to_none=True)
    pred = model(batch, train=True, frame_id=step)["pred_rgb"]
    image_objective = objective if objective_backend is None else objective_backend
    total, parts = image_objective(pred, batch.rgb_gt, plans, model.get_density(), model.get_scale(),
                                  config, loss_backend)
    strategy.pre_backward(step, model.scene_extent, dataset, batch=batch)
    total.backward()
    # Preserve the old logging synchronization point until a separate timing
    # experiment validates deferred metric transfers.
    values = {k: float(v.detach()) for k, v in parts.items()}
    del pred, total, parts
    strategy.post_backward(step, model.scene_extent, dataset, batch=batch)
    if check_finite and step % 100 == 0:
        for key in PARAMETER_NAMES:
            grad = getattr(model, key).grad
            if grad is not None and not torch.isfinite(grad).all():
                raise FloatingPointError(f"Non-finite gradient in {key} at {step}")
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    model.scheduler_step(step)
    old_count = model.num_gaussians
    strategy.paired_step = step
    strategy.post_optimizer_step(step, model.scene_extent, dataset, batch=batch)
    if model.progressive_training and feature_step_condition(
            step, 0, 1000000, model.feature_dim_increase_interval):
        model.increase_num_active_features()
    model.build_acc(rebuild=True)
    return dict(**values, N=model.num_gaussians, delta_N=model.num_gaussians-old_count,
                SH_degree=model.n_active_features)
