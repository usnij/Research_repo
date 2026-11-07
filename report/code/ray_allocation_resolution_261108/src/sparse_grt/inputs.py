"""COLMAP input adapter that gathers uint8 RGB before float conversion.

Optional full-view GPU residency is explicit and bounded. The fallback still
transfers a complete uint8 view, but never converts an unobserved RGB pixel.
Camera ray grids are reused from the dataset; they were already cached by the
old implementation. No camera model is replaced with a pinhole approximation.
"""
import torch
from .sampling import StratifiedGrid, MicropatchGrid
from .losses import coordinate_ssim
from .contracts import ray_geometry, validate_plans, candidate_radius


class SelectedInputs:
    def __init__(self, dataset, batch_factory, config, *, device="cuda",
                 resident_budget_bytes=0, weight_fn=coordinate_ssim.weights):
        if config.sampler == 'micropatch' and weight_fn is coordinate_ssim.weights:
            from .losses import micropatch_coordinate
            weight_fn = micropatch_coordinate.weights
        self.dataset, self.batch_factory, self.config = dataset, batch_factory, config
        self.device, self.weight_fn = torch.device(device), weight_fn
        self.items = {}
        for vid in range(len(dataset)):
            item = torch.utils.data.default_collate([dataset[vid]])
            if "mask" in item:
                raise ValueError("Masked datasets need a separately validated input adapter")
            if item["data"].dtype != torch.uint8 or item["data"].shape[-1] != 3:
                raise TypeError("Expected decoded uint8 RGB with exactly three channels")
            self.items[vid] = {k: v.pin_memory() if torch.is_tensor(v) and
                              self.device.type == "cuda" else v for k, v in item.items()}
        image_bytes = sum(v["data"].numel()*v["data"].element_size() for v in self.items.values())
        self.resident = resident_budget_bytes > 0 and image_bytes <= resident_budget_bytes
        self.resident_bytes = image_bytes if self.resident else 0
        self.images = {i: b["data"][0].to(self.device, non_blocking=True)
                       for i, b in self.items.items()} if self.resident else {}
        self.poses = {i: b["pose"][0].to(self.device, non_blocking=True)
                      for i, b in self.items.items()}
        self.intrinsics = {i: int(b["intr"][0].item()) for i, b in self.items.items()}
        self.cameras = dataset._lazy_worker_intrinsics_cache()
        if len(dataset) < config.views:
            raise ValueError("Fewer training views than requested distinct batch views")
        self.geometry = {}
        for vid,item in self.items.items():
            _,ro,rd,_ = self.cameras[self.intrinsics[vid]]
            h,w = rd.shape[1:3]
            if tuple(item['data'].shape) != (1,1,h,w,3) or ro.shape != rd.shape:
                raise ValueError(f"Image/camera ray grid mismatch in view {vid}")
            self.geometry[vid] = ray_geometry(h,w,config)
        if len({v['rays_per_view'] for v in self.geometry.values()}) != 1:
            raise ValueError("Unequal ray counts need a separately validated ragged-view adapter")
        self.grids = {}
        self._checked_plans = False
        self.identity = torch.eye(4, device=self.device)[None].repeat(config.views, 1, 1)

    def make(self, views, seed, auxiliary="point_ssim"):
        if auxiliary != "point_ssim" or len(views) != self.config.views:
            raise ValueError("Only the configured observed-ray view batch is supported")
        if len(set(map(int,views))) != len(views) or any(int(v) not in self.items for v in views):
            raise ValueError("Batch views must be distinct valid training indices")
        origins, directions, targets, plans = [], [], [], []
        offset = 0
        for j, vid in enumerate(views):
            vid = int(vid)
            _, ray_origin, ray_direction, _ = self.cameras[self.intrinsics[vid]]
            height, width = ray_direction.shape[1:3]
            key = (height, width)
            if key not in self.grids:
                grid_type = (MicropatchGrid if self.config.sampler == 'micropatch' else StratifiedGrid)
                self.grids[key] = grid_type(height, width, self.config.block, self.device)
            grid = self.grids[key]
            generator = torch.Generator(device=self.device).manual_seed(seed+104729*j)
            sample = grid.sample(generator)
            transform = self.poses[vid][0]
            origins.append((ray_origin[:,sample.y,sample.x] @ transform[:3,:3].T
                            + transform[:3,3]).reshape(1,1,-1,3).contiguous())
            directions.append((ray_direction[:,sample.y,sample.x] @ transform[:3,:3].T)
                              .reshape(1,1,-1,3).contiguous())
            image = self.images[vid] if self.resident else self.items[vid]["data"][0].to(
                self.device, non_blocking=True)
            target = (image[:,sample.y,sample.x]/255.0).reshape(1,1,-1,3)
            targets.append(target)
            h, w = grid.shape
            if self.config.structure_weight:
                weight_options = ({'candidate_radius': candidate_radius(self.config)}
                                  if self.config.sampler == 'micropatch' else {})
                weight, k = self.weight_fn(sample.y.reshape(h,w), sample.x.reshape(h,w),
                                           self.config.block, self.config.sigma, **weight_options)
            else:
                weight, k = None, 2*candidate_radius(self.config)+1
            count = sample.y.numel()
            plans.append(dict(gt=target.reshape(1,h,w,3), shape=(h,w), weights=weight,
                              k=k, start=offset, end=offset+count, y=sample.y, x=sample.x))
            if self.config.sampler == 'micropatch':
                plans[-1]['patch_indices'] = grid.patch_indices
            offset += count
        if len({v.shape for v in targets}) != 1:
            raise ValueError("This renderer adapter requires equal per-view ray counts")
        batch = self.batch_factory(rays_ori=torch.cat(origins).contiguous(),
                                   rays_dir=torch.cat(directions).contiguous(),
                                   rgb_gt=torch.cat(targets).contiguous(),
                                   T_to_world=self.identity, intrinsics=None)
        if not self._checked_plans:
            validate_plans(plans, offset, self.config.views, self.config.boundary)
            self._checked_plans = True
        return batch, None, plans
