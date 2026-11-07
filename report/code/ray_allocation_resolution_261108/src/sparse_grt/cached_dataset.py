"""Decode each training image once instead of once per iteration.

`ColmapDataset.__getitem__` calls `Image.open` every time it is asked for a view,
so a 3118x2078 JPEG is decoded on the main thread on every training step. That
costs 73.5 ms. At dense speed (770 ms/step) it is 10 percent and invisible; with
sparse rays the step is 115 ms and the decode becomes **64 percent of it**. The
renderer optimisation moved the bottleneck into the data pipeline.

Caching the decoded `uint8` tensors costs 255 x 6.48M x 3 = 4.96 GB of host RAM
for Bonsai at full resolution, which fits, and takes the sparse step to roughly
45 ms. Poses and intrinsics are kept alongside; they are negligible.

This wraps rather than patches `ColmapDataset` so nothing outside these
experiments changes behaviour.
"""

from __future__ import annotations

import time


class CachedDataset:
    """A `ColmapDataset` whose decoded images are held in host memory.

    Everything except `__getitem__` is delegated, so this is a drop-in wherever
    the dataset is used, including `get_gpu_batch_with_intrinsics`.
    """

    def __init__(self, dataset, eager: bool = True, verbose: bool = True):
        self.dataset = dataset
        self._cache: dict[int, dict] = {}
        if eager:
            start = time.time()
            total = 0
            for index in range(len(dataset)):
                item = dataset[index]
                self._cache[index] = item
                total += item["data"].numel()
            if verbose:
                print(
                    f"cached {len(dataset)} views, {total / 1e9:.2f} GB, "
                    f"{time.time() - start:.0f}s",
                    flush=True,
                )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self._cache.get(index)
        if item is None:
            item = self.dataset[index]
            self._cache[index] = item
        return item

    def __getattr__(self, name):
        # Only reached for attributes this class does not define, so the cache
        # and the wrapped dataset never shadow each other.
        return getattr(self.dataset, name)
