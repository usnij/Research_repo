"""Versioned input budgets. Neither formula estimates an optimal/minimal count.

The continuous formula is the frozen nine-scene exploratory calibration. It
uses training RGB AND SfM incidence, not RGB alone. Missing tracks are errors.
No quality labels, test RGB, fitting, or scene-name branches enter inference.
"""
import hashlib
import math
from pathlib import Path
import struct
import time

import numpy as np


FROZEN_FORMULA = dict(log_intercept=15.009178414327616,
                      log_fine=1.1232071568673763,
                      short_track_fraction=3.7407898766829666)


def continuous_count(fine, short_tracks):
    if not math.isfinite(fine) or fine <= 0:
        raise ValueError("The continuous budget is undefined for zero fine-band energy")
    if short_tracks is None or not math.isfinite(short_tracks) or not 0 <= short_tracks <= 1:
        raise ValueError("Measured training SfM tracks are required; missing does not mean zero")
    a = FROZEN_FORMULA
    value = math.exp(a['log_intercept'] + a['log_fine']*math.log(fine)
                     + a['short_track_fraction']*short_tracks)
    return max(1, round(value)), value


def fine_fraction(path):
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    with Image.open(path) as im:
        rgba = np.asarray(im.convert('RGBA'), dtype=np.float32)/255.
    y = rgba[..., :3] @ np.array([.299, .587, .114], np.float32)
    mask = rgba[..., 3]
    h, w = y.shape
    ratio = min(1., 512/max(h, w))
    size = (max(1, round(w*ratio)), max(1, round(h*ratio)))
    def resize(a):
        return np.asarray(Image.fromarray(a.astype(np.float32)).resize(size, Image.Resampling.BOX),
                          dtype=np.float32)
    m = resize(mask)
    y = resize(y*mask)/np.maximum(m, 1e-8)
    previous, bands = y, []
    for radius in (1, 2, 4, 8):
        smooth = gaussian_filter(y*m, radius, mode='reflect')/np.maximum(
            gaussian_filter(m, radius, mode='reflect'), 1e-8)
        bands.append(float(np.sum(np.abs(previous-smooth)*m)/max(float(m.sum()), 1e-8)))
        previous = smooth
    return bands[0]/max(sum(bands), 1e-8)


def haar_fraction(path):
    from PIL import Image
    with Image.open(path) as im:
        if 'A' in im.getbands():
            im = Image.alpha_composite(Image.new('RGBA', im.size, (0, 0, 0, 255)),
                                      im.convert('RGBA')).convert('RGB')
        else:
            im = im.convert('RGB')
        im.thumbnail((512, 512), Image.Resampling.BOX)
        y = (np.asarray(im, dtype=np.float64)/255.) @ np.array([.299, .587, .114])
    h, w = y.shape
    y = y[:h//32*32, :w//32*32]
    if min(y.shape) < 32:
        raise ValueError("Five-level Haar requires dimensions of at least 32")
    a, details = y, []
    for _ in range(5):
        p, q, r, s = a[::2, ::2], a[::2, 1::2], a[1::2, ::2], a[1::2, 1::2]
        details.extend([(p+q-r-s).ravel()/2, (p-q+r-s).ravel()/2, (p-q-r+s).ravel()/2])
        a = (p+q+r+s)/2
    energy = np.sort(np.concatenate(details)**2)
    discard = int(np.searchsorted(np.cumsum(energy), .02**2*y.size, side='right'))
    return (a.size+len(energy)-discard)/y.size


def short_track_fraction(images_bin, training_paths):
    """Fraction of points seen by one or two DISTINCT TRAINING cameras."""
    wanted = {Path(p).name for p in training_paths}
    if len(wanted) != len(training_paths):
        raise ValueError("Duplicate training basenames require an explicit COLMAP name mapping")
    seen, lists = set(), []
    with Path(images_bin).open('rb') as f:
        def read(n):
            value = f.read(n)
            if len(value) != n:
                raise ValueError("Truncated COLMAP images.bin")
            return value
        count, = struct.unpack('<Q', read(8))
        for _ in range(count):
            read(64)
            name = bytearray()
            while (b := read(1)) != b'\x00':
                name.extend(b)
            name = Path(name.decode()).name
            n, = struct.unpack('<Q', read(8))
            records = read(n*24)
            if name not in wanted:
                continue
            if name in seen:
                raise ValueError("Ambiguous COLMAP image basename")
            seen.add(name)
            ids = np.frombuffer(records, dtype=np.dtype([('x', '<f8'), ('y', '<f8'), ('id', '<i8')]))['id']
            lists.append(np.unique(ids[ids >= 0]))
    if seen != wanted:
        raise ValueError(f"Missing training camera records: {sorted(wanted-seen)[:5]}")
    _, counts = np.unique(np.concatenate(lists), return_counts=True)
    if not len(counts):
        raise ValueError("No observed SfM points; continuous budget unavailable")
    return float((counts <= 2).mean())


def estimate(training_paths, *, method, images_bin=None, view_limit=16):
    start = time.perf_counter()
    paths = list(map(Path, training_paths))
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Unique training paths are required")
    if method not in ('fine_tracks_v1', 'haar_v1'):
        raise ValueError("Choose an explicit, versioned budget formula")
    if view_limit is None or len(paths) <= view_limit:
        selected = paths
    else:
        if view_limit < 1:
            raise ValueError("view_limit must be positive or None")
        # Preserve the original fixed-16 selection, including integer flooring.
        indices = np.linspace(0, len(paths)-1, view_limit, dtype=int)
        selected = [paths[i] for i in indices]
    fn = fine_fraction if method == 'fine_tracks_v1' else haar_fraction
    rows = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), value=fn(p))
            for p in selected]
    value = float(np.mean([r['value'] for r in rows]))
    u = None
    if method == 'fine_tracks_v1':
        if images_bin is None:
            raise ValueError("images.bin is required")
        u = short_track_fraction(images_bin, paths)
        cap, raw = continuous_count(value, u)
    else:
        cap = 1000000 if value < .20 else 2000000 if value < .25 else 3000000
        raw = float(cap)
    return dict(schema=1, method=method, cap=cap, unrounded_cap=raw,
                descriptor=value, short_track_fraction=u, image_views=len(rows),
                train_views=len(paths), inputs=rows, seconds=time.perf_counter()-start,
                sfm_sha256=hashlib.sha256(Path(images_bin).read_bytes()).hexdigest()
                    if method == 'fine_tracks_v1' else None,
                coefficients=FROZEN_FORMULA if method == 'fine_tracks_v1' else None,
                calibration='Nine previously inspected MipNeRF360 historical budgets; not independent validation'
                    if method == 'fine_tracks_v1' else 'Fixed empirical three-bin map',
                test_RGB_used=False)
