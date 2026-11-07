"""Audit real-scene inputs on CPU and prepare content-checked per-scene bundles.

Run in the reference JPEG/Python environment. No GPU or training is used.
The local data-roots file is private deployment input, not part of the release.
"""
import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
import struct
import tarfile

from benchmark_common import read, sha, signature, write


def prepare_scene(name, spec, base, scene, output):
    from PIL import Image, features
    from sparse_grt.config import MethodConfig
    from sparse_grt.contracts import ray_geometry
    from sparse_grt.budget import estimate
    from sparse_grt_backend.datasets.dataset_colmap import ColmapDataset
    if features.version('jpg') != '9.0' or features.check('libjpeg_turbo'):
        raise ValueError('Use the reference IJG JPEG environment before freezing inputs')
    scene = Path(scene).resolve()
    method = MethodConfig(**(base | {'block': spec['block']}))
    dataset = ColmapDataset(str(scene), device='cpu', downsample_factor=spec['downsample'],
                            split='train', test_split_interval=8)
    validation = ColmapDataset(str(scene), device='cpu', downsample_factor=spec['downsample'],
                               split='val', test_split_interval=8)
    paths = [Path(p) for p in dataset.image_paths]
    tests = [Path(p) for p in validation.image_paths]
    if len(paths) < method.views or not tests:
        raise ValueError('Insufficient train/test views')
    geometry, files = [], []
    for p in paths + tests:
        if p.with_name(p.stem + '_mask.png').exists():
            raise ValueError(f'Mask adapter has not been validated: {p}')
        with Image.open(p) as im:
            if im.mode != 'RGB':
                raise ValueError(f'Expected RGB image: {p}: {im.mode}')
            w, h = im.size
        geometry.append(ray_geometry(h, w, method))
        files.append(p)
    if len({g['rays_per_view'] for g in geometry[:len(paths)]}) != 1:
        raise ValueError('Unequal training ray counts require a new adapter, not silent resizing')
    for field in ('cameras.bin', 'images.bin', 'points3D.bin'):
        files.append(scene / 'sparse/0' / field)
    with (scene/'sparse/0/points3D.bin').open('rb') as f:
        initial, = struct.unpack('<Q', f.read(8))
    budget = estimate(paths, method='fine_tracks_v1', images_bin=scene/'sparse/0/images.bin',
                      view_limit=None)
    if budget['cap'] < initial:
        raise ValueError('Input budget below initialization; no implicit pruning')
    method = replace(method, cap=budget['cap'])
    records = [dict(path=str(p.relative_to(scene)), bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(set(files))]
    scene_dir = output/'scenes'/name
    scene_dir.mkdir(parents=True, exist_ok=True)
    for arm in ('structure', 'l1'):
        m = replace(method, structure_weight=.2 if arm == 'structure' else 0.)
        write(scene_dir/f'{arm}.json', vars(m))
    write(scene_dir/'budget_reference.json', budget)
    write(scene_dir/'files.json', records)
    g = geometry[0]
    info = dict(scene=name, **spec, train_views=len(paths), test_views=len(tests),
                initial_N=initial, target_N=method.cap,
                train_images=[str(p.relative_to(scene)) for p in paths],
                test_images=[str(p.relative_to(scene)) for p in tests],
                shapes=dict(Counter(f"{x['width']}x{x['height']}" for x in geometry)),
                rays_per_update=g['total_rays_if_equal_views'],
                nominal_ray_fraction=method.nominal_total_ray_fraction(),
                actual_ray_fraction=g['actual_fraction_of_one_view'],
                input_bytes=sum(r['bytes'] for r in records),
                cache_bytes=sum(x['full_pixels']*3 for x in geometry[:len(paths)]),
                files_signature=signature(records),
                scheduling_weight=method.cap*g['total_rays_if_equal_views'])
    write(scene_dir/'audit.json', info)
    archive = output/'assets'/f'{name}.tar.gz'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise FileExistsError(archive)
    pending = archive.with_suffix('.pending')
    with tarfile.open(pending, 'w:gz', compresslevel=1) as tar:
        for r in records:
            tar.add(scene/r['path'], arcname=f"{name}/{r['path']}", recursive=False)
    pending.replace(archive)
    info['archive'] = dict(path=f'assets/{archive.name}', sha256=sha(archive), bytes=archive.stat().st_size)
    write(scene_dir/'audit.json', info)
    return info


def allocate(rows, nodes):
    """Static balance by cap*rays, not a promised runtime model."""
    queues = [[] for _ in range(nodes)]
    costs = [0] * nodes
    for r in sorted(rows, key=lambda x: x['scheduling_weight'], reverse=True):
        i = min(range(nodes), key=costs.__getitem__)
        queues[i].append(r['scene']); costs[i] += r['scheduling_weight']
    return [{'node':f'gpu{i+1:02}', 'scenes':q, 'cost_proxy':costs[i]}
            for i,q in enumerate(queues)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--data-roots', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--nodes', type=int, default=8)
    a = p.parse_args()
    if a.nodes < 1:
        p.error('nodes must be positive')
    spec, roots = read(a.spec), read(a.data_roots)
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output/'spec.json').exists() and read(a.output/'spec.json') != spec:
        raise ValueError('Specification changed; prepare a new directory')
    write(a.output/'spec.json', spec)
    rows = []
    for name, info in spec['scenes'].items():
        saved = a.output/'scenes'/name/'audit.json'
        if saved.exists() and 'archive' in read(saved):
            result = read(saved)
            if sha(a.output/result['archive']['path']) != result['archive']['sha256']:
                raise ValueError('Previously prepared archive changed')
        else:
            result = prepare_scene(name, info, spec['method'], roots[name], a.output)
        rows.append(result)
        print(name, result['target_N'], result['rays_per_update'], 'ready', flush=True)
        write(a.output/'preparation_progress.json', dict(completed=[r['scene'] for r in rows]))
    write(a.output/'scene_index.json', rows)
    write(a.output/'allocation.json', allocate(rows, a.nodes))
    write(a.output/'ready.json', dict(passed=True, scenes=len(rows),
          spec_signature=signature(spec), total_input_bytes=sum(r['input_bytes'] for r in rows)))


if __name__ == '__main__':
    main()
