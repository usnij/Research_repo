"""Frozen paper-v1 launcher. Default mode only verifies; no training starts.

Existing three-scene evidence is registered. New scenes/seeds/ablations require
an explicit protocol revision, not silently reusing a Counter preset.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sparse_grt.release import PaperRelease


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-name', choices=('counter', 'bonsai', 'bicycle'), required=True)
    p.add_argument('--arm', choices=('l1', 'structure'), default='structure')
    p.add_argument('--scene', type=Path, help='Dataset directory, required for smoke/train')
    p.add_argument('--output', type=Path, help='New directory, required for smoke/train')
    p.add_argument('--mode', choices=('check', 'smoke', 'train'), default='check')
    a = p.parse_args()
    release = ROOT / 'releases' / 'paper_v1_20261102'
    guard = PaperRelease(release, ROOT.parent)
    result = guard.verify()
    print(json.dumps(result, indent=2))
    if a.mode == 'check':
        return
    if a.scene is None or a.output is None:
        p.error('--scene and --output are required for smoke/train')
    preset = f'{a.scene_name}_{a.arm}'
    scene_config = guard.protocol['scenes'][a.scene_name]
    # The established main loop is kept intact. Guards run only outside its
    # timed steps, before the first optimizer update.
    from train_scene import main as train
    train(SimpleNamespace(
        output=a.output, scene=a.scene,
        legacy_entry=ROOT.parent / guard.protocol['legacy_entry'],
        config=release / 'configs' / f'{preset}.json',
        downsample=scene_config['DS'], budget='fine_tracks_v1', budget_views='all',
        growth='log_horizon', growth_end=5000, cache_mib=1200,
        mode=a.mode, manifest=None, keep_checkpoints='7000,15000,30000',
        paper_guard=guard, paper_preset=preset))


if __name__ == '__main__':
    main()
