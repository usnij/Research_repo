"""Standard-library helpers for immutable, one-process-per-GPU benchmarks."""
from pathlib import Path
import hashlib
import json
import math
import os
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def relative_file(root, name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError(f'Unsafe relative path: {name}')
    result = (Path(root) / p).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError(f'Path escapes root: {name}')
    return result


def verify_files(root, records):
    for row in records:
        p = relative_file(root, row['path'])
        if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError(f'File changed: {p}')


def verify_complete(out, expected_steps=30000):
    out = Path(out)
    c = read(out / 'complete.json')
    p = read(out / 'protocol.json')
    if not c.get('passed') or c['steps'] != expected_steps or not c.get('from_scratch'):
        raise ValueError('Incomplete or resumed run is not a completed from-scratch benchmark')
    if c['N'] != p['method']['cap']:
        raise ValueError('Final count does not match the input-derived target')
    metrics = {}
    for step in (7000, 15000, expected_steps):
        e = read(out / f'evaluation_{step}.json')
        if e['views'] <= 0 or e['views'] != len(e['per_view']):
            raise ValueError('Evaluation view coverage is incomplete')
        for row in [e, *e['per_view']]:
            if not all(math.isfinite(row[k]) for k in ('PSNR', 'SSIM', 'LPIPS')):
                raise ValueError('Non-finite evaluation metric')
        metrics[str(step)] = {k: e[k] for k in ('PSNR', 'SSIM', 'LPIPS', 'views', 'N')}
    return dict(complete=c, metrics=metrics)


def wait_for_ack(path, expected_signature, poll=15):
    """Block between timed jobs until the collector verifies retained files."""
    path = Path(path)
    while not path.exists():
        time.sleep(poll)
    d = read(path)
    if not d.get('verified') or d.get('artifact_signature') != expected_signature:
        raise ValueError('Archive acknowledgement does not match this run')
